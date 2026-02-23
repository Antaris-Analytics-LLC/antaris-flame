"""
Context window management and tracking.
"""

from typing import Callable, Dict, List, Optional, Tuple
import json


class ContextWindow:
    """Represents a context window state with token tracking per section."""
    
    def __init__(self, total_budget: int = 8000):
        """Initialize context window with total token budget.
        
        Args:
            total_budget: Maximum tokens available for the entire context window
        """
        self.total_budget = total_budget
        self.sections = {
            'system': {'used': 0, 'budget': 0, 'content': []},
            'memory': {'used': 0, 'budget': 0, 'content': []},
            'conversation': {'used': 0, 'budget': 0, 'content': []},
            'tools': {'used': 0, 'budget': 0, 'content': []}
        }
        self.overflow_warnings = []
        # Pluggable token estimator — default: ~4 chars per token
        self._token_estimator: Callable[[str], int] = lambda text: max(1, len(text) // 4)

    def set_token_estimator(self, fn: Callable[[str], int]) -> None:
        """Replace the default character-based token estimator.

        Args:
            fn: Callable that accepts a string and returns an integer token
                count.  Example using tiktoken::

                    import tiktoken
                    enc = tiktoken.get_encoding("cl100k_base")
                    cm.window.set_token_estimator(lambda t: len(enc.encode(t)))
        """
        self._token_estimator = fn
        
    def set_section_budget(self, section: str, budget: int) -> None:
        """Set token budget for a specific section.
        
        Args:
            section: Section name (system, memory, conversation, tools)
            budget: Token budget for this section
        """
        if section not in self.sections:
            raise ValueError(f"Unknown section: {section}")
        self.sections[section]['budget'] = budget
        
    def add_content(self, section: str, content: str, priority: str = 'normal') -> bool:
        """Add content to a section and track token usage.
        
        Args:
            section: Target section
            content: Content to add
            priority: Priority level (critical, important, normal, optional)
            
        Returns:
            True if content fits within budget, False if truncated
        """
        if section not in self.sections:
            raise ValueError(f"Unknown section: {section}")
            
        tokens = self._estimate_tokens(content)
        section_data = self.sections[section]
        
        content_item = {
            'content': content,
            'tokens': tokens,
            'priority': priority,
            'added_at': len(section_data['content'])
        }
        
        section_data['content'].append(content_item)
        section_data['used'] += tokens
        
        # Check for overflow
        if section_data['used'] > section_data['budget'] > 0:
            overflow = section_data['used'] - section_data['budget']
            self.overflow_warnings.append({
                'section': section,
                'overflow': overflow,
                'content_id': len(section_data['content']) - 1
            })
            return False
            
        return True
        
    def get_total_used(self) -> int:
        """Get total tokens used across all sections."""
        return sum(section['used'] for section in self.sections.values())
        
    def get_total_budget(self) -> int:
        """Get total allocated budget across all sections."""
        return sum(section['budget'] for section in self.sections.values())
        
    def is_over_budget(self) -> bool:
        """Check if total usage exceeds total budget."""
        return self.get_total_used() > self.total_budget
        
    def get_section_utilization(self, section: str) -> float:
        """Get utilization percentage for a section.
        
        Args:
            section: Section name
            
        Returns:
            Utilization as percentage (0.0 - 1.0+)
        """
        if section not in self.sections:
            raise ValueError(f"Unknown section: {section}")
            
        section_data = self.sections[section]
        if section_data['budget'] == 0:
            return 0.0 if section_data['used'] == 0 else 1.0
        return section_data['used'] / section_data['budget']
        
    def get_usage_report(self) -> Dict:
        """Generate detailed usage report."""
        total_used = self.get_total_used()
        total_allocated = self.get_total_budget()
        
        report = {
            'total_budget': self.total_budget,
            'total_used': total_used,
            'total_allocated': total_allocated,
            'utilization': total_used / self.total_budget if self.total_budget > 0 else 0.0,
            'over_budget': self.is_over_budget(),
            'sections': {},
            'overflow_warnings': self.overflow_warnings
        }
        
        for section_name, section_data in self.sections.items():
            report['sections'][section_name] = {
                'budget': section_data['budget'],
                'used': section_data['used'],
                'available': max(0, section_data['budget'] - section_data['used']),
                'utilization': self.get_section_utilization(section_name),
                'content_count': len(section_data['content']),
                'over_budget': section_data['used'] > section_data['budget'] if section_data['budget'] > 0 else False
            }
            
        return report
        
    def clear_section(self, section: str) -> None:
        """Clear all content from a section."""
        if section not in self.sections:
            raise ValueError(f"Unknown section: {section}")
            
        self.sections[section]['used'] = 0
        self.sections[section]['content'] = []
        # Remove overflow warnings for this section
        self.overflow_warnings = [w for w in self.overflow_warnings if w['section'] != section]
        
    def get_section_content(self, section: str) -> List[Dict]:
        """Get all content items for a section."""
        if section not in self.sections:
            raise ValueError(f"Unknown section: {section}")
        return self.sections[section]['content']
        
    def remove_content_by_priority(self, section: str, priority: str) -> int:
        """Remove all content items with specific priority from a section.
        
        Args:
            section: Section name
            priority: Priority level to remove
            
        Returns:
            Number of tokens freed
        """
        if section not in self.sections:
            raise ValueError(f"Unknown section: {section}")
            
        section_data = self.sections[section]
        tokens_freed = 0
        new_content = []
        
        for item in section_data['content']:
            if item['priority'] == priority:
                tokens_freed += item['tokens']
            else:
                new_content.append(item)
                
        section_data['content'] = new_content
        section_data['used'] -= tokens_freed
        
        # Clean up overflow warnings for removed content
        self.overflow_warnings = [
            w for w in self.overflow_warnings 
            if not (w['section'] == section and w['content_id'] >= len(new_content))
        ]
        
        return tokens_freed
        
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using character-based approximation.
        
        IMPORTANT LIMITATION: This uses a simple character-based approximation
        of ~4 characters per token for English text. This is NOT accurate for 
        all providers or languages.
        
        For production use, consider implementing provider-aware token counting
        using actual tokenizer libraries (e.g., tiktoken for OpenAI, transformers
        for HuggingFace models, etc.). The current approximation is sufficient
        for basic budget planning but will not match exact provider counts.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count (minimum 1 for non-empty text)
        """
        if not text:
            return 0
        return self._token_estimator(text)
        
    def to_json(self) -> str:
        """Serialize window state to JSON."""
        data = {
            'total_budget': self.total_budget,
            'sections': {}
        }
        
        for section_name, section_data in self.sections.items():
            data['sections'][section_name] = {
                'budget': section_data['budget'],
                'used': section_data['used'],
                'content_count': len(section_data['content'])
            }
            
        return json.dumps(data, indent=2)
        
    @classmethod
    def from_json(cls, json_str: str) -> 'ContextWindow':
        """Create ContextWindow from JSON state (structure only, no content)."""
        data = json.loads(json_str)
        window = cls(total_budget=data['total_budget'])
        
        for section_name, section_data in data['sections'].items():
            if section_name in window.sections:
                window.sections[section_name]['budget'] = section_data['budget']
                window.sections[section_name]['used'] = section_data.get('used', 0)
                
        return window