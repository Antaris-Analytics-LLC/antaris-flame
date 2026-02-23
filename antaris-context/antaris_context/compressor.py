"""
Message compression utilities for reducing context size.
"""

import re
from typing import List, Dict, Optional, Tuple
import json


class MessageCompressor:
    """Compress messages to reduce token usage while preserving important information."""
    
    COMPRESSION_LEVELS = {
        'light': {
            'remove_empty_lines': True,
            'collapse_whitespace': True,
            'remove_redundant_spaces': False,
            'truncate_tool_outputs': False,
            'collapse_repeated_patterns': False,
            'aggressive_newlines': False
        },
        'moderate': {
            'remove_empty_lines': True,
            'collapse_whitespace': True,
            'remove_redundant_spaces': True,
            'truncate_tool_outputs': True,
            'collapse_repeated_patterns': True,
            'aggressive_newlines': False
        },
        'aggressive': {
            'remove_empty_lines': True,
            'collapse_whitespace': True,
            'remove_redundant_spaces': True,
            'truncate_tool_outputs': True,
            'collapse_repeated_patterns': True,
            'aggressive_newlines': True,
            'smart_sentence_selection': True
        }
    }
    
    def __init__(self, level: str = 'moderate'):
        """Initialize compressor with compression level.
        
        Args:
            level: Compression level (light, moderate, aggressive)
        """
        if level not in self.COMPRESSION_LEVELS:
            raise ValueError(f"Unknown compression level: {level}. Use: {list(self.COMPRESSION_LEVELS.keys())}")
        
        self.level = level
        self.config = self.COMPRESSION_LEVELS[level].copy()
        self.stats = {
            'original_length': 0,
            'compressed_length': 0,
            'bytes_saved': 0,
            'compression_ratio': 0.0
        }
    
    def compress(self, text: str) -> str:
        """Compress text according to configuration.
        
        Args:
            text: Input text to compress
            
        Returns:
            Compressed text
        """
        if not text:
            return text
            
        original_length = len(text)
        self.stats['original_length'] += original_length
        
        compressed = text
        
        if self.config['remove_empty_lines']:
            compressed = self._remove_empty_lines(compressed)
            
        if self.config['collapse_whitespace']:
            compressed = self._collapse_whitespace(compressed)
            
        if self.config['remove_redundant_spaces']:
            compressed = self._remove_redundant_spaces(compressed)
            
        if self.config['collapse_repeated_patterns']:
            compressed = self._collapse_repeated_patterns(compressed)
            
        if self.config['aggressive_newlines']:
            compressed = self._aggressive_newline_removal(compressed)
            
        if self.config.get('smart_sentence_selection', False):
            compressed = self._smart_sentence_selection(compressed)
        
        compressed_length = len(compressed)
        self.stats['compressed_length'] += compressed_length
        self.stats['bytes_saved'] += (original_length - compressed_length)
        
        if self.stats['original_length'] > 0:
            self.stats['compression_ratio'] = 1.0 - (self.stats['compressed_length'] / self.stats['original_length'])
            
        return compressed
    
    def compress_tool_output(self, output: str, max_lines: int = 50, keep_first: int = 25, keep_last: int = 25) -> str:
        """Compress tool output by keeping first and last N lines.
        
        Args:
            output: Tool output text
            max_lines: Maximum lines to keep
            keep_first: Number of lines to keep from start
            keep_last: Number of lines to keep from end
            
        Returns:
            Compressed tool output
        """
        if not output or not self.config['truncate_tool_outputs']:
            return output
            
        lines = output.strip().split('\n')
        
        if len(lines) <= max_lines:
            return output

        # Guard against overlap: if first+last covers all lines, return as-is
        if keep_first + keep_last >= len(lines):
            return "\n".join(lines)

        # Keep first N and last N lines
        first_part = lines[:keep_first]
        last_part = lines[-keep_last:] if keep_last > 0 else []
        
        # Add truncation marker
        truncated_count = len(lines) - keep_first - keep_last
        marker = f"\n... [truncated {truncated_count} lines] ...\n"
        
        if last_part:
            result = '\n'.join(first_part) + marker + '\n'.join(last_part)
        else:
            result = '\n'.join(first_part) + marker.rstrip()
            
        return result
    
    def compress_message_list(self, messages: List[Dict], max_content_length: int = 1000) -> List[Dict]:
        """Compress a list of messages.
        
        Args:
            messages: List of message dictionaries
            max_content_length: Maximum length for message content
            
        Returns:
            List of compressed messages
        """
        compressed_messages = []
        
        for message in messages:
            if isinstance(message, dict):
                compressed_msg = message.copy()
                
                # Compress content field
                if 'content' in message and isinstance(message['content'], str):
                    content = message['content']
                    
                    # Apply general compression
                    compressed_content = self.compress(content)
                    
                    # Handle tool outputs specifically
                    if message.get('role') == 'tool':
                        compressed_content = self.compress_tool_output(compressed_content)
                    
                    # Truncate if still too long
                    if len(compressed_content) > max_content_length:
                        compressed_content = self._smart_truncate(compressed_content, max_content_length)
                    
                    compressed_msg['content'] = compressed_content
                    
                compressed_messages.append(compressed_msg)
            else:
                compressed_messages.append(message)
                
        return compressed_messages
    
    def get_compression_stats(self) -> Dict:
        """Get compression statistics."""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset compression statistics."""
        self.stats = {
            'original_length': 0,
            'compressed_length': 0,
            'bytes_saved': 0,
            'compression_ratio': 0.0
        }
    
    def set_config(self, **kwargs) -> None:
        """Update compression configuration.
        
        Args:
            **kwargs: Configuration options to update
        """
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
            else:
                raise ValueError(f"Unknown configuration option: {key}")
    
    def _remove_empty_lines(self, text: str) -> str:
        """Remove empty lines from text."""
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        return '\n'.join(non_empty_lines)
    
    def _collapse_whitespace(self, text: str) -> str:
        """Collapse multiple whitespace characters into single spaces."""
        # Collapse multiple spaces, but preserve single newlines
        text = re.sub(r'[ \t]+', ' ', text)
        # Collapse multiple newlines
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()
    
    def _remove_redundant_spaces(self, text: str) -> str:
        """Remove spaces before punctuation and other redundant spacing."""
        # Remove spaces before punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        # Remove spaces after opening brackets (NOT quotes — would mangle JSON/strings)
        text = re.sub(r'([\[\(\{])\s+', r'\1', text)
        # Remove spaces before closing brackets (NOT quotes)
        text = re.sub(r'\s+([\]\)\}])', r'\1', text)
        return text
    
    def _collapse_repeated_patterns(self, text: str) -> str:
        """Collapse repeated patterns like "..." or "---"."""
        # Collapse multiple dots
        text = re.sub(r'\.{3,}', '...', text)
        # Collapse multiple dashes
        text = re.sub(r'-{3,}', '---', text)
        # Collapse multiple equals
        text = re.sub(r'={3,}', '===', text)
        # Collapse multiple asterisks
        text = re.sub(r'\*{3,}', '***', text)
        return text
    
    def _aggressive_newline_removal(self, text: str) -> str:
        """Aggressively remove newlines while preserving structure."""
        lines = text.split('\n')
        processed_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
                
            # Keep lines that look like headers or important structure
            if (stripped.startswith('#') or 
                stripped.startswith('*') or 
                stripped.startswith('-') or
                stripped.endswith(':') or
                stripped.isupper()):
                processed_lines.append(line)
            else:
                # For regular text, join with previous line if it doesn't end with punctuation
                if (processed_lines and 
                    not processed_lines[-1].rstrip().endswith(('.', '!', '?', ':', ';')) and
                    not stripped.startswith(('*', '-', '#'))):
                    processed_lines[-1] += ' ' + stripped
                else:
                    processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def _smart_truncate(self, text: str, max_length: int) -> str:
        """Truncate text intelligently at word boundaries."""
        if len(text) <= max_length:
            return text
            
        # Try to truncate at sentence boundary
        truncate_pos = max_length - 20  # Leave room for ellipsis
        
        # Look for sentence endings
        for punct in ['. ', '! ', '? ']:
            pos = text.rfind(punct, 0, truncate_pos)
            if pos > max_length * 0.7:  # Don't truncate too early
                return text[:pos + 1] + '...'
        
        # Fall back to word boundary
        pos = text.rfind(' ', 0, truncate_pos)
        if pos > max_length * 0.7:
            return text[:pos] + '...'
        
        # Hard truncate as last resort
        return text[:max_length - 3] + '...'
    
    def _smart_sentence_selection(self, text: str, target_ratio: float = 0.7) -> str:
        """Intelligently select most important sentences from text.
        
        Args:
            text: Input text
            target_ratio: Target compression ratio (0.7 = keep 70% of sentences)
            
        Returns:
            Text with less important sentences removed
        """
        if len(text) < 200:  # Don't process very short text
            return text
            
        # Split into sentences
        sentences = self._split_sentences(text)
        if len(sentences) <= 3:  # Keep very short texts intact
            return text
        
        # Score sentences by importance, tracking original index
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence_importance(sentence, i, len(sentences))
            sentence_scores.append((score, i, sentence))
        
        # Sort by score (highest first) and select top sentences
        sentence_scores.sort(key=lambda x: x[0], reverse=True)
        target_count = max(1, int(len(sentences) * target_ratio))
        selected = sentence_scores[:target_count]
        
        # Restore original order by index
        selected.sort(key=lambda x: x[1])
        return ' '.join(s[2] for s in selected)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting on periods, exclamations, questions
        # Handle common abbreviations to avoid false splits
        text = re.sub(r'\b(Mr|Mrs|Dr|Prof|etc|vs|Inc|Ltd)\.\s*', r'\1<DOT> ', text)
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Restore dots in abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        # Clean up and filter empty sentences
        return [s.strip() for s in sentences if s.strip()]
    
    def _score_sentence_importance(self, sentence: str, position: int, total_sentences: int) -> float:
        """Score a sentence for importance (higher = more important)."""
        score = 0.0
        
        # Position scoring - first and last sentences are more important
        if position == 0:
            score += 0.3  # First sentence bonus
        elif position == total_sentences - 1:
            score += 0.2  # Last sentence bonus
        
        # Length scoring - very short or very long sentences are less important
        length = len(sentence)
        if 20 <= length <= 150:
            score += 0.2
        elif length < 10:
            score -= 0.3  # Penalize very short sentences
        
        # Content scoring
        sentence_lower = sentence.lower()
        
        # Important indicators
        important_words = ['important', 'critical', 'error', 'warning', 'note', 
                          'remember', 'key', 'main', 'primary', 'essential']
        for word in important_words:
            if word in sentence_lower:
                score += 0.4
                break
        
        # Question sentences are often important
        if sentence.strip().endswith('?'):
            score += 0.3
        
        # Sentences with numbers or specific details
        if re.search(r'\d+', sentence):
            score += 0.2
            
        # Sentences with colons often introduce lists or explanations
        if ':' in sentence:
            score += 0.1
            
        # Penalize very common/filler words
        filler_phrases = ['i think', 'maybe', 'perhaps', 'it seems', 'probably']
        for phrase in filler_phrases:
            if phrase in sentence_lower:
                score -= 0.2
                break
        
        # Penalize sentences that are mostly punctuation or whitespace
        word_count = len(sentence.split())
        if word_count < 3:
            score -= 0.4
            
        return max(0.0, score)  # Ensure non-negative scores