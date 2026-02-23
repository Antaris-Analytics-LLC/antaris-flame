"""
Context selection strategies for determining what content to include.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
import json
import re


class ContextStrategy(ABC):
    """Base class for context selection strategies."""
    
    @abstractmethod
    def select_content(self, available_content: List[Dict], budget: int, query: Optional[str] = None) -> List[Dict]:
        """Select content items to include within budget.
        
        Args:
            available_content: List of content items with metadata
            budget: Token budget available
            query: Optional current query for relevance scoring
            
        Returns:
            List of selected content items
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get human-readable strategy name."""
        pass


class RecencyStrategy(ContextStrategy):
    """Select most recent content first."""
    
    def __init__(self, prefer_high_priority: bool = True):
        """Initialize recency strategy.
        
        Args:
            prefer_high_priority: Whether to prefer high priority items when tokens are equal
        """
        self.prefer_high_priority = prefer_high_priority
        self.priority_weights = {
            'critical': 4,
            'important': 3,
            'normal': 2,
            'optional': 1
        }
    
    def select_content(self, available_content: List[Dict], budget: int, query: Optional[str] = None) -> List[Dict]:
        """Select most recent content within budget."""
        if not available_content or budget <= 0:
            return []
        
        # Sort by recency (newest first) and priority as tiebreaker
        def sort_key(item):
            recency_score = item.get('added_at', 0)
            priority_weight = self.priority_weights.get(item.get('priority', 'normal'), 2)
            
            if self.prefer_high_priority:
                return (-priority_weight, -recency_score)  # Priority first, then recency
            else:
                return -recency_score
        
        sorted_content = sorted(available_content, key=sort_key)
        
        selected = []
        used_tokens = 0
        
        for item in sorted_content:
            item_tokens = item.get('tokens', 0)
            if used_tokens + item_tokens <= budget:
                selected.append(item)
                used_tokens += item_tokens
            else:
                continue  # skip oversized item; smaller ones later may still fit
        
        return selected
    
    def get_strategy_name(self) -> str:
        return "Recency Strategy"


class RelevanceStrategy(ContextStrategy):
    """Select content based on relevance to current query."""
    
    def __init__(self, min_score: float = 0.1, case_sensitive: bool = False):
        """Initialize relevance strategy.
        
        Args:
            min_score: Minimum relevance score to consider
            case_sensitive: Whether keyword matching is case-sensitive
        """
        self.min_score = min_score
        self.case_sensitive = case_sensitive
        self.priority_multipliers = {
            'critical': 2.0,
            'important': 1.5,
            'normal': 1.0,
            'optional': 0.8
        }
    
    def select_content(self, available_content: List[Dict], budget: int, query: Optional[str] = None) -> List[Dict]:
        """Select content based on relevance to query."""
        if not available_content or budget <= 0:
            return []
        
        if not query:
            # Fall back to priority-based selection if no query
            return self._select_by_priority(available_content, budget)
        
        # Score each content item by relevance
        scored_content = []
        query_keywords = self.extract_keywords(query)
        
        for item in available_content:
            relevance_score = self.calculate_relevance(item, query_keywords)
            priority_multiplier = self.priority_multipliers.get(item.get('priority', 'normal'), 1.0)
            final_score = relevance_score * priority_multiplier
            
            if final_score >= self.min_score:
                scored_item = item.copy()
                scored_item['relevance_score'] = final_score
                scored_content.append(scored_item)
        
        # Sort by relevance score (highest first)
        scored_content.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Select within budget
        selected = []
        used_tokens = 0
        
        for item in scored_content:
            item_tokens = item.get('tokens', 0)
            if used_tokens + item_tokens <= budget:
                selected.append(item)
                used_tokens += item_tokens
            else:
                break
        
        return selected
    
    def extract_keywords(self, query: str) -> Set[str]:
        """Extract keywords from query for relevance scoring."""
        # Simple word extraction - could be enhanced with stemming, etc.
        if not self.case_sensitive:
            query = query.lower()
        
        # Extract words, removing punctuation
        words = re.findall(r'\b\w+\b', query)
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = {word for word in words if len(word) > 2 and word not in stop_words}
        
        return keywords
    
    def calculate_relevance(self, item: Dict, query_keywords: Set[str]) -> float:
        """Calculate relevance score for an item against query keywords."""
        content = item.get('content', '')
        if not content or not query_keywords:
            return 0.0
        
        if not self.case_sensitive:
            content = content.lower()
        
        # Count keyword matches
        content_words = set(re.findall(r'\b\w+\b', content))
        matches = query_keywords.intersection(content_words)
        
        if not matches:
            return 0.0
        
        # Calculate score based on match ratio and frequency
        match_ratio = len(matches) / len(query_keywords)
        
        # Bonus for exact phrase matches
        exact_matches = 0
        for keyword in query_keywords:
            if keyword in content:
                exact_matches += content.count(keyword)
        
        # Normalize by content length to avoid bias toward longer content
        content_length = max(1, len(content.split()))
        frequency_score = exact_matches / content_length
        
        # Combine match ratio and frequency
        relevance_score = (match_ratio * 0.7) + (frequency_score * 0.3)
        
        return min(1.0, relevance_score)  # Cap at 1.0
    
    def _select_by_priority(self, available_content: List[Dict], budget: int) -> List[Dict]:
        """Fallback selection based on priority only."""
        priority_order = ['critical', 'important', 'normal', 'optional']
        
        # Group by priority
        by_priority = {p: [] for p in priority_order}
        for item in available_content:
            priority = item.get('priority', 'normal')
            if priority in by_priority:
                by_priority[priority].append(item)
        
        # Select in priority order
        selected = []
        used_tokens = 0
        
        for priority in priority_order:
            for item in by_priority[priority]:
                item_tokens = item.get('tokens', 0)
                if used_tokens + item_tokens <= budget:
                    selected.append(item)
                    used_tokens += item_tokens
                else:
                    break
            if used_tokens >= budget:
                break
        
        return selected
    
    def get_strategy_name(self) -> str:
        return "Relevance Strategy"


class HybridStrategy(ContextStrategy):
    """Combine recency and relevance strategies with configurable weights."""
    
    def __init__(self, recency_weight: float = 0.4, relevance_weight: float = 0.6, 
                 recency_strategy: Optional[RecencyStrategy] = None,
                 relevance_strategy: Optional[RelevanceStrategy] = None):
        """Initialize hybrid strategy.
        
        Args:
            recency_weight: Weight for recency scoring (0.0 - 1.0)
            relevance_weight: Weight for relevance scoring (0.0 - 1.0)
            recency_strategy: Custom recency strategy (optional)
            relevance_strategy: Custom relevance strategy (optional)
        """
        if abs(recency_weight + relevance_weight - 1.0) > 0.01:
            raise ValueError("Recency and relevance weights must sum to 1.0")
        
        self.recency_weight = recency_weight
        self.relevance_weight = relevance_weight
        self.recency_strategy = recency_strategy or RecencyStrategy()
        self.relevance_strategy = relevance_strategy or RelevanceStrategy()
    
    def select_content(self, available_content: List[Dict], budget: int, query: Optional[str] = None) -> List[Dict]:
        """Select content using hybrid scoring."""
        if not available_content or budget <= 0:
            return []
        
        # Calculate scores for each item
        scored_content = []
        max_recency = max((item.get('added_at', 0) for item in available_content), default=0)
        
        for item in available_content:
            # Recency score (normalized 0-1, newer is higher)
            recency_raw = item.get('added_at', 0)
            recency_score = recency_raw / max_recency if max_recency > 0 else 0
            
            # Relevance score (from relevance strategy)
            relevance_score = 0
            if query:
                query_keywords = self.relevance_strategy.extract_keywords(query)
                relevance_score = self.relevance_strategy.calculate_relevance(item, query_keywords)
            
            # Priority boost
            priority_multiplier = self.relevance_strategy.priority_multipliers.get(
                item.get('priority', 'normal'), 1.0
            )
            
            # Combined score
            hybrid_score = (
                (recency_score * self.recency_weight) + 
                (relevance_score * self.relevance_weight)
            ) * priority_multiplier
            
            scored_item = item.copy()
            scored_item['hybrid_score'] = hybrid_score
            scored_item['recency_component'] = recency_score
            scored_item['relevance_component'] = relevance_score
            scored_content.append(scored_item)
        
        # Sort by hybrid score (highest first)
        scored_content.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Select within budget
        selected = []
        used_tokens = 0
        
        for item in scored_content:
            item_tokens = item.get('tokens', 0)
            if used_tokens + item_tokens <= budget:
                selected.append(item)
                used_tokens += item_tokens
            else:
                break
        
        return selected
    
    def get_strategy_name(self) -> str:
        return f"Hybrid Strategy (recency: {self.recency_weight}, relevance: {self.relevance_weight})"


class BudgetStrategy(ContextStrategy):
    """Fit maximum content within token budget using multiple approaches."""
    
    def __init__(self, approach: str = 'balanced'):
        """Initialize budget strategy.
        
        Args:
            approach: Selection approach ('greedy', 'balanced', 'priority_first')
        """
        if approach not in ['greedy', 'balanced', 'priority_first']:
            raise ValueError(f"Unknown approach: {approach}. Use: greedy, balanced, priority_first")
        
        self.approach = approach
        self.priority_values = {
            'critical': 100,
            'important': 75,
            'normal': 50,
            'optional': 25
        }
    
    def select_content(self, available_content: List[Dict], budget: int, query: Optional[str] = None) -> List[Dict]:
        """Select content to maximize value within budget."""
        if not available_content or budget <= 0:
            return []
        
        if self.approach == 'greedy':
            return self._greedy_selection(available_content, budget)
        elif self.approach == 'balanced':
            return self._balanced_selection(available_content, budget)
        elif self.approach == 'priority_first':
            return self._priority_first_selection(available_content, budget)
        else:
            raise ValueError(f"Unknown approach: {self.approach}")
    
    def _greedy_selection(self, available_content: List[Dict], budget: int) -> List[Dict]:
        """Greedy selection by value-to-token ratio."""
        # Calculate value per token for each item
        valued_content = []
        for item in available_content:
            tokens = max(1, item.get('tokens', 1))  # Avoid division by zero
            priority_value = self.priority_values.get(item.get('priority', 'normal'), 50)
            value_per_token = priority_value / tokens
            
            valued_item = item.copy()
            valued_item['value_per_token'] = value_per_token
            valued_content.append(valued_item)
        
        # Sort by value per token (highest first)
        valued_content.sort(key=lambda x: x['value_per_token'], reverse=True)
        
        # Greedy selection — continue past oversized items so smaller ones can still fit
        selected = []
        used_tokens = 0
        
        for item in valued_content:
            item_tokens = item.get('tokens', 0)
            if used_tokens + item_tokens <= budget:
                selected.append(item)
                used_tokens += item_tokens
            else:
                continue  # smaller items later may still fit
        
        return selected
    
    def _balanced_selection(self, available_content: List[Dict], budget: int) -> List[Dict]:
        """Balanced selection considering both priority and recency."""
        # Score items by combined priority and recency
        max_added_at = max((item.get('added_at', 0) for item in available_content), default=1)
        
        scored_content = []
        for item in available_content:
            priority_value = self.priority_values.get(item.get('priority', 'normal'), 50)
            recency_score = (item.get('added_at', 0) / max_added_at) * 50  # Scale to 0-50
            combined_score = priority_value + recency_score
            
            scored_item = item.copy()
            scored_item['combined_score'] = combined_score
            scored_content.append(scored_item)
        
        # Sort by combined score
        scored_content.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Select within budget
        selected = []
        used_tokens = 0
        
        for item in scored_content:
            item_tokens = item.get('tokens', 0)
            if used_tokens + item_tokens <= budget:
                selected.append(item)
                used_tokens += item_tokens
        
        return selected
    
    def _priority_first_selection(self, available_content: List[Dict], budget: int) -> List[Dict]:
        """Select all high priority items first, then fill remaining budget."""
        priority_order = ['critical', 'important', 'normal', 'optional']
        
        # Group by priority
        by_priority = {p: [] for p in priority_order}
        for item in available_content:
            priority = item.get('priority', 'normal')
            if priority in by_priority:
                # Sort by recency within priority level
                by_priority[priority].append(item)
        
        # Sort each priority group by recency
        for priority in priority_order:
            by_priority[priority].sort(key=lambda x: x.get('added_at', 0), reverse=True)
        
        # Select in priority order
        selected = []
        used_tokens = 0
        
        for priority in priority_order:
            for item in by_priority[priority]:
                item_tokens = item.get('tokens', 0)
                if used_tokens + item_tokens <= budget:
                    selected.append(item)
                    used_tokens += item_tokens
                if used_tokens >= budget:
                    break
            if used_tokens >= budget:
                break
        
        return selected
    
    def get_strategy_name(self) -> str:
        return f"Budget Strategy ({self.approach})"