"""
Context profiler for analyzing usage patterns and optimization suggestions.
"""

from typing import Dict, List, Optional, Tuple
import json
import re
import time
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)


class ContextProfiler:
    """Analyze context usage and provide optimization suggestions."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize context profiler.
        
        Args:
            log_file: Optional file path for logging analysis history
        """
        self.log_file = log_file
        self.analysis_history = []
        self.waste_thresholds = {
            'low_utilization': 0.3,      # Section using <30% of budget
            'high_overhead': 0.8,        # Content with >80% overhead (whitespace, etc.)
            'redundant_content': 0.7     # Similar content threshold
        }
    
    def analyze_window(self, window) -> Dict:
        """Perform comprehensive analysis of a context window.
        
        Args:
            window: ContextWindow instance to analyze
            
        Returns:
            Analysis report with findings and suggestions
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'overview': self._analyze_overview(window),
            'section_analysis': self._analyze_sections(window),
            'waste_detection': self._detect_waste(window),
            'optimization_suggestions': [],
            'efficiency_score': 0.0
        }
        
        # Generate optimization suggestions based on findings
        report['optimization_suggestions'] = self._generate_suggestions(report, window)
        
        # Calculate efficiency score
        report['efficiency_score'] = self._calculate_efficiency_score(report)
        
        # Log analysis if log file specified
        if self.log_file:
            self._log_analysis(report)
        
        self.analysis_history.append(report)
        
        return report
    
    def analyze_content_distribution(self, window) -> Dict:
        """Analyze how content is distributed across sections."""
        distribution = {
            'total_content_items': 0,
            'by_section': {},
            'by_priority': {'critical': 0, 'important': 0, 'normal': 0, 'optional': 0},
            'token_distribution': {}
        }
        
        for section_name, section_data in window.sections.items():
            content_items = section_data['content']
            distribution['by_section'][section_name] = {
                'count': len(content_items),
                'tokens': section_data['used'],
                'avg_tokens_per_item': section_data['used'] / max(1, len(content_items))
            }
            distribution['total_content_items'] += len(content_items)
            
            # Count by priority
            for item in content_items:
                priority = item.get('priority', 'normal')
                if priority in distribution['by_priority']:
                    distribution['by_priority'][priority] += 1
        
        # Calculate token distribution percentages
        total_tokens = window.get_total_used()
        for section_name, data in distribution['by_section'].items():
            percentage = (data['tokens'] / total_tokens * 100) if total_tokens > 0 else 0
            distribution['token_distribution'][section_name] = round(percentage, 1)
        
        return distribution
    
    def detect_redundant_content(self, window, similarity_threshold: float = 0.7) -> List[Dict]:
        """Detect potentially redundant content across sections.
        
        Args:
            window: ContextWindow to analyze
            similarity_threshold: Threshold for considering content similar
            
        Returns:
            List of redundant content groups
        """
        redundant_groups = []
        all_content = []
        
        # Collect all content with section info
        for section_name, section_data in window.sections.items():
            for i, item in enumerate(section_data['content']):
                all_content.append({
                    'section': section_name,
                    'index': i,
                    'content': item.get('content', ''),
                    'tokens': item.get('tokens', 0),
                    'priority': item.get('priority', 'normal')
                })
        
        # Compare all pairs
        checked_pairs = set()
        
        for i, item1 in enumerate(all_content):
            for j, item2 in enumerate(all_content[i+1:], i+1):
                if (i, j) in checked_pairs:
                    continue
                
                similarity = self._calculate_content_similarity(
                    item1['content'], item2['content']
                )
                
                if similarity >= similarity_threshold:
                    redundant_groups.append({
                        'similarity': similarity,
                        'items': [item1, item2],
                        'potential_savings': min(item1['tokens'], item2['tokens'])
                    })
                
                checked_pairs.add((i, j))
        
        return redundant_groups
    
    def suggest_budget_reallocation(self, window) -> Dict:
        """Suggest better budget allocation across sections."""
        current_usage = {}
        current_budgets = {}
        
        for section_name, section_data in window.sections.items():
            current_usage[section_name] = section_data['used']
            current_budgets[section_name] = section_data['budget']
        
        total_budget = window.total_budget
        total_used = window.get_total_used()
        
        # Calculate ideal allocation based on actual usage patterns
        usage_ratios = {}
        total_usage = sum(current_usage.values())
        
        if total_usage > 0:
            for section in current_usage:
                usage_ratios[section] = current_usage[section] / total_usage
        else:
            # Equal distribution if no usage
            usage_ratios = {section: 0.25 for section in current_usage}
        
        # Suggest new budgets with some buffer
        suggested_budgets = {}
        buffer_multiplier = 1.2  # 20% buffer
        
        for section, ratio in usage_ratios.items():
            base_suggestion = int(total_budget * ratio * buffer_multiplier)
            suggested_budgets[section] = base_suggestion
        
        # Adjust to fit total budget
        total_suggested = sum(suggested_budgets.values())
        if total_suggested != total_budget:
            adjustment_factor = total_budget / total_suggested
            for section in suggested_budgets:
                suggested_budgets[section] = int(suggested_budgets[section] * adjustment_factor)
        
        return {
            'current_budgets': current_budgets,
            'suggested_budgets': suggested_budgets,
            'potential_improvements': self._calculate_improvement_metrics(
                current_budgets, suggested_budgets, current_usage
            )
        }
    
    def get_historical_trends(self, days: int = 7) -> Dict:
        """Analyze trends from historical analysis data.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Trend analysis
        """
        if not self.analysis_history:
            return {'error': 'No historical data available'}
        
        # Filter recent analyses
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        recent_analyses = [
            analysis for analysis in self.analysis_history
            if self._parse_timestamp(analysis['timestamp']) >= cutoff_time
        ]
        
        if not recent_analyses:
            return {'error': f'No data available for last {days} days'}
        
        # Calculate trends
        efficiency_scores = [a['efficiency_score'] for a in recent_analyses]
        waste_counts = [len(a['waste_detection']['waste_items']) for a in recent_analyses]
        
        return {
            'period_days': days,
            'total_analyses': len(recent_analyses),
            'efficiency_trend': {
                'current': efficiency_scores[-1] if efficiency_scores else 0,
                'average': sum(efficiency_scores) / len(efficiency_scores),
                'min': min(efficiency_scores),
                'max': max(efficiency_scores)
            },
            'waste_trend': {
                'current': waste_counts[-1] if waste_counts else 0,
                'average': sum(waste_counts) / len(waste_counts),
                'improving': waste_counts[-1] < waste_counts[0] if len(waste_counts) > 1 else None
            }
        }
    
    def _analyze_overview(self, window) -> Dict:
        """Analyze overall window statistics."""
        return {
            'total_budget': window.total_budget,
            'total_used': window.get_total_used(),
            'total_allocated': window.get_total_budget(),
            'utilization': window.get_total_used() / window.total_budget if window.total_budget > 0 else 0,
            'budget_allocation_efficiency': window.get_total_budget() / window.total_budget if window.total_budget > 0 else 0,
            'is_over_budget': window.is_over_budget(),
            'overflow_warnings': len(window.overflow_warnings)
        }
    
    def _analyze_sections(self, window) -> Dict:
        """Analyze individual sections."""
        section_analysis = {}
        
        for section_name, section_data in window.sections.items():
            utilization = window.get_section_utilization(section_name)
            content_count = len(section_data['content'])
            
            analysis = {
                'utilization': utilization,
                'content_count': content_count,
                'avg_tokens_per_item': section_data['used'] / max(1, content_count),
                'status': self._classify_section_status(utilization, content_count),
                'priority_distribution': self._analyze_section_priorities(section_data['content'])
            }
            
            section_analysis[section_name] = analysis
        
        return section_analysis
    
    def _detect_waste(self, window) -> Dict:
        """Detect various types of waste in the context window."""
        waste_items = []
        total_waste_tokens = 0
        
        for section_name, section_data in window.sections.items():
            # Check for low utilization
            utilization = window.get_section_utilization(section_name)
            if (section_data['budget'] > 0 and 
                utilization < self.waste_thresholds['low_utilization'] and 
                section_data['used'] > 0):
                
                waste_tokens = section_data['budget'] - section_data['used']
                waste_items.append({
                    'type': 'low_utilization',
                    'section': section_name,
                    'description': f"Section using only {utilization:.1%} of allocated budget",
                    'waste_tokens': waste_tokens,
                    'severity': 'medium' if waste_tokens > 100 else 'low'
                })
                total_waste_tokens += waste_tokens
            
            # Check for high overhead content
            for i, item in enumerate(section_data['content']):
                content = item.get('content', '')
                if content:
                    overhead_ratio = self._calculate_overhead_ratio(content)
                    if overhead_ratio > self.waste_thresholds['high_overhead']:
                        waste_items.append({
                            'type': 'high_overhead',
                            'section': section_name,
                            'content_index': i,
                            'description': f"Content with {overhead_ratio:.1%} overhead (whitespace, etc.)",
                            'waste_tokens': int(item.get('tokens', 0) * overhead_ratio),
                            'severity': 'low'
                        })
        
        return {
            'waste_items': waste_items,
            'total_waste_tokens': total_waste_tokens,
            'waste_percentage': (total_waste_tokens / window.get_total_used()) * 100 if window.get_total_used() > 0 else 0
        }
    
    def _generate_suggestions(self, report: Dict, window) -> List[Dict]:
        """Generate optimization suggestions based on analysis."""
        suggestions = []
        
        # Suggest budget reallocation if needed
        overview = report['overview']
        if overview['budget_allocation_efficiency'] < 0.8:
            suggestions.append({
                'type': 'budget_reallocation',
                'priority': 'high',
                'description': 'Consider reallocating section budgets based on actual usage patterns',
                'action': 'Run suggest_budget_reallocation() for specific recommendations'
            })
        
        # Suggest compression for high overhead content
        waste_items = report['waste_detection']['waste_items']
        high_overhead_count = sum(1 for item in waste_items if item['type'] == 'high_overhead')
        if high_overhead_count > 0:
            suggestions.append({
                'type': 'content_compression',
                'priority': 'medium',
                'description': f'Found {high_overhead_count} items with high overhead',
                'action': 'Apply message compression to reduce whitespace and redundancy'
            })
        
        # Suggest removing low priority content if over budget
        if overview['is_over_budget']:
            suggestions.append({
                'type': 'priority_filtering',
                'priority': 'high',
                'description': 'Context window is over budget',
                'action': 'Remove optional and normal priority items to fit within budget'
            })
        
        # Suggest strategy optimization
        section_analysis = report['section_analysis']
        underutilized_sections = [
            name for name, data in section_analysis.items()
            if data['utilization'] < 0.5 and data['content_count'] > 0
        ]
        
        if underutilized_sections:
            suggestions.append({
                'type': 'strategy_optimization',
                'priority': 'medium',
                'description': f'Sections {underutilized_sections} are underutilized',
                'action': 'Consider using more aggressive content selection strategies'
            })
        
        return suggestions
    
    def _calculate_efficiency_score(self, report: Dict) -> float:
        """Calculate overall efficiency score (0.0 - 1.0)."""
        overview = report['overview']
        waste_detection = report['waste_detection']
        
        # Base score from utilization
        utilization_score = min(1.0, overview['utilization'])
        
        # Penalty for waste
        waste_penalty = waste_detection['waste_percentage'] / 100
        
        # Penalty for being over budget
        over_budget_penalty = 0.2 if overview['is_over_budget'] else 0
        
        # Bonus for good budget allocation
        allocation_bonus = overview['budget_allocation_efficiency'] * 0.1
        
        efficiency_score = utilization_score - waste_penalty - over_budget_penalty + allocation_bonus
        
        return max(0.0, min(1.0, efficiency_score))
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        if not content1 or not content2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_overhead_ratio(self, content: str) -> float:
        """Calculate the ratio of whitespace/overhead to actual content."""
        if not content:
            return 0.0
        
        total_chars = len(content)
        whitespace_chars = len(content) - len(content.strip())
        
        # Count excessive whitespace
        # re imported at module level
        excessive_whitespace = len(re.findall(r'\s{2,}', content)) * 2  # Approximate
        
        return (whitespace_chars + excessive_whitespace) / total_chars
    
    def _classify_section_status(self, utilization: float, content_count: int) -> str:
        """Classify section status based on utilization and content."""
        if content_count == 0:
            return 'empty'
        elif utilization > 0.9:
            return 'full'
        elif utilization > 0.7:
            return 'well_utilized'
        elif utilization > 0.3:
            return 'moderate'
        else:
            return 'underutilized'
    
    def _analyze_section_priorities(self, content_items: List[Dict]) -> Dict:
        """Analyze priority distribution in a section."""
        priority_counts = {'critical': 0, 'important': 0, 'normal': 0, 'optional': 0}
        
        for item in content_items:
            priority = item.get('priority', 'normal')
            if priority in priority_counts:
                priority_counts[priority] += 1
        
        total = len(content_items)
        if total == 0:
            return priority_counts
        
        # Convert to percentages
        priority_percentages = {
            priority: (count / total) * 100
            for priority, count in priority_counts.items()
        }
        
        return priority_percentages
    
    def _calculate_improvement_metrics(self, current: Dict, suggested: Dict, usage: Dict) -> Dict:
        """Calculate potential improvement metrics."""
        improvements = {}
        
        for section in current:
            current_waste = max(0, current[section] - usage[section])
            suggested_waste = max(0, suggested[section] - usage[section])
            
            improvements[section] = {
                'current_waste': current_waste,
                'suggested_waste': suggested_waste,
                'waste_reduction': current_waste - suggested_waste,
                'efficiency_gain': (suggested_waste < current_waste)
            }
        
        return improvements
    
    def _log_analysis(self, report: Dict) -> None:
        """Log analysis to file (append mode).
        
        Note: Uses append mode, not atomic write. Log entries may be
        incomplete if the process is interrupted mid-write.
        """
        if not self.log_file:
            return
        
        try:
            # Ensure directory exists
            dir_path = os.path.dirname(self.log_file)
            if dir_path:  # Guard against empty dirname
                os.makedirs(dir_path, exist_ok=True)
            
            # Append to log file
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(report) + '\n')
        except Exception as e:
            # Log the error instead of silently swallowing it
            logger.warning("Failed to log analysis to %s: %s", self.log_file, e)
    
    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse ISO timestamp to Unix timestamp."""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.timestamp()
        except (ValueError, AttributeError, TypeError):
            return 0.0