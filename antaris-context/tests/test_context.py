"""
Comprehensive test suite for antaris-context package.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, mock_open

from antaris_context import (
    ContextManager, ContextWindow, MessageCompressor, ContextProfiler,
    RecencyStrategy, RelevanceStrategy, HybridStrategy, BudgetStrategy
)


class TestContextWindow:
    """Test ContextWindow functionality."""
    
    def test_init_default(self):
        """Test ContextWindow initialization with defaults."""
        window = ContextWindow()
        assert window.total_budget == 8000
        assert len(window.sections) == 4
        assert 'system' in window.sections
        assert 'memory' in window.sections
        assert 'conversation' in window.sections
        assert 'tools' in window.sections
    
    def test_init_custom_budget(self):
        """Test ContextWindow initialization with custom budget."""
        window = ContextWindow(total_budget=4000)
        assert window.total_budget == 4000
    
    def test_set_section_budget(self):
        """Test setting section budgets."""
        window = ContextWindow()
        window.set_section_budget('system', 1000)
        assert window.sections['system']['budget'] == 1000
    
    def test_set_invalid_section_budget(self):
        """Test setting budget for invalid section."""
        window = ContextWindow()
        with pytest.raises(ValueError, match="Unknown section"):
            window.set_section_budget('invalid', 1000)
    
    def test_add_content_basic(self):
        """Test adding content to a section."""
        window = ContextWindow()
        window.set_section_budget('system', 100)
        
        result = window.add_content('system', 'Hello world', priority='normal')
        assert result is True
        assert window.sections['system']['used'] > 0
        assert len(window.sections['system']['content']) == 1
    
    def test_add_content_overflow(self):
        """Test adding content that exceeds budget."""
        window = ContextWindow()
        window.set_section_budget('system', 10)  # Very small budget
        
        long_text = "This is a very long text that should exceed the small budget we set for this section"
        result = window.add_content('system', long_text, priority='normal')
        
        assert result is False  # Should indicate overflow
        assert len(window.overflow_warnings) > 0
    
    def test_get_total_used(self):
        """Test getting total tokens used."""
        window = ContextWindow()
        window.set_section_budget('system', 100)
        window.set_section_budget('memory', 100)
        
        window.add_content('system', 'Hello')
        window.add_content('memory', 'World')
        
        total = window.get_total_used()
        assert total > 0
        assert total == (window.sections['system']['used'] + window.sections['memory']['used'])
    
    def test_is_over_budget(self):
        """Test budget overflow detection."""
        window = ContextWindow(total_budget=50)
        window.set_section_budget('system', 100)  # Exceeds total budget
        
        # Add content that uses the full section budget
        long_text = "A" * 400  # Should be around 100 tokens
        window.add_content('system', long_text)
        
        assert window.is_over_budget() is True
    
    def test_get_section_utilization(self):
        """Test section utilization calculation."""
        window = ContextWindow()
        window.set_section_budget('system', 100)
        
        # Add content that uses half the budget
        text = "A" * 200  # Should be around 50 tokens
        window.add_content('system', text)
        
        utilization = window.get_section_utilization('system')
        assert 0.4 <= utilization <= 0.6  # Should be around 0.5
    
    def test_clear_section(self):
        """Test clearing section content."""
        window = ContextWindow()
        window.set_section_budget('system', 100)
        window.add_content('system', 'Hello world')
        
        assert window.sections['system']['used'] > 0
        assert len(window.sections['system']['content']) > 0
        
        window.clear_section('system')
        
        assert window.sections['system']['used'] == 0
        assert len(window.sections['system']['content']) == 0
    
    def test_remove_content_by_priority(self):
        """Test removing content by priority level."""
        window = ContextWindow()
        window.set_section_budget('system', 1000)
        
        window.add_content('system', 'Critical content', priority='critical')
        window.add_content('system', 'Optional content', priority='optional')
        
        tokens_freed = window.remove_content_by_priority('system', 'optional')
        
        assert tokens_freed > 0
        assert len(window.sections['system']['content']) == 1
        assert window.sections['system']['content'][0]['priority'] == 'critical'
    
    def test_token_estimation(self):
        """Test token estimation accuracy."""
        window = ContextWindow()
        
        # Test basic estimation
        tokens = window._estimate_tokens("hello world")
        assert tokens >= 2  # Should be at least 2 tokens
        
        # Test empty string
        tokens = window._estimate_tokens("")
        assert tokens == 0
        
        # Test longer text
        long_text = "This is a longer piece of text with multiple words"
        tokens = window._estimate_tokens(long_text)
        assert tokens > 10
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        window = ContextWindow(total_budget=4000)
        window.set_section_budget('system', 500)
        
        json_str = window.to_json()
        assert isinstance(json_str, str)
        
        # Parse JSON to ensure it's valid
        data = json.loads(json_str)
        assert data['total_budget'] == 4000
        assert data['sections']['system']['budget'] == 500
        
        # Test deserialization
        restored_window = ContextWindow.from_json(json_str)
        assert restored_window.total_budget == 4000
        assert restored_window.sections['system']['budget'] == 500


class TestMessageCompressor:
    """Test MessageCompressor functionality."""
    
    def test_init_compression_levels(self):
        """Test initialization with different compression levels."""
        for level in ['light', 'moderate', 'aggressive']:
            compressor = MessageCompressor(level)
            assert compressor.level == level
    
    def test_init_invalid_level(self):
        """Test initialization with invalid compression level."""
        with pytest.raises(ValueError, match="Unknown compression level"):
            MessageCompressor('invalid')
    
    def test_compress_basic(self):
        """Test basic text compression."""
        compressor = MessageCompressor('moderate')
        
        text_with_whitespace = "Hello    world\n\n\nTest   text"
        compressed = compressor.compress(text_with_whitespace)
        
        assert len(compressed) < len(text_with_whitespace)
        assert 'Hello' in compressed
        assert 'world' in compressed
    
    def test_compress_empty_string(self):
        """Test compressing empty string."""
        compressor = MessageCompressor('moderate')
        result = compressor.compress("")
        assert result == ""
    
    def test_compression_stats(self):
        """Test compression statistics tracking."""
        compressor = MessageCompressor('moderate')
        
        original_text = "This  has    lots   of    whitespace\n\n\n"
        compressor.compress(original_text)
        
        stats = compressor.get_compression_stats()
        assert stats['original_length'] > 0
        assert stats['compressed_length'] > 0
        assert stats['bytes_saved'] >= 0
        assert 0 <= stats['compression_ratio'] <= 1
    
    def test_compress_tool_output(self):
        """Test tool output compression."""
        compressor = MessageCompressor('moderate')
        
        # Create output with many lines
        lines = [f"Line {i}" for i in range(100)]
        long_output = '\n'.join(lines)
        
        compressed = compressor.compress_tool_output(long_output, max_lines=20, keep_first=10, keep_last=10)
        
        assert len(compressed.split('\n')) < len(lines)
        assert 'Line 0' in compressed  # First line preserved
        assert 'Line 99' in compressed  # Last line preserved
        assert 'truncated' in compressed
    
    def test_compress_message_list(self):
        """Test compressing a list of messages."""
        compressor = MessageCompressor('moderate')
        
        messages = [
            {'role': 'user', 'content': 'Hello    world   with   whitespace'},
            {'role': 'assistant', 'content': 'Response   with    spacing'},
            {'role': 'tool', 'content': '\n'.join([f'Output line {i}' for i in range(50)])}
        ]
        
        compressed = compressor.compress_message_list(messages, max_content_length=100)
        
        assert len(compressed) == 3
        assert len(compressed[0]['content']) < len(messages[0]['content'])
        # Tool message should be truncated due to length limit
        assert 'truncated' in compressed[2]['content'] or len(compressed[2]['content']) <= 100
    
    def test_set_config(self):
        """Test updating compression configuration."""
        compressor = MessageCompressor('light')
        
        compressor.set_config(remove_redundant_spaces=True)
        assert compressor.config['remove_redundant_spaces'] is True
    
    def test_set_invalid_config(self):
        """Test setting invalid configuration option."""
        compressor = MessageCompressor('light')
        
        with pytest.raises(ValueError, match="Unknown configuration option"):
            compressor.set_config(invalid_option=True)
    
    def test_reset_stats(self):
        """Test resetting compression statistics."""
        compressor = MessageCompressor('moderate')
        
        compressor.compress("Some text to compress")
        assert compressor.get_compression_stats()['original_length'] > 0
        
        compressor.reset_stats()
        stats = compressor.get_compression_stats()
        assert stats['original_length'] == 0
        assert stats['compressed_length'] == 0


class TestContextStrategies:
    """Test context selection strategies."""
    
    def test_recency_strategy(self):
        """Test RecencyStrategy content selection."""
        strategy = RecencyStrategy()
        
        content = [
            {'content': 'Old message', 'tokens': 10, 'priority': 'normal', 'added_at': 1},
            {'content': 'New message', 'tokens': 10, 'priority': 'normal', 'added_at': 2},
            {'content': 'Critical old', 'tokens': 10, 'priority': 'critical', 'added_at': 0}
        ]
        
        selected = strategy.select_content(content, budget=25)
        
        assert len(selected) <= 3
        # Should prefer newer content and critical priority
        selected_contents = [item['content'] for item in selected]
        assert 'Critical old' in selected_contents  # Critical priority
        assert 'New message' in selected_contents  # Most recent
    
    def test_relevance_strategy_with_query(self):
        """Test RelevanceStrategy with query context."""
        strategy = RelevanceStrategy()
        
        content = [
            {'content': 'Python programming tutorial', 'tokens': 10, 'priority': 'normal'},
            {'content': 'JavaScript web development', 'tokens': 10, 'priority': 'normal'},
            {'content': 'Python data analysis', 'tokens': 10, 'priority': 'normal'}
        ]
        
        selected = strategy.select_content(content, budget=25, query="Python coding help")
        
        assert len(selected) > 0
        # Should prefer Python-related content
        python_count = sum(1 for item in selected if 'Python' in item['content'])
        assert python_count >= 1
    
    def test_relevance_strategy_no_query(self):
        """Test RelevanceStrategy without query (fallback to priority)."""
        strategy = RelevanceStrategy()
        
        content = [
            {'content': 'Normal priority', 'tokens': 10, 'priority': 'normal'},
            {'content': 'Critical priority', 'tokens': 10, 'priority': 'critical'},
            {'content': 'Optional priority', 'tokens': 10, 'priority': 'optional'}
        ]
        
        selected = strategy.select_content(content, budget=25, query=None)
        
        assert len(selected) > 0
        # Should prefer higher priority items
        priorities = [item['priority'] for item in selected]
        assert 'critical' in priorities
    
    def test_hybrid_strategy(self):
        """Test HybridStrategy combining recency and relevance."""
        strategy = HybridStrategy(recency_weight=0.5, relevance_weight=0.5)
        
        content = [
            {'content': 'Old Python tutorial', 'tokens': 10, 'priority': 'normal', 'added_at': 1},
            {'content': 'New JavaScript guide', 'tokens': 10, 'priority': 'normal', 'added_at': 10},
            {'content': 'Recent Python example', 'tokens': 10, 'priority': 'normal', 'added_at': 8}
        ]
        
        selected = strategy.select_content(content, budget=25, query="Python programming")
        
        assert len(selected) > 0
        # Should balance recency and relevance
        contents = [item['content'] for item in selected]
        # Recent Python example should score well on both recency and relevance
        assert 'Recent Python example' in contents
    
    def test_budget_strategy_greedy(self):
        """Test BudgetStrategy with greedy approach."""
        strategy = BudgetStrategy(approach='greedy')
        
        content = [
            {'content': 'Critical short', 'tokens': 5, 'priority': 'critical'},  # High value per token
            {'content': 'Long optional text', 'tokens': 20, 'priority': 'optional'},  # Low value per token
            {'content': 'Important medium', 'tokens': 10, 'priority': 'important'}  # Medium value per token
        ]
        
        selected = strategy.select_content(content, budget=20)
        
        # Should prefer high value-per-token items
        contents = [item['content'] for item in selected]
        assert 'Critical short' in contents
    
    def test_strategy_empty_content(self):
        """Test strategies with empty content list."""
        strategies = [
            RecencyStrategy(),
            RelevanceStrategy(),
            HybridStrategy(),
            BudgetStrategy()
        ]
        
        for strategy in strategies:
            selected = strategy.select_content([], budget=100)
            assert selected == []
    
    def test_strategy_zero_budget(self):
        """Test strategies with zero budget."""
        strategy = RecencyStrategy()
        content = [{'content': 'Test', 'tokens': 10, 'priority': 'normal', 'added_at': 1}]
        
        selected = strategy.select_content(content, budget=0)
        assert selected == []


class TestContextProfiler:
    """Test ContextProfiler functionality."""
    
    def test_init_with_log_file(self):
        """Test profiler initialization with log file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            profiler = ContextProfiler(log_file=tmp.name)
            assert profiler.log_file == tmp.name
        os.unlink(tmp.name)
    
    def test_analyze_window(self):
        """Test comprehensive window analysis."""
        profiler = ContextProfiler()
        window = ContextWindow(total_budget=1000)
        window.set_section_budget('system', 200)
        window.add_content('system', 'Test content', priority='normal')
        
        analysis = profiler.analyze_window(window)
        
        assert 'timestamp' in analysis
        assert 'overview' in analysis
        assert 'section_analysis' in analysis
        assert 'waste_detection' in analysis
        assert 'optimization_suggestions' in analysis
        assert 'efficiency_score' in analysis
        assert 0 <= analysis['efficiency_score'] <= 1.0
    
    def test_analyze_content_distribution(self):
        """Test content distribution analysis."""
        profiler = ContextProfiler()
        window = ContextWindow()
        window.set_section_budget('system', 100)
        window.set_section_budget('memory', 100)
        
        window.add_content('system', 'System content', priority='critical')
        window.add_content('memory', 'Memory content', priority='important')
        
        distribution = profiler.analyze_content_distribution(window)
        
        assert 'total_content_items' in distribution
        assert distribution['total_content_items'] == 2
        assert 'by_section' in distribution
        assert 'by_priority' in distribution
        assert distribution['by_priority']['critical'] == 1
        assert distribution['by_priority']['important'] == 1
    
    def test_detect_redundant_content(self):
        """Test redundant content detection."""
        profiler = ContextProfiler()
        window = ContextWindow()
        window.set_section_budget('system', 200)
        
        # Add similar content
        window.add_content('system', 'This is a test message', priority='normal')
        window.add_content('system', 'This is a test message', priority='normal')  # Identical
        window.add_content('system', 'Completely different content', priority='normal')
        
        redundant = profiler.detect_redundant_content(window, similarity_threshold=0.9)
        
        assert len(redundant) > 0  # Should find the identical messages
        assert redundant[0]['similarity'] >= 0.9
    
    def test_suggest_budget_reallocation(self):
        """Test budget reallocation suggestions."""
        profiler = ContextProfiler()
        window = ContextWindow(total_budget=1000)
        window.set_section_budget('system', 100)
        window.set_section_budget('memory', 900)  # Over-allocated
        
        # Use only system section
        window.add_content('system', 'A' * 200, priority='normal')  # ~50 tokens
        
        suggestions = profiler.suggest_budget_reallocation(window)
        
        assert 'current_budgets' in suggestions
        assert 'suggested_budgets' in suggestions
        # Should suggest reducing memory budget and increasing system
        assert suggestions['suggested_budgets']['memory'] < suggestions['current_budgets']['memory']
    
    def test_get_historical_trends_no_data(self):
        """Test historical trends with no data."""
        profiler = ContextProfiler()
        trends = profiler.get_historical_trends()
        
        assert 'error' in trends
        assert 'No historical data available' in trends['error']


class TestContextManager:
    """Test ContextManager functionality."""
    
    def test_init_default(self):
        """Test ContextManager initialization with defaults."""
        manager = ContextManager()
        assert manager.total_budget == 8000
        assert isinstance(manager.window, ContextWindow)
        assert isinstance(manager.compressor, MessageCompressor)
        assert isinstance(manager.profiler, ContextProfiler)
    
    def test_init_with_config_file(self):
        """Test ContextManager initialization with config file."""
        config = {
            'compression_level': 'aggressive',
            'section_budgets': {'system': 500, 'memory': 1500, 'conversation': 3000, 'tools': 1000}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(config, tmp)
            tmp.flush()
            
            manager = ContextManager(config_file=tmp.name)
            assert manager.config['compression_level'] == 'aggressive'
            assert manager.window.sections['system']['budget'] == 500
        
        os.unlink(tmp.name)
    
    def test_set_section_budget(self):
        """Test setting section budgets."""
        manager = ContextManager()
        manager.set_section_budget('system', 1500)
        
        assert manager.window.sections['system']['budget'] == 1500
        assert manager.config['section_budgets']['system'] == 1500
    
    def test_add_content_string(self):
        """Test adding string content."""
        manager = ContextManager()
        manager.set_section_budget('system', 100)
        
        result = manager.add_content('system', 'Hello world', priority='normal')
        assert result is True
        assert len(manager.window.sections['system']['content']) == 1
    
    def test_add_content_message_list(self):
        """Test adding message list content."""
        manager = ContextManager()
        manager.set_section_budget('conversation', 1000)
        
        messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there'},
            {'role': 'user', 'content': 'How are you?'}
        ]
        
        result = manager.add_content('conversation', messages, priority='normal')
        assert result is True
        assert len(manager.window.sections['conversation']['content']) > 0
    
    def test_set_strategy(self):
        """Test setting different strategies."""
        manager = ContextManager()
        
        # Test each strategy type
        manager.set_strategy('recency', prefer_high_priority=True)
        assert isinstance(manager.strategy, RecencyStrategy)
        
        manager.set_strategy('relevance', min_score=0.3)
        assert isinstance(manager.strategy, RelevanceStrategy)
        
        manager.set_strategy('hybrid', recency_weight=0.3, relevance_weight=0.7)
        assert isinstance(manager.strategy, HybridStrategy)
        
        manager.set_strategy('budget', approach='greedy')
        assert isinstance(manager.strategy, BudgetStrategy)
    
    def test_set_invalid_strategy(self):
        """Test setting invalid strategy."""
        manager = ContextManager()
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            manager.set_strategy('invalid_strategy')
    
    def test_set_compression_level(self):
        """Test setting compression level."""
        manager = ContextManager()
        
        manager.set_compression_level('aggressive')
        assert manager.compressor.level == 'aggressive'
        assert manager.config['compression_level'] == 'aggressive'
    
    def test_optimize_context(self):
        """Test context optimization."""
        manager = ContextManager(total_budget=100)  # Small budget for testing
        manager.set_section_budget('system', 50)
        manager.add_content('system', 'A' * 400, priority='normal')  # Exceed budget
        
        optimization = manager.optimize_context(target_utilization=0.8)
        
        assert 'initial_state' in optimization
        assert 'actions_taken' in optimization
        assert 'final_state' in optimization
        assert 'success' in optimization
        assert isinstance(optimization['actions_taken'], list)
    
    def test_get_usage_report(self):
        """Test getting usage report."""
        manager = ContextManager()
        manager.set_section_budget('system', 100)
        manager.add_content('system', 'Test content', priority='normal')
        
        report = manager.get_usage_report()
        
        assert 'total_budget' in report
        assert 'total_used' in report
        assert 'sections' in report
        assert 'configuration' in report
        assert 'compression_stats' in report
        assert report['total_budget'] == manager.total_budget
    
    def test_clear_section(self):
        """Test clearing section content."""
        manager = ContextManager()
        manager.set_section_budget('system', 100)
        manager.add_content('system', 'Test content', priority='normal')
        
        assert len(manager.window.sections['system']['content']) > 0
        
        manager.clear_section('system')
        assert len(manager.window.sections['system']['content']) == 0
    
    def test_clear_all_content(self):
        """Test clearing all content."""
        manager = ContextManager()
        manager.set_section_budget('system', 100)
        manager.set_section_budget('memory', 100)
        manager.add_content('system', 'System content', priority='normal')
        manager.add_content('memory', 'Memory content', priority='normal')
        
        manager.clear_all_content()
        
        for section in manager.window.sections:
            assert len(manager.window.sections[section]['content']) == 0
    
    def test_get_available_budget_single_section(self):
        """Test getting available budget for single section."""
        manager = ContextManager()
        manager.set_section_budget('system', 100)
        manager.add_content('system', 'A' * 100, priority='normal')  # Use ~25 tokens
        
        available = manager.get_available_budget('system')
        assert available > 0
        assert available < 100
    
    def test_get_available_budget_all_sections(self):
        """Test getting available budget for all sections."""
        manager = ContextManager()
        manager.set_section_budget('system', 100)
        manager.set_section_budget('memory', 200)
        
        available = manager.get_available_budget()
        
        assert isinstance(available, dict)
        assert 'system' in available
        assert 'memory' in available
        assert available['system'] <= 100
        assert available['memory'] <= 200
    
    def test_export_import_state(self):
        """Test exporting and importing state."""
        manager = ContextManager()
        manager.set_section_budget('system', 500)
        manager.set_compression_level('aggressive')
        
        # Export state
        state_json = manager.export_state()
        assert isinstance(state_json, str)
        
        # Create new manager and import state
        new_manager = ContextManager()
        new_manager.import_state(state_json)
        
        assert new_manager.window.sections['system']['budget'] == 500
        assert new_manager.config['compression_level'] == 'aggressive'
    
    def test_save_load_config(self):
        """Test saving and loading configuration."""
        manager = ContextManager()
        manager.set_compression_level('aggressive')
        manager.set_section_budget('system', 777)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            manager.save_config(tmp.name)
            
            # Load with new manager
            new_manager = ContextManager(config_file=tmp.name)
            assert new_manager.config['compression_level'] == 'aggressive'
            assert new_manager.window.sections['system']['budget'] == 777
        
        os.unlink(tmp.name)
    
    def test_analyze_context(self):
        """Test context analysis."""
        manager = ContextManager()
        manager.set_section_budget('system', 100)
        manager.add_content('system', 'Test content', priority='normal')
        
        analysis = manager.analyze_context(log_analysis=False)
        
        assert 'timestamp' in analysis
        assert 'efficiency_score' in analysis
        assert 0 <= analysis['efficiency_score'] <= 1.0


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_workflow(self):
        """Test complete workflow with all components."""
        # Initialize manager with configuration
        manager = ContextManager(total_budget=2000)
        
        # Set up budgets
        manager.set_section_budget('system', 300)
        manager.set_section_budget('memory', 500)
        manager.set_section_budget('conversation', 1000)
        manager.set_section_budget('tools', 200)
        
        # Set strategy and compression
        manager.set_strategy('hybrid', recency_weight=0.4, relevance_weight=0.6)
        manager.set_compression_level('moderate')
        
        # Add various content types
        manager.add_content('system', 'You are a helpful assistant', priority='critical')
        manager.add_content('memory', 'User prefers concise responses', priority='important')
        
        # Add conversation with query context
        messages = [
            {'role': 'user', 'content': 'What is Python?'},
            {'role': 'assistant', 'content': 'Python is a programming language...'},
            {'role': 'user', 'content': 'How do I install Python packages?'},
            {'role': 'assistant', 'content': 'You can use pip to install packages...'}
        ]
        
        query = "Python programming help"
        manager.add_content('conversation', messages, query=query, priority='normal')
        
        # Add tool output
        tool_output = "Package installed successfully\nVersion: 1.2.3\nLocation: /usr/local/lib/python3.9/"
        manager.add_content('tools', tool_output, priority='optional')
        
        # Get usage report
        report = manager.get_usage_report()
        assert report['total_used'] > 0
        assert report['total_used'] <= manager.total_budget
        
        # Optimize context
        optimization = manager.optimize_context(query=query, target_utilization=0.85)
        assert 'success' in optimization
        
        # Analyze context
        analysis = manager.analyze_context()
        assert analysis['efficiency_score'] >= 0  # Can be 0 if utilization is very low
        
        # Verify the system still works after optimization
        final_report = manager.get_usage_report()
        assert final_report['total_used'] <= manager.total_budget
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        manager = ContextManager()
        
        # Test with zero budget sections
        manager.set_section_budget('system', 0)
        result = manager.add_content('system', 'Test', priority='normal')
        # Should handle gracefully
        
        # Test with very large content
        large_content = 'A' * 10000
        manager.set_section_budget('memory', 100)  # Small budget
        result = manager.add_content('memory', large_content, priority='optional')
        # Should not crash
        
        # Test optimization with empty context
        empty_manager = ContextManager()
        optimization = empty_manager.optimize_context()
        assert 'success' in optimization
    
    def test_compression_integration(self):
        """Test compression integration across components."""
        manager = ContextManager()
        manager.set_compression_level('aggressive')
        manager.set_section_budget('conversation', 500)
        
        # Add content with lots of whitespace
        whitespace_content = "This   has    lots\n\n\nof    whitespace   and   formatting"
        original_length = len(whitespace_content)
        
        manager.add_content('conversation', whitespace_content, compress=True)
        
        # Check that content was compressed
        stored_content = manager.window.sections['conversation']['content'][0]['content']
        assert len(stored_content) < original_length
        
        # Verify compression stats were updated
        stats = manager.compressor.get_compression_stats()
        assert stats['bytes_saved'] > 0


class TestCascadeOverflow:
    """Test cascade_overflow budget reallocation."""

    def test_cascade_basic(self):
        manager = ContextManager(total_budget=8000)
        manager.set_section_budget('system', 1000)
        manager.set_section_budget('memory', 2000)
        manager.set_section_budget('conversation', 200)  # Tiny budget
        manager.set_section_budget('tools', 1000)
        # Add content that overflows the small budget (bypass compression)
        manager.add_content('conversation', 'word ' * 500, priority='normal', compress=False)
        assert manager.window.sections['conversation']['used'] > manager.window.sections['conversation']['budget']
        redistributed = manager.cascade_overflow('conversation')
        assert redistributed > 0
        assert manager.window.sections['conversation']['budget'] > 200

    def test_cascade_no_overflow(self):
        manager = ContextManager(total_budget=8000)
        manager.set_section_budget('conversation', 4000)
        manager.add_content('conversation', 'small text')
        redistributed = manager.cascade_overflow('conversation')
        assert redistributed == 0

    def test_cascade_no_slack(self):
        manager = ContextManager(total_budget=8000)
        for section in ['system', 'memory', 'conversation', 'tools']:
            manager.set_section_budget(section, 2000)
            manager.add_content(section, 'x' * 2500)
        redistributed = manager.cascade_overflow('conversation')
        assert redistributed == 0

    def test_cascade_invalid_section(self):
        manager = ContextManager()
        assert manager.cascade_overflow('nonexistent') == 0


class TestSnapshots:
    """Test save/restore snapshot functionality."""

    def test_save_and_restore(self):
        manager = ContextManager(total_budget=8000)
        manager.set_section_budget('system', 1000)
        manager.add_content('system', 'system prompt here')
        manager.save_snapshot('checkpoint1')
        
        # Modify state
        manager.clear_section('system')
        assert manager.window.sections['system']['used'] == 0
        
        # Restore
        restored = manager.restore_snapshot('checkpoint1')
        assert restored is True

    def test_restore_nonexistent(self):
        manager = ContextManager()
        assert manager.restore_snapshot('nope') is False

    def test_list_snapshots(self):
        manager = ContextManager()
        assert manager.list_snapshots() == []
        manager.save_snapshot('snap1')
        manager.save_snapshot('snap2')
        snapshots = manager.list_snapshots()
        assert len(snapshots) == 2
        names = [s['name'] for s in snapshots]
        assert 'snap1' in names
        assert 'snap2' in names

    def test_multiple_save_overwrite(self):
        manager = ContextManager()
        manager.save_snapshot('test')
        manager.save_snapshot('test')  # Overwrite
        assert len(manager.list_snapshots()) == 1

    def test_snapshot_does_not_restore_content(self):
        """Snapshots are structural only — content is NOT preserved."""
        manager = ContextManager(total_budget=8000)
        manager.set_section_budget('system', 1000)
        manager.add_content('system', 'important system prompt', compress=False)
        assert len(manager.window.sections['system']['content']) > 0
        
        manager.save_snapshot('with_content')
        manager.clear_all_content()
        
        restored = manager.restore_snapshot('with_content')
        assert restored is True
        # Structural state restored but content is empty
        assert manager.window.sections['system']['content'] == []


class TestAdaptiveBudgets:
    """Test adaptive budget management."""

    def test_enable_disable(self):
        manager = ContextManager()
        manager.enable_adaptive_budgets(True, reallocation_threshold=0.4)
        assert manager.config['adaptive_budgets']['enabled'] is True
        assert manager.config['adaptive_budgets']['reallocation_threshold'] == 0.4
        manager.enable_adaptive_budgets(False)
        assert manager.config['adaptive_budgets']['enabled'] is False

    def test_track_usage(self):
        manager = ContextManager()
        manager.enable_adaptive_budgets(True)
        manager.set_section_budget('system', 1000)
        manager.add_content('system', 'test content')
        manager.track_usage()
        history = manager.config['adaptive_budgets']['usage_history']
        assert len(history) == 1
        assert 'section_usage' in history[0]

    def test_track_usage_disabled(self):
        manager = ContextManager()
        manager.track_usage()  # Should be no-op
        assert len(manager.config['adaptive_budgets']['usage_history']) == 0

    def test_suggest_needs_data(self):
        manager = ContextManager()
        manager.enable_adaptive_budgets(True)
        result = manager.suggest_adaptive_reallocation()
        assert result is None  # Not enough data

    def test_auto_apply_respects_flag(self):
        """auto_apply=False should never apply, regardless of savings."""
        manager = ContextManager()
        manager.enable_adaptive_budgets(True)
        # Populate enough history
        manager.set_section_budget('system', 1000)
        manager.set_section_budget('memory', 2000)
        for _ in range(15):
            manager.track_usage()
        result = manager.apply_adaptive_reallocation(auto_apply=False)
        assert result is False


class TestTemplates:
    """Test section template functionality."""

    def test_apply_template(self):
        manager = ContextManager()
        manager.apply_template('chatbot')
        assert manager.window.sections['system']['budget'] == 800
        assert manager.window.sections['conversation']['budget'] == 5000

    def test_apply_invalid_template(self):
        manager = ContextManager()
        with pytest.raises(ValueError):
            manager.apply_template('nonexistent')

    def test_get_available_templates(self):
        templates = ContextManager.get_available_templates()
        assert 'chatbot' in templates
        assert 'agent_with_tools' in templates
        assert 'rag_pipeline' in templates
        assert 'code_assistant' in templates
        assert 'balanced' in templates

    def test_template_budgets_sum_to_8000(self):
        templates = ContextManager.get_available_templates()
        for name, budgets in templates.items():
            assert sum(budgets.values()) == 8000, f"{name} sums to {sum(budgets.values())}"

    def test_template_via_constructor(self):
        manager = ContextManager(template='rag_pipeline')
        assert manager.window.sections['conversation']['budget'] == 4500


class TestSmartSentenceDuplicates:
    """Regression test for sentence dedup bug (#1)."""

    def test_duplicate_sentences_preserve_order(self):
        compressor = MessageCompressor('aggressive')
        # Text with duplicate sentences
        text = ("Important first sentence. " 
                "Some filler content here. "
                "Some filler content here. "  # Duplicate
                "Critical ending statement with numbers 42.")
        result = compressor.compress(text)
        # Should not crash or produce garbled output
        assert len(result) > 0
        # First sentence should still come before ending
        if "Important" in result and "Critical" in result:
            assert result.index("Important") < result.index("Critical")


class TestFromJsonRestoresUsed:
    """Regression test for from_json not restoring used counts (#6)."""

    def test_from_json_restores_used(self):
        window = ContextWindow(8000)
        window.set_section_budget('system', 1000)
        window.add_content('system', 'test content')
        used_before = window.sections['system']['used']
        
        json_str = window.to_json()
        restored = ContextWindow.from_json(json_str)
        assert restored.sections['system']['used'] == used_before


class TestUtilizationNeverInf:
    """Regression test for get_section_utilization returning inf (#12)."""

    def test_zero_budget_with_content(self):
        window = ContextWindow(8000)
        # Budget is 0 but force some used tokens
        window.sections['system']['used'] = 100
        util = window.get_section_utilization('system')
        assert util == 1.0  # Not inf
        assert util != float('inf')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])