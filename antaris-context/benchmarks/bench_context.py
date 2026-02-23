#!/usr/bin/env python3
"""
Performance benchmarks for antaris-context.

Measures operations per second for all major operations to detect
performance regressions and optimize bottlenecks.
"""

import time
import sys
import os
import statistics
from typing import Dict, List, Callable, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from antaris_context import (
    ContextManager,
    ContextWindow,
    MessageCompressor,
    RecencyStrategy,
    RelevanceStrategy,
    HybridStrategy,
    ContextProfiler
)


class ContextBenchmark:
    """Benchmark suite for context operations."""
    
    def __init__(self, warmup_iterations: int = 100, benchmark_iterations: int = 1000):
        """Initialize benchmark suite.
        
        Args:
            warmup_iterations: Number of warmup runs
            benchmark_iterations: Number of benchmark runs for measurement
        """
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.results = {}
        
        # Generate test data
        self.test_content = self._generate_test_content()
        self.test_messages = self._generate_test_messages()
    
    def run_all_benchmarks(self) -> Dict[str, Dict]:
        """Run all benchmarks and return results."""
        print("🚀 Running antaris-context performance benchmarks...")
        print(f"   Warmup: {self.warmup_iterations} iterations")
        print(f"   Benchmark: {self.benchmark_iterations} iterations")
        print()
        
        benchmarks = [
            ("Context Window - Add Content", self._bench_window_add_content),
            ("Context Window - Get Usage Report", self._bench_window_usage_report),
            ("Context Window - Clear Section", self._bench_window_clear_section),
            ("Message Compressor - Light", self._bench_compressor_light),
            ("Message Compressor - Moderate", self._bench_compressor_moderate),
            ("Message Compressor - Aggressive", self._bench_compressor_aggressive),
            ("Strategy - Recency Selection", self._bench_strategy_recency),
            ("Strategy - Relevance Selection", self._bench_strategy_relevance),
            ("Strategy - Hybrid Selection", self._bench_strategy_hybrid),
            ("Context Manager - Add Content", self._bench_manager_add_content),
            ("Context Manager - Optimize Context", self._bench_manager_optimize),
            ("Context Profiler - Analyze Window", self._bench_profiler_analyze),
            ("Snapshot - Save/Restore", self._bench_snapshot_operations),
            ("Adaptive Budgets - Track Usage", self._bench_adaptive_tracking)
        ]
        
        for name, benchmark_func in benchmarks:
            print(f"⏱️  {name}...")
            result = self._run_benchmark(benchmark_func)
            self.results[name] = result
            self._print_result(name, result)
        
        print("\n📊 Benchmark Summary:")
        self._print_summary()
        
        return self.results
    
    def _run_benchmark(self, benchmark_func: Callable) -> Dict:
        """Run a single benchmark with warmup."""
        # Warmup
        for _ in range(self.warmup_iterations):
            benchmark_func()
        
        # Actual benchmark
        times = []
        for _ in range(self.benchmark_iterations):
            start_time = time.perf_counter()
            benchmark_func()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Calculate statistics
        total_time = sum(times)
        avg_time = total_time / len(times)
        ops_per_sec = 1.0 / avg_time if avg_time > 0 else float('inf')
        
        return {
            'avg_time_ms': avg_time * 1000,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000,
            'std_dev_ms': statistics.stdev(times) * 1000 if len(times) > 1 else 0.0,
            'ops_per_sec': ops_per_sec,
            'total_iterations': len(times)
        }
    
    def _print_result(self, name: str, result: Dict):
        """Print benchmark result."""
        print(f"   {result['ops_per_sec']:.0f} ops/sec "
              f"(avg: {result['avg_time_ms']:.3f}ms, "
              f"std: {result['std_dev_ms']:.3f}ms)")
    
    def _print_summary(self):
        """Print benchmark summary."""
        total_ops = sum(r['ops_per_sec'] for r in self.results.values())
        print(f"   Total throughput: {total_ops:.0f} combined ops/sec")
        
        # Find fastest and slowest operations
        by_ops = [(name, r['ops_per_sec']) for name, r in self.results.items()]
        by_ops.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   Fastest: {by_ops[0][0]} ({by_ops[0][1]:.0f} ops/sec)")
        print(f"   Slowest: {by_ops[-1][0]} ({by_ops[-1][1]:.0f} ops/sec)")
    
    def _generate_test_content(self) -> List[str]:
        """Generate test content of various sizes."""
        return [
            "Short message.",
            "Medium length message with some more content to test compression and processing.",
            """Long message with multiple sentences and paragraphs.
            
            This content should be long enough to trigger various compression
            strategies and test the performance of different algorithms.
            
            It includes multiple lines, different punctuation marks, and
            various content patterns that the system might encounter in
            real-world usage scenarios.
            
            The goal is to have realistic test data that represents the
            kind of content that would actually be processed by the
            context management system in production environments.""",
            "Another short one.",
            "Yet another medium-sized message for testing purposes.",
        ]
    
    def _generate_test_messages(self) -> List[Dict]:
        """Generate test message list."""
        messages = []
        roles = ['user', 'assistant', 'system', 'tool']
        
        for i in range(20):
            messages.append({
                'role': roles[i % len(roles)],
                'content': self.test_content[i % len(self.test_content)],
                'priority': ['normal', 'important', 'optional'][i % 3]
            })
        
        return messages
    
    # Benchmark implementations
    def _bench_window_add_content(self):
        """Benchmark context window add content."""
        window = ContextWindow(8000)
        window.set_section_budget('conversation', 4000)
        
        for content in self.test_content[:3]:  # Use subset to avoid overflow
            window.add_content('conversation', content, 'normal')
    
    def _bench_window_usage_report(self):
        """Benchmark window usage report generation."""
        window = ContextWindow(8000)
        window.set_section_budget('conversation', 4000)
        
        # Add some content first
        for content in self.test_content[:2]:
            window.add_content('conversation', content, 'normal')
        
        # Generate report (this is what we're benchmarking)
        window.get_usage_report()
    
    def _bench_window_clear_section(self):
        """Benchmark clearing a section."""
        window = ContextWindow(8000)
        window.set_section_budget('conversation', 4000)
        
        # Add content then clear
        for content in self.test_content[:3]:
            window.add_content('conversation', content, 'normal')
        
        window.clear_section('conversation')
    
    def _bench_compressor_light(self):
        """Benchmark light compression."""
        compressor = MessageCompressor('light')
        for content in self.test_content:
            compressor.compress(content)
    
    def _bench_compressor_moderate(self):
        """Benchmark moderate compression.""" 
        compressor = MessageCompressor('moderate')
        for content in self.test_content:
            compressor.compress(content)
    
    def _bench_compressor_aggressive(self):
        """Benchmark aggressive compression."""
        compressor = MessageCompressor('aggressive')
        for content in self.test_content:
            compressor.compress(content)
    
    def _bench_strategy_recency(self):
        """Benchmark recency strategy."""
        strategy = RecencyStrategy()
        
        # Create content items
        content_items = []
        for i, content in enumerate(self.test_content):
            content_items.append({
                'content': content,
                'tokens': len(content) // 4,
                'priority': 'normal',
                'added_at': i
            })
        
        strategy.select_content(content_items, 500)
    
    def _bench_strategy_relevance(self):
        """Benchmark relevance strategy."""
        strategy = RelevanceStrategy()
        
        # Create content items  
        content_items = []
        for content in self.test_content:
            content_items.append({
                'content': content,
                'tokens': len(content) // 4,
                'priority': 'normal',
                'added_at': 0
            })
        
        strategy.select_content(content_items, 500, "test message compression")
    
    def _bench_strategy_hybrid(self):
        """Benchmark hybrid strategy."""
        strategy = HybridStrategy()
        
        # Create content items
        content_items = []
        for i, content in enumerate(self.test_content):
            content_items.append({
                'content': content,
                'tokens': len(content) // 4,
                'priority': 'normal',
                'added_at': i
            })
        
        strategy.select_content(content_items, 500, "test hybrid selection")
    
    def _bench_manager_add_content(self):
        """Benchmark context manager add content."""
        manager = ContextManager(8000)
        
        for content in self.test_content[:3]:
            manager.add_content('conversation', content, 'normal')
    
    def _bench_manager_optimize(self):
        """Benchmark context optimization."""
        manager = ContextManager(8000)
        
        # Add some content first
        for content in self.test_content:
            manager.add_content('conversation', content, 'normal')
        
        manager.optimize_context()
    
    def _bench_profiler_analyze(self):
        """Benchmark profiler analysis."""
        profiler = ContextProfiler()
        window = ContextWindow(8000)
        window.set_section_budget('conversation', 4000)
        
        # Add some content
        for content in self.test_content[:3]:
            window.add_content('conversation', content, 'normal')
        
        profiler.analyze_window(window)
    
    def _bench_snapshot_operations(self):
        """Benchmark snapshot save/restore."""
        manager = ContextManager(8000)
        
        # Add some content
        for content in self.test_content[:2]:
            manager.add_content('conversation', content, 'normal')
        
        # Save and restore snapshot
        manager.save_snapshot('test')
        manager.restore_snapshot('test')
    
    def _bench_adaptive_tracking(self):
        """Benchmark adaptive budget tracking."""
        manager = ContextManager(8000)
        manager.enable_adaptive_budgets(True)
        
        # Add some content
        for content in self.test_content[:2]:
            manager.add_content('conversation', content, 'normal')
        
        # Track usage (this is what we're benchmarking)
        manager.track_usage()


def main():
    """Run benchmarks from command line."""
    benchmark = ContextBenchmark()
    results = benchmark.run_all_benchmarks()
    
    # Optional: Save results to file
    import json
    with open('benchmark_results.json', 'w') as f:
        # Convert any non-serializable values
        serializable_results = {}
        for name, result in results.items():
            serializable_results[name] = {
                k: v if not (isinstance(v, float) and (v == float('inf') or v != v)) else None
                for k, v in result.items()
            }
        json.dump({
            'timestamp': time.time(),
            'results': serializable_results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to benchmark_results.json")


if __name__ == '__main__':
    main()