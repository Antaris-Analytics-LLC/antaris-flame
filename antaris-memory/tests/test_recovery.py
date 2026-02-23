"""Tests for recovery configuration and memory restoration."""

import json
from datetime import datetime, timedelta
import pytest
from antaris_memory.recovery import RecoveryConfig, RecoveryManager


class TestRecoveryConfig:
    """Test recovery configuration options."""
    
    def test_smart_preset(self):
        """Test 'smart' preset defaults."""
        cfg = RecoveryConfig(recovery_mode="smart")
        assert cfg.mode == "smart"
        assert cfg.search_limit == 50
        assert cfg.time_window == "24h"
        assert cfg.channels == "all"
        assert cfg.inject == "cache"
    
    def test_minimal_preset(self):
        """Test 'minimal' preset defaults."""
        cfg = RecoveryConfig(recovery_mode="minimal")
        assert cfg.mode == "minimal"
        assert cfg.search_limit == 10
        assert cfg.time_window == "session"
        assert cfg.channels is None
        assert cfg.inject == "manual"
    
    def test_custom_search_limit(self):
        """Test custom search limit."""
        cfg = RecoveryConfig(recovery_search_limit=100)
        assert cfg.search_limit == 100
    
    def test_search_limit_clamping(self):
        """Test search limit validation."""
        cfg = RecoveryConfig(recovery_search_limit=5)  # Below min
        assert cfg.search_limit == 10  # Clamped to min
        
        cfg = RecoveryConfig(recovery_search_limit=2000)  # Above max
        assert cfg.search_limit == 999  # Clamped to max
    
    def test_custom_time_window(self):
        """Test custom time window."""
        cfg = RecoveryConfig(recovery_time_window="7d")
        assert cfg.time_window == "7d"
    
    def test_invalid_time_window(self):
        """Test invalid time window handling."""
        cfg = RecoveryConfig(recovery_time_window="invalid")
        assert cfg.time_window == "24h"  # Defaults to 24h
    
    def test_custom_channels(self):
        """Test custom channel specification."""
        cfg = RecoveryConfig(recovery_channels=["ch1", "ch2"])
        assert cfg.channels == ["ch1", "ch2"]
        
        cfg = RecoveryConfig(recovery_channels="all")
        assert cfg.channels == "all"
    
    def test_to_dict(self):
        """Test serialization."""
        cfg = RecoveryConfig(recovery_mode="smart", recovery_search_limit=75)
        d = cfg.to_dict()
        assert d["recovery_mode"] == "smart"
        assert d["recovery_search_limit"] == 75
        assert d["recovery_time_window"] == "24h"
        assert d["recovery_inject"] == "cache"
    
    def test_help_text(self):
        """Test help documentation exists."""
        help_text = RecoveryConfig.help()
        assert "RECOVERY CONFIGURATION" in help_text
        assert "smart" in help_text
        assert "full" in help_text
        assert "minimal" in help_text
        assert "recovery_search_limit" in help_text
        assert "recovery_time_window" in help_text


class TestRecoveryManager:
    """Test recovery manager functionality."""
    
    def test_init(self):
        """Test recovery manager initialization."""
        cfg = RecoveryConfig(recovery_mode="smart")
        manager = RecoveryManager(None, cfg)
        assert manager.config == cfg
        assert manager.recovered_cache == {}
    
    def test_inject_into_context_empty(self):
        """Test context injection with no recovered memories."""
        cfg = RecoveryConfig()
        manager = RecoveryManager(None, cfg)
        result = manager.inject_into_context()
        assert result == ""
    
    def test_inject_into_context_with_memories(self):
        """Test context injection with recovered memories."""
        cfg = RecoveryConfig()
        manager = RecoveryManager(None, cfg)
        manager.recovered_cache = {
            "memories": [
                {"id": "m1", "text": "memory 1"},
                {"id": "m2", "text": "memory 2"}
            ],
            "tags": {
                "discord": 1,
                "slack": 1
            }
        }
        
        result = manager.inject_into_context()
        assert "[MEMORY RESTORED]" in result
        assert "2 memories" in result
        assert "2 sources" in result
        assert "discord" in result
        assert "slack" in result


class TestRecoveryIntegration:
    """Integration tests for recovery in a memory store."""
    
    def test_recovery_config_in_memory_init(self):
        """Test that recovery config can be passed to MemorySystem."""
        # This is a placeholder for integration test
        # Actual test would require a real MemorySystem instance
        cfg = RecoveryConfig(recovery_mode="smart")
        assert cfg is not None
        assert cfg.search_limit == 50


class TestRecoveryStrategies:
    """Test different recovery strategies for different use cases."""
    
    def test_strategy_token_efficiency(self):
        """Test minimal strategy for token efficiency."""
        cfg = RecoveryConfig(recovery_mode="minimal")
        # Minimal: 10 memories, session only, current channel, manual
        assert cfg.search_limit == 10
        assert cfg.time_window == "session"
        assert cfg.inject == "manual"
    
    def test_strategy_balanced(self):
        """Test smart strategy for balanced usage."""
        cfg = RecoveryConfig(recovery_mode="smart")
        # Smart: 50 memories, 24h, all channels, cache
        assert cfg.search_limit == 50
        assert cfg.time_window == "24h"
        assert cfg.inject == "cache"
    
    def test_custom_strategy(self):
        """Test creating a custom strategy."""
        cfg = RecoveryConfig(
            recovery_search_limit=100,
            recovery_time_window="24h",
            recovery_channels=["discord", "slack"],
            recovery_inject="cache"
        )
        assert cfg.search_limit == 100
        assert cfg.time_window == "24h"
        assert cfg.channels == ["discord", "slack"]
        assert cfg.inject == "cache"
