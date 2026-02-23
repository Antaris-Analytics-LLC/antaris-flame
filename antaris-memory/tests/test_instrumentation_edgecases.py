# tests/test_instrumentation_edgecases.py
"""Edge case tests for extract_memory_references — single-token, dedup."""

from antaris_memory.instrumentation import extract_memory_references


class _Entry:
    def __init__(self, h: str, content: str):
        self.hash = h
        self.content = content


class _Result:
    def __init__(self, entry):
        self.entry = entry


def test_extract_memory_references_single_token_respects_min_overlap():
    # Single word "api" should NOT match when min_overlap > len("api")
    e1 = _Entry("h_api", "api")
    text = "We should use the api for this."
    used = extract_memory_references(text, [_Result(e1)], min_overlap=6)
    assert used == []


def test_extract_memory_references_single_token_can_match_when_long_enough():
    # Single word "deployment" is >= 6 chars; should match if present
    e1 = _Entry("h_deploy", "deployment")
    text = "The deployment failed last night."
    used = extract_memory_references(text, [_Result(e1)], min_overlap=6)
    assert used == ["h_deploy"]


def test_extract_memory_references_dedup_ids_when_multiple_strategies_hit():
    # Memory content appears both via overlap and via key phrase path; should dedup.
    e1 = _Entry("h1", "Set systemd service to restart always on failure.")
    text = "Remember: set systemd service to restart always on failure."
    used = extract_memory_references(text, [_Result(e1)], min_overlap=6)
    assert used == ["h1"]
