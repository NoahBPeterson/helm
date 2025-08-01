import pytest
from helm.benchmark.scenarios.lisan_bench import _levenshtein1, _normalize

def test_levenshtein1():
    assert _levenshtein1("stone","stoke")
    assert not _levenshtein1("stone","stone")
    assert not _levenshtein1("stone","tones")
    assert not _levenshtein1("stone","stonyy")

def test_normalize():
    assert _normalize(" Stone! ") == "stone"
