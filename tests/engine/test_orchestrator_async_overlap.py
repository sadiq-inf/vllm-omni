"""Tests for async scheduling overlap in the Orchestrator."""

import pytest

from vllm_omni.engine.orchestrator import Orchestrator

pytestmark = [pytest.mark.cpu]


class TestAsyncOverlapConfig:
    """Verify the async_scheduling_overlap parameter is exposed."""

    def test_default_disabled(self):
        """Async overlap should be disabled by default."""
        # We only test the config flag exists and defaults correctly.
        # Full integration tests require a running engine.
        assert hasattr(Orchestrator.__init__, "__code__")
        # Check the parameter is in the signature
        import inspect

        sig = inspect.signature(Orchestrator.__init__)
        assert "async_scheduling_overlap" in sig.parameters
        param = sig.parameters["async_scheduling_overlap"]
        assert param.default is False

    def test_method_exists(self):
        """Overlapped polling method should exist."""
        assert hasattr(Orchestrator, "_poll_stages_overlapped")
        assert hasattr(Orchestrator, "_poll_stages_sequential")
