"""Tests for pressing-status integration in the Orchestrator."""

from unittest.mock import MagicMock, patch

import pytest

from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState

pytestmark = [pytest.mark.cpu]


class TestOrchestratorSpeechPriorityTrackerInit:
    """Verify that the speech priority tracker is created for audio pipelines."""

    def _make_stage_client(self, final_output_type=None):
        client = MagicMock()
        client.final_output_type = final_output_type
        client.stage_type = "llm"
        return client

    def test_tracker_created_for_audio_pipeline(self):
        """Orchestrator should create a tracker when a stage produces audio."""
        clients = [
            self._make_stage_client("latent"),
            self._make_stage_client("audio"),
        ]
        with patch.object(Orchestrator, "__init__", lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
            orch.stage_clients = clients
            orch._is_speech_pipeline = any(
                getattr(sc, "final_output_type", None) == "audio"
                for sc in orch.stage_clients
            )
            assert orch._is_speech_pipeline is True

    def test_no_tracker_for_image_pipeline(self):
        """Orchestrator should NOT create a tracker when no audio stages."""
        clients = [
            self._make_stage_client("latent"),
            self._make_stage_client("image"),
        ]
        with patch.object(Orchestrator, "__init__", lambda self, *a, **kw: None):
            orch = Orchestrator.__new__(Orchestrator)
            orch.stage_clients = clients
            orch._is_speech_pipeline = any(
                getattr(sc, "final_output_type", None) == "audio"
                for sc in orch.stage_clients
            )
            assert orch._is_speech_pipeline is False


class TestPressingStatusUpdatedInLoop:
    """Verify that pressing status is computed for speech requests."""

    def test_pressing_status_set_on_new_speech_request(self):
        """A new speech request with no chunks should be marked pressing."""
        state = OrchestratorRequestState(
            request_id="r1",
            is_speech_streaming=True,
        )
        assert state.is_pressing is False  # default

        from vllm_omni.engine.speech_scheduling import SpeechRequestPriorityTracker

        tracker = SpeechRequestPriorityTracker()
        tracker.update_pressing_status({"r1": state})
        assert state.is_pressing is True  # no chunks sent

    def test_non_speech_request_not_affected(self):
        """Non-speech requests should not be marked pressing."""
        state = OrchestratorRequestState(
            request_id="r1",
            is_speech_streaming=False,
        )
        from vllm_omni.engine.speech_scheduling import SpeechRequestPriorityTracker

        tracker = SpeechRequestPriorityTracker()
        tracker.update_pressing_status({"r1": state})
        assert state.is_pressing is False
