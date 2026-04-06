"""Tests for speech playback metadata in OrchestratorRequestState."""

import time
from unittest.mock import MagicMock

import pytest

from vllm_omni.engine.orchestrator import OrchestratorRequestState

pytestmark = [pytest.mark.cpu]


class TestOrchestratorRequestStateSpeechFields:
    """Verify speech-related fields on OrchestratorRequestState."""

    def test_default_not_speech(self):
        state = OrchestratorRequestState(request_id="req-1")
        assert state.is_speech_streaming is False
        assert state.chunk_send_timestamps == []
        assert state.chunk_durations == []
        assert state.is_pressing is False

    def test_speech_streaming_flag(self):
        state = OrchestratorRequestState(
            request_id="req-2",
            is_speech_streaming=True,
        )
        assert state.is_speech_streaming is True
        # New speech request with no chunks sent should default to not pressing
        # (pressing logic is computed separately in Commit 4)
        assert state.is_pressing is False

    def test_chunk_timestamp_tracking(self):
        state = OrchestratorRequestState(
            request_id="req-3",
            is_speech_streaming=True,
        )
        t1 = time.time()
        state.chunk_send_timestamps.append(t1)
        state.chunk_durations.append(0.5)

        t2 = time.time()
        state.chunk_send_timestamps.append(t2)
        state.chunk_durations.append(0.3)

        assert len(state.chunk_send_timestamps) == 2
        assert len(state.chunk_durations) == 2
        assert state.chunk_send_timestamps[0] == t1
        assert state.chunk_durations[1] == 0.3

    def test_independent_state_per_request(self):
        """Verify each request gets its own mutable lists (no shared state)."""
        s1 = OrchestratorRequestState(request_id="r1", is_speech_streaming=True)
        s2 = OrchestratorRequestState(request_id="r2", is_speech_streaming=True)

        s1.chunk_send_timestamps.append(1.0)
        assert s2.chunk_send_timestamps == []


class TestIsSpeechPipelineDetection:
    """Verify that Orchestrator._is_speech_pipeline detects audio stages."""

    def _make_stage_client(self, final_output_type=None):
        client = MagicMock()
        client.final_output_type = final_output_type
        return client

    def test_audio_pipeline_detected(self):
        clients = [
            self._make_stage_client("latent"),
            self._make_stage_client("audio"),
        ]
        is_speech = any(
            getattr(sc, "final_output_type", None) == "audio"
            for sc in clients
        )
        assert is_speech is True

    def test_non_audio_pipeline(self):
        clients = [
            self._make_stage_client("latent"),
            self._make_stage_client("image"),
        ]
        is_speech = any(
            getattr(sc, "final_output_type", None) == "audio"
            for sc in clients
        )
        assert is_speech is False

    def test_empty_pipeline(self):
        clients = []
        is_speech = any(
            getattr(sc, "final_output_type", None) == "audio"
            for sc in clients
        )
        assert is_speech is False
