"""Tests for deadline-aware speech scheduling utilities."""

from dataclasses import dataclass, field

import pytest

from vllm_omni.engine.speech_scheduling import (
    SpeechRequestPriorityTracker,
    SpeechSchedulingConfig,
)

pytestmark = [pytest.mark.cpu]


@dataclass
class _FakeState:
    """Minimal SpeechRequestState for testing."""

    is_speech_streaming: bool = True
    chunk_send_timestamps: list[float] = field(default_factory=list)
    chunk_durations: list[float] = field(default_factory=list)
    is_pressing: bool = False


class TestPressingStatusCalculation:
    def test_pressing_when_no_chunks_sent(self):
        tracker = SpeechRequestPriorityTracker()
        state = _FakeState()
        states = {"r1": state}
        tracker.update_pressing_status(states, current_time=100.0)
        assert state.is_pressing is True

    def test_pressing_when_playback_caught_up(self):
        """Client has played all buffered audio — pressing."""
        tracker = SpeechRequestPriorityTracker(
            SpeechSchedulingConfig(playback_buffer_s=1.0)
        )
        state = _FakeState(
            chunk_send_timestamps=[10.0, 10.5],
            chunk_durations=[0.5, 0.5],
        )
        states = {"r1": state}
        # Playback end = 10.0 + 0.5 + 0.5 = 11.0
        # Latest chunk start = 11.0 - 0.5 = 10.5
        # Pressing if now >= 10.5 - 1.0 = 9.5
        tracker.update_pressing_status(states, current_time=11.0)
        assert state.is_pressing is True

    def test_not_pressing_when_well_buffered(self):
        """Client has plenty of audio buffered — not pressing."""
        tracker = SpeechRequestPriorityTracker(
            SpeechSchedulingConfig(playback_buffer_s=1.0)
        )
        state = _FakeState(
            chunk_send_timestamps=[100.0, 100.5, 101.0, 101.5],
            chunk_durations=[0.5, 0.5, 0.5, 0.5],
        )
        states = {"r1": state}
        # Playback end = 100.0 + 2.0 = 102.0
        # Latest chunk start = 102.0 - 0.5 = 101.5
        # Pressing if now >= 101.5 - 1.0 = 100.5
        # At time 100.0, not pressing
        tracker.update_pressing_status(states, current_time=100.0)
        assert state.is_pressing is False

    def test_non_speech_request_untouched(self):
        """Non-speech requests should not have is_pressing modified."""
        tracker = SpeechRequestPriorityTracker()
        state = _FakeState(is_speech_streaming=False, is_pressing=False)
        states = {"r1": state}
        tracker.update_pressing_status(states, current_time=100.0)
        assert state.is_pressing is False

    def test_multiple_requests_independent(self):
        """Each request's pressing status computed independently."""
        tracker = SpeechRequestPriorityTracker(
            SpeechSchedulingConfig(playback_buffer_s=1.0)
        )
        pressing_state = _FakeState()  # No chunks — pressing
        buffered_state = _FakeState(
            chunk_send_timestamps=[100.0, 100.5, 101.0],
            chunk_durations=[0.5, 0.5, 0.5],
        )
        states = {"r1": pressing_state, "r2": buffered_state}
        # r2: playback end = 100.0 + 1.5 = 101.5
        # Latest chunk start = 101.5 - 0.5 = 101.0
        # Pressing if now >= 101.0 - 1.0 = 100.0
        # At time 99.0, r2 not pressing
        tracker.update_pressing_status(states, current_time=99.0)
        assert pressing_state.is_pressing is True
        assert buffered_state.is_pressing is False

    def test_configurable_buffer(self):
        """Larger buffer makes pressing trigger earlier."""
        state = _FakeState(
            chunk_send_timestamps=[100.0],
            chunk_durations=[2.0],
        )
        states = {"r1": state}

        # With 1s buffer: latest_chunk_start = 100.0, pressing if now >= 99.0
        tracker_1s = SpeechRequestPriorityTracker(
            SpeechSchedulingConfig(playback_buffer_s=1.0)
        )
        tracker_1s.update_pressing_status(states, current_time=98.5)
        assert state.is_pressing is False

        # With 3s buffer: pressing if now >= 100.0 - 3.0 = 97.0
        tracker_3s = SpeechRequestPriorityTracker(
            SpeechSchedulingConfig(playback_buffer_s=3.0)
        )
        tracker_3s.update_pressing_status(states, current_time=98.5)
        assert state.is_pressing is True


class TestChunkDurationCalculation:
    def test_basic_calculation(self):
        # 48000 bytes at 24kHz mono 16-bit = 24000 samples = 1.0s
        duration = SpeechRequestPriorityTracker.calculate_chunk_duration_s(48000)
        assert duration == pytest.approx(1.0)

    def test_small_chunk(self):
        # 4800 bytes = 2400 samples = 0.1s
        duration = SpeechRequestPriorityTracker.calculate_chunk_duration_s(4800)
        assert duration == pytest.approx(0.1)

    def test_stereo(self):
        # 96000 bytes at 24kHz stereo 16-bit = 24000 samples = 1.0s
        duration = SpeechRequestPriorityTracker.calculate_chunk_duration_s(
            96000, channels=2
        )
        assert duration == pytest.approx(1.0)

    def test_zero_bytes(self):
        duration = SpeechRequestPriorityTracker.calculate_chunk_duration_s(0)
        assert duration == 0.0
