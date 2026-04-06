"""Deadline-aware scheduling utilities for speech/TTS pipelines.

Ported from vox-serve's OnlineScheduler priority logic to prevent audio
underrun during real-time playback.  The core idea: a streaming request
is **pressing** when the client is about to run out of buffered audio,
and pressing requests should be prioritised in batch composition.

This module is intentionally a standalone utility — it does not import
engine internals so it can be unit-tested in isolation.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Lightweight protocol so we don't depend on OrchestratorRequestState
# ---------------------------------------------------------------------------


class SpeechRequestState(Protocol):
    """Minimal interface a request state must expose for pressing-status."""

    is_speech_streaming: bool
    chunk_send_timestamps: list[float]
    chunk_durations: list[float]
    is_pressing: bool


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SpeechSchedulingConfig:
    """Tuning knobs for deadline-aware speech scheduling."""

    # How far ahead of the playback cursor (in seconds) a request is still
    # considered "not pressing".  Larger values are more conservative.
    playback_buffer_s: float = 1.0


# ---------------------------------------------------------------------------
# Core tracker
# ---------------------------------------------------------------------------


class SpeechRequestPriorityTracker:
    """Computes ``is_pressing`` for speech streaming requests.

    Algorithm (from vox-serve ``OnlineScheduler._update_pressing_status``):

    A request is **pressing** when:
    1. No audio chunks have been sent yet (client has nothing to play).
    2. The client's playback cursor has caught up to the latest chunk —
       i.e. the client will run out of audio within ``playback_buffer_s``.

    The playback cursor is estimated as:
        playback_end = first_chunk_send_time + sum(all_chunk_durations)
        latest_chunk_start = playback_end - last_chunk_duration
        pressing = now >= latest_chunk_start - buffer
    """

    def __init__(self, config: SpeechSchedulingConfig | None = None) -> None:
        self.config = config or SpeechSchedulingConfig()

    def update_pressing_status(
        self,
        states: dict[str, SpeechRequestState],
        current_time: float | None = None,
    ) -> None:
        """Update ``is_pressing`` for every speech-streaming request in *states*.

        Non-speech requests are left untouched.
        """
        if current_time is None:
            current_time = _time.time()

        for state in states.values():
            if not state.is_speech_streaming:
                continue

            if not state.chunk_send_timestamps:
                # Nothing sent yet — urgent by definition.
                state.is_pressing = True
                continue

            first_send = state.chunk_send_timestamps[0]
            total_playback = sum(state.chunk_durations)
            if not state.chunk_durations:
                state.is_pressing = True
                continue

            # When the latest chunk *starts* playing at the client
            latest_chunk_start = first_send + total_playback - state.chunk_durations[-1]

            state.is_pressing = current_time >= latest_chunk_start - self.config.playback_buffer_s

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_chunk_duration_s(
        audio_bytes: int,
        sample_rate: int = 24000,
        channels: int = 1,
        bytes_per_sample: int = 2,
    ) -> float:
        """Compute audio duration in seconds from raw PCM byte count."""
        num_samples = audio_bytes // (channels * bytes_per_sample)
        return num_samples / sample_rate
