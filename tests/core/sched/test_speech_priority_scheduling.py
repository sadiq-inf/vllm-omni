"""Tests for speech priority hints in OmniARScheduler."""

from collections import deque
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.cpu]


class _FakeRequest:
    """Lightweight stand-in for a Request in the running queue."""

    def __init__(self, request_id: str, is_pressing: bool = False):
        self.request_id = request_id
        self.is_pressing = is_pressing

    def __repr__(self):
        return f"_FakeRequest({self.request_id!r}, pressing={self.is_pressing})"


class TestSpeechPrioritySortLogic:
    """Test the sorting logic that reorders the running queue.

    These tests exercise the sort key in isolation — no real scheduler
    needed, just the same sort() call used in OmniARScheduler.schedule().
    """

    def _sort_by_pressing(self, requests):
        """Apply the same stable sort used in the scheduler."""
        requests.sort(
            key=lambda req: (0 if getattr(req, "is_pressing", False) else 1),
        )
        return requests

    def test_pressing_requests_sorted_first(self):
        reqs = [
            _FakeRequest("r1", is_pressing=False),
            _FakeRequest("r2", is_pressing=True),
            _FakeRequest("r3", is_pressing=False),
            _FakeRequest("r4", is_pressing=True),
        ]
        self._sort_by_pressing(reqs)
        ids = [r.request_id for r in reqs]
        assert ids == ["r2", "r4", "r1", "r3"]

    def test_stable_sort_preserves_fifo_within_priority(self):
        reqs = [
            _FakeRequest("r1", is_pressing=True),
            _FakeRequest("r2", is_pressing=True),
            _FakeRequest("r3", is_pressing=False),
            _FakeRequest("r4", is_pressing=False),
        ]
        self._sort_by_pressing(reqs)
        ids = [r.request_id for r in reqs]
        # Within each group, original order preserved
        assert ids == ["r1", "r2", "r3", "r4"]

    def test_all_pressing(self):
        reqs = [
            _FakeRequest("r1", is_pressing=True),
            _FakeRequest("r2", is_pressing=True),
        ]
        self._sort_by_pressing(reqs)
        ids = [r.request_id for r in reqs]
        assert ids == ["r1", "r2"]

    def test_no_pressing(self):
        reqs = [
            _FakeRequest("r1", is_pressing=False),
            _FakeRequest("r2", is_pressing=False),
        ]
        self._sort_by_pressing(reqs)
        ids = [r.request_id for r in reqs]
        assert ids == ["r1", "r2"]

    def test_single_request(self):
        reqs = [_FakeRequest("r1", is_pressing=True)]
        self._sort_by_pressing(reqs)
        assert reqs[0].request_id == "r1"

    def test_missing_is_pressing_attribute_treated_as_not_pressing(self):
        """Requests without is_pressing (e.g. non-OmniRequest) are not pressing."""
        plain = MagicMock(spec=["request_id"])
        plain.request_id = "plain"
        pressing = _FakeRequest("speech", is_pressing=True)

        reqs = [plain, pressing]
        self._sort_by_pressing(reqs)
        assert reqs[0].request_id == "speech"
        assert reqs[1].request_id == "plain"


class TestRunningQueueRestoration:
    """Test that the original running queue order is restored after scheduling."""

    def test_restore_removes_scheduled_requests(self):
        """After scheduling, the restore logic should exclude requests
        that were removed by the scheduler."""
        original = [
            _FakeRequest("r1"),
            _FakeRequest("r2"),
            _FakeRequest("r3"),
        ]
        # Simulate scheduler removing r2 from running
        current_running = [_FakeRequest("r1"), _FakeRequest("r3")]
        current_ids = {r.request_id for r in current_running}

        restored = [r for r in original if r.request_id in current_ids]
        ids = [r.request_id for r in restored]
        assert ids == ["r1", "r3"]
