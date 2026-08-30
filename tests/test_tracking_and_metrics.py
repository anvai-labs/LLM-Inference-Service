from datetime import datetime, timedelta

import pytest

from ollama_server.core.request_tracker import RequestTracker
from ollama_server.core.schemas import InternalRequest
from ollama_server.utils.api_metrics import APIMetricsTracker


class Manager:
    @staticmethod
    def estimate_tokens(text: str) -> int:
        return len(text) // 4


def request(request_id: str = "r1") -> InternalRequest:
    return InternalRequest(request_id, "test", "12345678", 4096, 10, 0.8, 2, None, -1, False, {})


def test_request_tracker_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    class Timer:
        def __init__(self, *_args: object, **_kwargs: object):
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr("ollama_server.core.request_tracker.threading.Timer", Timer)
    tracker = RequestTracker(Manager())
    tracker.add_request(request())
    tracker.add_request(request())
    assert tracker.get_request("r1") is not None
    assert tracker.get_request("r1").prompt_tokens == 2  # type: ignore[union-attr]

    tracker.update_request("r1", status="completed", actual_tokens=4)
    assert len(tracker.get_recent_requests()) == 2
    tracker.get_request("r1").last_update = 0  # type: ignore[union-attr]
    assert tracker.remove_completed(older_than_seconds=0) == 1
    assert tracker.get_all_requests() == {}
    tracker.remove_request("missing")
    tracker.update_request("missing", status="error")


def test_metrics_track_status_rates_charts_and_reset() -> None:
    metrics = APIMetricsTracker()
    metrics.record_request("openai", 100, success=True)
    metrics.record_request("openai", 6000, success=False)
    metrics.record_request("missing", 1)

    endpoint = metrics.endpoints["openai"]
    assert endpoint.total_requests == 2
    assert endpoint.success_rate == 50
    assert endpoint.error_rate == 50
    assert endpoint.status == "error"
    assert metrics.get_requests_per_minute("missing") == 0
    metrics.recent_requests["openai"].append(datetime.now() - timedelta(minutes=2))
    assert metrics.get_requests_per_minute("openai") == 2

    summary = metrics.get_metrics_summary()
    charts = metrics.get_endpoint_data_for_charts()
    assert summary["openai"]["total_requests"] == 2
    assert next(item for item in charts if item["name"] == "OpenAI API")["color"]
    metrics.reset_metrics()
    assert metrics.endpoints["openai"].total_requests == 0
