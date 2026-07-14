from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests

from core.qianfan_route import (
    QUESTION_SUFFIX,
    QianfanRouteClient,
    QianfanRouteConfig,
    QianfanRouteDecision,
    QianfanRouteRequest,
    QianfanRouteState,
    build_question,
    parse_direction,
)


class FakeResponse:
    def __init__(self, status_code: int, content: str = "left", text: str = "") -> None:
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = text
        self.content = content

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


class FakeSession:
    def __init__(self, results) -> None:
        self.results = list(results)
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeQueue:
    def __init__(self) -> None:
        self.items = []

    def put(self, item) -> None:
        self.items.append(item)


def test_question_and_direction_parser_are_strict() -> None:
    assert build_question("右边危险") == "右边危险" + QUESTION_SUFFIX
    assert parse_direction(" LEFT\n") == "left"
    assert parse_direction("Right") == "right"
    assert parse_direction("left，因为右边危险") is None
    assert parse_direction(123) is None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("connect_timeout_sec", 0),
        ("read_timeout_sec", -1),
        ("max_attempts", 0),
        ("retry_interval_sec", -0.1),
    ],
)
def test_config_rejects_invalid_retry_values(key: str, value: float) -> None:
    with pytest.raises(ValueError):
        QianfanRouteConfig.from_mapping({key: value})


def test_config_defaults_match_rk3588_runtime_policy() -> None:
    config = QianfanRouteConfig.from_mapping({})
    assert config.connect_timeout_sec == 2.0
    assert config.read_timeout_sec == 8.0
    assert config.max_attempts == 2
    assert config.retry_interval_sec == 0.5
    assert config.fallback_direction == "left"


def test_custom_timeout_and_attempt_count_are_passed_to_http(monkeypatch) -> None:
    monkeypatch.setenv("QIANFAN_API_KEY", "secret")
    session = FakeSession([FakeResponse(200, "right")])
    config = QianfanRouteConfig.from_mapping(
        {"connect_timeout_sec": 1.25, "read_timeout_sec": 3.5, "max_attempts": 4}
    )
    result = QianfanRouteClient(config, session=session).decide(QianfanRouteRequest(7, "左边封路"))

    assert result.direction == "right"
    assert result.status == "success"
    assert session.calls[0][1]["timeout"] == (1.25, 3.5)
    assert session.calls[0][1]["json"]["model"] == "ernie-4.5-turbo-vl"


def test_retryable_timeout_then_success(monkeypatch) -> None:
    monkeypatch.setenv("QIANFAN_API_KEY", "secret")
    session = FakeSession([requests.exceptions.ReadTimeout(), FakeResponse(200, "left")])
    sleeps = []
    client = QianfanRouteClient(
        {"max_attempts": 2, "retry_interval_sec": 0.25},
        session=session,
        sleep=sleeps.append,
    )

    result = client.decide(QianfanRouteRequest(1, "测试"))

    assert result.direction == "left"
    assert result.attempts == 2
    assert not result.fallback
    assert sleeps == [0.25]


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_retryable_http_status_then_success(monkeypatch, status_code: int) -> None:
    monkeypatch.setenv("QIANFAN_API_KEY", "secret")
    session = FakeSession([FakeResponse(status_code, text="temporary"), FakeResponse(200, "right")])
    result = QianfanRouteClient(
        {"max_attempts": 2, "retry_interval_sec": 0}, session=session
    ).decide(QianfanRouteRequest(8, "测试"))

    assert result.direction == "right"
    assert result.attempts == 2
    assert len(session.calls) == 2


def test_invalid_answers_exhaust_attempts_and_fallback_left(monkeypatch) -> None:
    monkeypatch.setenv("QIANFAN_API_KEY", "secret")
    session = FakeSession([FakeResponse(200, "左"), FakeResponse(200, "left because safe")])
    result = QianfanRouteClient(
        {"max_attempts": 2, "retry_interval_sec": 0}, session=session
    ).decide(QianfanRouteRequest(2, "测试"))

    assert result.direction == "left"
    assert result.status == "fallback"
    assert result.fallback
    assert result.attempts == 2


@pytest.mark.parametrize("status_code", [401, 403])
def test_authorization_errors_do_not_retry(monkeypatch, status_code: int) -> None:
    monkeypatch.setenv("QIANFAN_API_KEY", "secret")
    session = FakeSession([FakeResponse(status_code), FakeResponse(200, "right")])
    result = QianfanRouteClient({"max_attempts": 2}, session=session).decide(
        QianfanRouteRequest(3, "测试")
    )

    assert result.fallback
    assert result.attempts == 1
    assert len(session.calls) == 1


def test_missing_api_key_immediately_falls_back(monkeypatch) -> None:
    monkeypatch.delenv("QIANFAN_API_KEY", raising=False)
    session = FakeSession([FakeResponse(200, "right")])
    result = QianfanRouteClient({}, session=session).decide(QianfanRouteRequest(4, "测试"))

    assert result.direction == "left"
    assert result.attempts == 0
    assert len(session.calls) == 0


def test_route_state_deduplicates_and_clears_after_fork_release() -> None:
    state = QianfanRouteState()
    queue = FakeQueue()
    assert state.submit(5, "右边危险", queue)
    assert not state.submit(5, "重复", queue)
    assert len(queue.items) == 1

    pending_fork = SimpleNamespace(fork_detected=True, selected_direction=None)
    assert state.should_stop(pending_fork)
    assert not state.observe_fork(pending_fork)
    assert not state.observe_fork(SimpleNamespace(fork_detected=False, selected_direction=None))
    assert state.pending_event_id == 5
    decision = QianfanRouteDecision(5, "success", "left")
    assert state.accept(decision)
    assert state.should_stop(pending_fork)

    selected_fork = SimpleNamespace(fork_detected=True, selected_direction="left")
    assert not state.should_stop(selected_fork)
    assert not state.observe_fork(selected_fork)
    assert state.observe_fork(SimpleNamespace(fork_detected=False, selected_direction=None))
    assert state.idle
    assert state.route_direction is None
