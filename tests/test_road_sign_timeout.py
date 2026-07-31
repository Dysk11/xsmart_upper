from __future__ import annotations

import queue

from core.planning.road_sign_analyzer import (
    RoadSignAnalysisRequest,
    RoadSignAnalyzer,
    RoadSignAnalyzerConfig,
    RoadSignAnalysisDecision,
    RoadSignAnalysisState,
)


class SuccessfulResponse:
    status_code = 200
    ok = True
    text = ""

    @staticmethod
    def json() -> dict[str, object]:
        return {"choices": [{"message": {"content": "left"}}]}


class RecordingSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, **kwargs: object) -> SuccessfulResponse:
        self.calls.append({"url": url, **kwargs})
        return SuccessfulResponse()


def test_api_prompt_contains_route_meanings_and_keeps_ocr_in_user_message(
    monkeypatch,
) -> None:
    monkeypatch.setenv("QIANFAN_API_KEY", "test-key")
    session = RecordingSession()
    analyzer = RoadSignAnalyzer(
        RoadSignAnalyzerConfig(max_attempts=1),
        session=session,
    )

    decision = analyzer.decide(
        RoadSignAnalysisRequest(event_id=1, ocr_text="请选择近路")
    )

    assert decision.direction == "left"
    assert len(session.calls) == 1
    payload = session.calls[0]["json"]
    assert isinstance(payload, dict)
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    system_prompt = messages[0]["content"]
    assert "直道对应左道" in system_prompt
    assert "弯道对应右道" in system_prompt
    assert "右道是近路" in system_prompt
    assert "左道是远路" in system_prompt
    assert "请选择近路" not in system_prompt
    assert messages[1]["role"] == "user"
    user_prompt = messages[1]["content"]
    assert user_prompt.startswith("请选择近路")
    assert "只用“left”或“right”回答" in user_prompt


def test_cancelled_api_request_rejects_late_result_and_restores_left() -> None:
    state = RoadSignAnalysisState(default_direction="left")
    output_queue: queue.Queue[object] = queue.Queue()
    assert state.submit(2, "向右", output_queue)
    assert state.accept(
        RoadSignAnalysisDecision(event_id=2, status="ok", direction="right")
    )
    assert state.route_direction == "right"

    assert state.submit(3, "向左", output_queue)
    assert state.cancel_pending(reset_decision=True) == 3
    assert state.route_direction == "left"

    late = RoadSignAnalysisDecision(event_id=3, status="ok", direction="right")
    assert not state.accept(late)
    assert state.route_direction == "left"


def test_current_default_exposes_no_explicit_route_and_does_not_stop_at_fork() -> None:
    config = RoadSignAnalyzerConfig.from_mapping({})
    assert config.default_direction == "current"
    state = RoadSignAnalysisState(default_direction=config.default_direction)
    fork_result = type(
        "ForkResult",
        (),
        {"fork_detected": True, "selected_direction": "left"},
    )()

    assert state.route_direction is None
    assert not state.should_stop(fork_result)


def test_current_fallback_decision_and_ttl_expiry_restore_no_explicit_route() -> None:
    now = [10.0]
    state = RoadSignAnalysisState(
        default_direction="current",
        decision_ttl_sec=2.0,
        clock=lambda: now[0],
    )
    output_queue: queue.Queue[object] = queue.Queue()
    assert state.submit(1, "向右", output_queue)
    assert state.accept(
        RoadSignAnalysisDecision(event_id=1, status="ok", direction="right")
    )
    assert state.route_direction == "right"

    now[0] = 12.1
    assert state.route_direction is None

    assert state.submit(2, "无效", output_queue)
    assert state.accept(
        RoadSignAnalysisDecision(
            event_id=2,
            status="fallback",
            direction="current",
            fallback=True,
        )
    )
    assert state.route_direction is None
