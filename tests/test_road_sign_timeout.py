from __future__ import annotations

import queue

from core.planning.road_sign_analyzer import (
    RoadSignAnalysisDecision,
    RoadSignAnalysisState,
)


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
