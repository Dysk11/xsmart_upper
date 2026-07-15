"""Asynchronous road-sign analysis backed by the configured Qianfan API."""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping

import requests


QUESTION_SUFFIX = (
    "只有“左”和“右”两个方向，回答最终要走哪个方向。"
    "只用“left”或“right”回答。"
)
SYSTEM_PROMPT = (
    "你是岔路方向决策助手。必须严格遵守输出格式，"
    "最终只能输出left或right，不得输出解释或其他文字。"
)


@dataclass(frozen=True)
class RoadSignAnalyzerConfig:
    enable: bool = True
    api_url: str = "https://qianfan.baidubce.com/v2/chat/completions"
    model: str = "ernie-4.5-turbo-vl"
    api_key_env: str = "QIANFAN_API_KEY"
    connect_timeout_sec: float = 2.0
    read_timeout_sec: float = 8.0
    max_attempts: int = 2
    retry_interval_sec: float = 0.5
    default_direction: str = "left"
    decision_ttl_sec: float = 20.0
    output_dir: str = "outputs/logs/api"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RoadSignAnalyzerConfig":
        source = value or {}
        config = cls(
            enable=bool(source.get("enable", True)),
            api_url=str(source.get("api_url", cls.api_url)).strip(),
            model=str(source.get("model", cls.model)).strip(),
            api_key_env=str(source.get("api_key_env", cls.api_key_env)).strip(),
            connect_timeout_sec=float(source.get("connect_timeout_sec", cls.connect_timeout_sec)),
            read_timeout_sec=float(source.get("read_timeout_sec", cls.read_timeout_sec)),
            max_attempts=int(source.get("max_attempts", cls.max_attempts)),
            retry_interval_sec=float(source.get("retry_interval_sec", cls.retry_interval_sec)),
            default_direction=str(source.get("default_direction", cls.default_direction)).strip().casefold(),
            decision_ttl_sec=float(source.get("decision_ttl_sec", cls.decision_ttl_sec)),
            output_dir=str(source.get("output_dir", cls.output_dir)).strip(),
        )
        if not config.api_url:
            raise ValueError("road_sign_analyzer.api_url must not be empty")
        if not config.model:
            raise ValueError("road_sign_analyzer.model must not be empty")
        if not config.api_key_env:
            raise ValueError("road_sign_analyzer.api_key_env must not be empty")
        if config.connect_timeout_sec <= 0:
            raise ValueError("road_sign_analyzer.connect_timeout_sec must be greater than zero")
        if config.read_timeout_sec <= 0:
            raise ValueError("road_sign_analyzer.read_timeout_sec must be greater than zero")
        if config.max_attempts < 1:
            raise ValueError("road_sign_analyzer.max_attempts must be at least one")
        if config.retry_interval_sec < 0:
            raise ValueError("road_sign_analyzer.retry_interval_sec must not be negative")
        if config.default_direction not in {"left", "right"}:
            raise ValueError("road_sign_analyzer.default_direction must be left or right")
        if config.decision_ttl_sec <= 0:
            raise ValueError("road_sign_analyzer.decision_ttl_sec must be greater than zero")
        if not config.output_dir:
            raise ValueError("road_sign_analyzer.output_dir must not be empty")
        return config


@dataclass(frozen=True)
class RoadSignAnalysisRequest:
    event_id: int
    ocr_text: str


@dataclass(frozen=True)
class RoadSignAnalysisDecision:
    event_id: int
    status: str
    direction: str
    raw_answer: str = ""
    attempts: int = 0
    elapsed_sec: float = 0.0
    error: str = ""
    fallback: bool = False
    attempt_records: tuple[dict[str, Any], ...] = ()


def build_question(ocr_text: str) -> str:
    return f"{ocr_text}{QUESTION_SUFFIX}"


def parse_direction(answer: Any) -> str | None:
    if not isinstance(answer, str):
        return None
    normalized = answer.strip().casefold()
    return normalized if normalized in {"left", "right"} else None


class RoadSignAnalysisLogger:
    """Append API attempts and final decisions to one UTF-8 JSONL file per run."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.file_path: Path | None = None

    def append_result(
        self,
        decision: RoadSignAnalysisDecision,
        decision_ttl_sec: float,
        received_at: datetime | None = None,
    ) -> Path:
        received = received_at or datetime.now().astimezone()
        for attempt in decision.attempt_records:
            self._append({
                "timestamp": received.isoformat(timespec="milliseconds"),
                "record_type": "attempt",
                **attempt,
            })
        expires = received + timedelta(seconds=float(decision_ttl_sec))
        return self._append({
            "timestamp": received.isoformat(timespec="milliseconds"),
            "record_type": "decision",
            "event_id": decision.event_id,
            "status": decision.status,
            "direction": decision.direction,
            "raw_answer": decision.raw_answer,
            "attempts": decision.attempts,
            "elapsed_sec": decision.elapsed_sec,
            "error": decision.error,
            "fallback": decision.fallback,
            "decision_ttl_sec": float(decision_ttl_sec),
            "expires_at": expires.isoformat(timespec="milliseconds"),
        })

    def _append(self, record: Mapping[str, Any]) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.file_path is None:
            timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
            self.file_path = self.output_dir / f"road_sign_analysis_events_{timestamp}.jsonl"
        with self.file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        return self.file_path


class RoadSignAnalyzer:
    """Call Qianfan with bounded retries and return a safe route decision."""

    def __init__(
        self,
        config: RoadSignAnalyzerConfig | Mapping[str, Any],
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
        attempt_logger: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config if isinstance(config, RoadSignAnalyzerConfig) else RoadSignAnalyzerConfig.from_mapping(config)
        self.session = session or requests.Session()
        self._owns_session = session is None
        self.sleep = sleep
        self.clock = clock
        self.attempt_logger = attempt_logger

    def decide(self, request: RoadSignAnalysisRequest) -> RoadSignAnalysisDecision:
        started_at = self.clock()
        attempt_records: list[dict[str, Any]] = []
        api_key = os.getenv(self.config.api_key_env, "").strip()
        if not api_key:
            return self._fallback(
                request,
                0,
                started_at,
                f"missing environment variable {self.config.api_key_env}",
                attempt_records=attempt_records,
            )

        question = build_question(request.ocr_text)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "stream": False,
        }
        last_error = "unknown road-sign analysis error"
        last_answer = ""

        for attempt in range(1, self.config.max_attempts + 1):
            attempt_started = self.clock()
            retryable = True
            status_code: int | None = None
            try:
                response = self.session.post(
                    self.config.api_url,
                    headers=headers,
                    json=payload,
                    timeout=(self.config.connect_timeout_sec, self.config.read_timeout_sec),
                )
                status_code = int(response.status_code)
                if status_code in {401, 403}:
                    last_error = f"HTTP {status_code}: authorization rejected"
                    retryable = False
                elif status_code == 429 or 500 <= status_code < 600:
                    last_error = f"HTTP {status_code}: {response.text}"
                elif not response.ok:
                    last_error = f"HTTP {status_code}: {response.text}"
                    retryable = False
                else:
                    data = response.json()
                    last_answer = data["choices"][0]["message"]["content"]
                    direction = parse_direction(last_answer)
                    if direction is not None:
                        attempt_records.append(
                            self._log_attempt(
                                request,
                                attempt,
                                attempt_started,
                                last_answer,
                                direction,
                                "",
                                status_code,
                            )
                        )
                        return RoadSignAnalysisDecision(
                            event_id=request.event_id,
                            status="success",
                            direction=direction,
                            raw_answer=last_answer.strip(),
                            attempts=attempt,
                            elapsed_sec=self.clock() - started_at,
                            attempt_records=tuple(attempt_records),
                        )
                    last_error = f"invalid answer: {last_answer!r}"
            except requests.exceptions.ConnectTimeout:
                last_error = "connection timeout"
            except requests.exceptions.ReadTimeout:
                last_error = "read timeout"
            except requests.exceptions.ConnectionError as exc:
                last_error = f"connection error: {exc}"
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                last_error = f"invalid response: {type(exc).__name__}: {exc}"
            except requests.exceptions.RequestException as exc:
                last_error = f"request error: {exc}"

            attempt_records.append(
                self._log_attempt(
                    request,
                    attempt,
                    attempt_started,
                    last_answer,
                    "",
                    last_error,
                    status_code,
                )
            )
            if not retryable or attempt >= self.config.max_attempts:
                return self._fallback(
                    request,
                    attempt,
                    started_at,
                    last_error,
                    last_answer,
                    attempt_records,
                )
            if self.config.retry_interval_sec > 0:
                self.sleep(self.config.retry_interval_sec)

        return self._fallback(
            request,
            self.config.max_attempts,
            started_at,
            last_error,
            last_answer,
            attempt_records,
        )

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def _fallback(
        self,
        request: RoadSignAnalysisRequest,
        attempts: int,
        started_at: float,
        error: str,
        raw_answer: str = "",
        attempt_records: list[dict[str, Any]] | None = None,
    ) -> RoadSignAnalysisDecision:
        return RoadSignAnalysisDecision(
            event_id=request.event_id,
            status="fallback",
            direction=self.config.default_direction,
            raw_answer=raw_answer.strip(),
            attempts=attempts,
            elapsed_sec=self.clock() - started_at,
            error=error,
            fallback=True,
            attempt_records=tuple(attempt_records or ()),
        )

    def _log_attempt(
        self,
        request: RoadSignAnalysisRequest,
        attempt: int,
        started_at: float,
        raw_answer: Any,
        direction: str,
        error: str,
        http_status: int | None,
    ) -> dict[str, Any]:
        details = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "event_id": request.event_id,
            "attempt": attempt,
            "max_attempts": self.config.max_attempts,
            "connect_timeout_sec": self.config.connect_timeout_sec,
            "read_timeout_sec": self.config.read_timeout_sec,
            "elapsed_sec": self.clock() - started_at,
            "question": build_question(request.ocr_text),
            "http_status": http_status,
            "raw_answer": raw_answer if isinstance(raw_answer, str) else repr(raw_answer),
            "direction": direction,
            "error": error,
        }
        if self.attempt_logger is not None:
            self.attempt_logger(details)
        return details


class RoadSignAnalysisState:
    """Keep the newest API direction active for a bounded time window."""

    def __init__(
        self,
        enabled: bool = True,
        default_direction: str = "left",
        decision_ttl_sec: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.enabled = bool(enabled)
        self.default_direction = str(default_direction).casefold()
        self.decision_ttl_sec = float(decision_ttl_sec)
        self.clock = clock
        if self.default_direction not in {"left", "right"}:
            raise ValueError("default_direction must be left or right")
        if self.decision_ttl_sec <= 0:
            raise ValueError("decision_ttl_sec must be greater than zero")
        self.pending_event_id: int | None = None
        self.decision: RoadSignAnalysisDecision | None = None
        self.decision_expires_at = 0.0
        self.last_submitted_event_id = 0

    @property
    def route_direction(self) -> str:
        self._expire_if_needed()
        return self.default_direction if self.decision is None else self.decision.direction

    @property
    def idle(self) -> bool:
        return self.pending_event_id is None

    def submit(self, event_id: int, text: str, output_queue: Any) -> bool:
        if (
            not self.enabled
            or self.pending_event_id is not None
            or event_id <= 0
            or event_id <= self.last_submitted_event_id
            or not text
        ):
            return False
        output_queue.put(RoadSignAnalysisRequest(event_id=event_id, ocr_text=text))
        self.pending_event_id = event_id
        self.last_submitted_event_id = event_id
        return True

    def accept(self, result: RoadSignAnalysisDecision) -> bool:
        if self.pending_event_id is None or result.event_id != self.pending_event_id:
            return False
        self.pending_event_id = None
        self.decision = result
        self.decision_expires_at = self.clock() + self.decision_ttl_sec
        return True

    def cancel_pending(self, reset_decision: bool = False) -> int | None:
        """Cancel an in-flight request so that a late response cannot be accepted."""

        cancelled_event_id = self.pending_event_id
        self.pending_event_id = None
        if reset_decision:
            self.decision = None
            self.decision_expires_at = 0.0
        return cancelled_event_id

    def should_stop(self, fork_result: Any) -> bool:
        if not bool(getattr(fork_result, "fork_detected", False)):
            return False
        if self.pending_event_id is not None:
            return True
        return getattr(fork_result, "selected_direction", None) != self.route_direction

    def _expire_if_needed(self) -> None:
        if self.decision is not None and self.clock() >= self.decision_expires_at:
            self.decision = None
            self.decision_expires_at = 0.0
