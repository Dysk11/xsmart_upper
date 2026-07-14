"""Asynchronous Qianfan-backed route decisions for accepted OCR events."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
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
class QianfanRouteConfig:
    enable: bool = True
    api_url: str = "https://qianfan.baidubce.com/v2/chat/completions"
    model: str = "ernie-4.5-turbo-vl"
    api_key_env: str = "QIANFAN_API_KEY"
    connect_timeout_sec: float = 2.0
    read_timeout_sec: float = 8.0
    max_attempts: int = 2
    retry_interval_sec: float = 0.5
    fallback_direction: str = "left"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "QianfanRouteConfig":
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
            fallback_direction=str(source.get("fallback_direction", cls.fallback_direction)).strip().casefold(),
        )
        if not config.api_url:
            raise ValueError("extensions.qianfan_route.api_url must not be empty")
        if not config.model:
            raise ValueError("extensions.qianfan_route.model must not be empty")
        if not config.api_key_env:
            raise ValueError("extensions.qianfan_route.api_key_env must not be empty")
        if config.connect_timeout_sec <= 0:
            raise ValueError("extensions.qianfan_route.connect_timeout_sec must be greater than zero")
        if config.read_timeout_sec <= 0:
            raise ValueError("extensions.qianfan_route.read_timeout_sec must be greater than zero")
        if config.max_attempts < 1:
            raise ValueError("extensions.qianfan_route.max_attempts must be at least one")
        if config.retry_interval_sec < 0:
            raise ValueError("extensions.qianfan_route.retry_interval_sec must not be negative")
        if config.fallback_direction not in {"left", "right"}:
            raise ValueError("extensions.qianfan_route.fallback_direction must be left or right")
        return config


@dataclass(frozen=True)
class QianfanRouteRequest:
    event_id: int
    ocr_text: str


@dataclass(frozen=True)
class QianfanRouteDecision:
    event_id: int
    status: str
    direction: str
    raw_answer: str = ""
    attempts: int = 0
    elapsed_sec: float = 0.0
    error: str = ""
    fallback: bool = False


def build_question(ocr_text: str) -> str:
    return f"{ocr_text}{QUESTION_SUFFIX}"


def parse_direction(answer: Any) -> str | None:
    if not isinstance(answer, str):
        return None
    normalized = answer.strip().casefold()
    return normalized if normalized in {"left", "right"} else None


class QianfanRouteClient:
    """Call Qianfan with bounded retries and return a safe route decision."""

    def __init__(
        self,
        config: QianfanRouteConfig | Mapping[str, Any],
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
        attempt_logger: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config if isinstance(config, QianfanRouteConfig) else QianfanRouteConfig.from_mapping(config)
        self.session = session or requests.Session()
        self._owns_session = session is None
        self.sleep = sleep
        self.clock = clock
        self.attempt_logger = attempt_logger

    def decide(self, request: QianfanRouteRequest) -> QianfanRouteDecision:
        started_at = self.clock()
        api_key = os.getenv(self.config.api_key_env, "").strip()
        if not api_key:
            return self._fallback(request, 0, started_at, f"missing environment variable {self.config.api_key_env}")

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
        last_error = "unknown Qianfan error"
        last_answer = ""

        for attempt in range(1, self.config.max_attempts + 1):
            attempt_started = self.clock()
            retryable = True
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
                        self._log_attempt(request, attempt, attempt_started, last_answer, direction, "")
                        return QianfanRouteDecision(
                            event_id=request.event_id,
                            status="success",
                            direction=direction,
                            raw_answer=last_answer.strip(),
                            attempts=attempt,
                            elapsed_sec=self.clock() - started_at,
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

            self._log_attempt(request, attempt, attempt_started, last_answer, "", last_error)
            if not retryable or attempt >= self.config.max_attempts:
                return self._fallback(request, attempt, started_at, last_error, last_answer)
            if self.config.retry_interval_sec > 0:
                self.sleep(self.config.retry_interval_sec)

        return self._fallback(request, self.config.max_attempts, started_at, last_error, last_answer)

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def _fallback(
        self,
        request: QianfanRouteRequest,
        attempts: int,
        started_at: float,
        error: str,
        raw_answer: str = "",
    ) -> QianfanRouteDecision:
        return QianfanRouteDecision(
            event_id=request.event_id,
            status="fallback",
            direction=self.config.fallback_direction,
            raw_answer=raw_answer.strip(),
            attempts=attempts,
            elapsed_sec=self.clock() - started_at,
            error=error,
            fallback=True,
        )

    def _log_attempt(
        self,
        request: QianfanRouteRequest,
        attempt: int,
        started_at: float,
        raw_answer: Any,
        direction: str,
        error: str,
    ) -> None:
        if self.attempt_logger is None:
            return
        self.attempt_logger(
            {
                "event_id": request.event_id,
                "attempt": attempt,
                "max_attempts": self.config.max_attempts,
                "connect_timeout_sec": self.config.connect_timeout_sec,
                "read_timeout_sec": self.config.read_timeout_sec,
                "elapsed_sec": self.clock() - started_at,
                "question": build_question(request.ocr_text),
                "raw_answer": raw_answer if isinstance(raw_answer, str) else repr(raw_answer),
                "direction": direction,
                "error": error,
            }
        )


class QianfanRouteState:
    """Own one OCR-to-fork decision from submission until fork release."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.pending_event_id: int | None = None
        self.decision: QianfanRouteDecision | None = None
        self.last_submitted_event_id = 0
        self.fork_engaged = False

    @property
    def route_direction(self) -> str | None:
        return None if self.decision is None else self.decision.direction

    @property
    def idle(self) -> bool:
        return self.pending_event_id is None and self.decision is None

    def submit(self, event_id: int, text: str, output_queue: Any) -> bool:
        if (
            not self.enabled
            or not self.idle
            or event_id <= 0
            or event_id <= self.last_submitted_event_id
            or not text
        ):
            return False
        output_queue.put(QianfanRouteRequest(event_id=event_id, ocr_text=text))
        self.pending_event_id = event_id
        self.last_submitted_event_id = event_id
        return True

    def accept(self, result: QianfanRouteDecision) -> bool:
        if self.pending_event_id is None or result.event_id != self.pending_event_id:
            return False
        self.pending_event_id = None
        self.decision = result
        return True

    def should_stop(self, fork_result: Any) -> bool:
        if not bool(getattr(fork_result, "fork_detected", False)):
            return False
        if self.pending_event_id is not None:
            return True
        if self.decision is None:
            return False
        return getattr(fork_result, "selected_direction", None) != self.decision.direction

    def observe_fork(self, fork_result: Any) -> bool:
        fork_detected = bool(getattr(fork_result, "fork_detected", False))
        selected_direction = getattr(fork_result, "selected_direction", None)
        if (
            fork_detected
            and self.decision is not None
            and selected_direction == self.decision.direction
        ):
            self.fork_engaged = True
            return False
        if self.fork_engaged:
            self.pending_event_id = None
            self.decision = None
            self.fork_engaged = False
            return True
        return False
