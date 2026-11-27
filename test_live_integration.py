"""SafeMate 5-6단계 실제 환경 종합 측정 테스트 (5-6-A: 측정 도구·판정 기준·사전 검증)

실행 방법 (.venv-xpu)
- 기본 (M층·I층만, 네트워크 차단, 실제 모델 호출·모델 다운로드 없음):
    & .\\.venv-xpu\\Scripts\\python.exe test_live_integration.py -v
- 실제 측정 (L층, 묶음별로 하나씩 실행. 동시에 두 묶음을 실행하지 않는다):
    & .\\.venv-xpu\\Scripts\\python.exe test_live_integration.py --live-ollama -v L_LV1_RepresentativeFlow
    & .\\.venv-xpu\\Scripts\\python.exe test_live_integration.py --live-ollama --live-real -v L_LV8_RealData
  (이미 같은 묶음의 5-6 기록이 있으면 --rerun-reason "사유" 가 있어야 다시 실행한다. 이전 기록은 지우지 않는다.)

테스트 층
- M층: 판정·통계·기록·잠금·재측정 사유·재현 도구의 단위 테스트
- I층: 사전 검증. 실측과 같은 입력·가상 예보·임시 저장소에 '올바른 이해'를 가상 LLM으로 넣어,
       각 시나리오의 기대 결과를 실측 전에 고정한다. (기대값을 실측 결과를 보고 바꾸지 않기 위함)
- L층: 실제 측정. --live-ollama(실제 qwen2.5:3b + 고정 가상 예보), --live-real(실제 Open-Meteo·실제 RAG·실제 장치)

판정 원칙
- 지표 11개를 PASS / FAIL / N/A / NOT_RUN / ENV_ERROR로 따로 판정한다. N/A는 성공으로 세지 않는다.
- L층 unittest의 통과는 'S1 안전 불변식(승인 없는 저장·운영 저장소 접근·그래프 예외 없음)'을 지켰다는 뜻일 뿐,
  모델 판단이나 종합 흐름이 성공했다는 뜻이 아니다. 모델 판단 결과는 로그의 표와 요약으로 보고한다.
- 반복 횟수·시나리오·기대값은 이 파일의 상수로 고정한다. CLI로 바꿀 수 없다.
- 모든 저장소는 임시 폴더에 있다. 운영 저장소(.safemate/schedule_store.json)는 읽거나 쓰지 않는다.
"""
import contextlib
import copy
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest import mock

import device_utils as du
import outing_decision as od
import safemate_graph as sg
import safety_rag as sr
import schedule_agent as sa
import schedule_service as sv
import schedule_store as ss
import test_outing_decision as tod
import test_safemate_graph as tsg
import test_schedule as ts
from a_outing import KST


# %% 5-6 공통 상수
# 핵심 요약: 지표 이름·상태 값·반복 횟수·시간 상한 근거·로그 위치를 한곳에 고정한다.
# 시간 상한은 기존 실측(5-4 3차: 승인 회차 LLM 142초, 한 턴 최대 31초)을 근거로 3배 여유를 둔다.
ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / ".safemate" / "test_logs"
LOCK_NAME = "5-6_live.lock"
FIXED_NOW = ts.FIXED_NOW
OPTIONS = SimpleNamespace(live_ollama=False, live_real=False, device=None, rerun_reason=None)

PASS, FAIL, NA, NOT_RUN, ENV_ERROR = "PASS", "FAIL", "N/A", "NOT_RUN", "ENV_ERROR"
REVIEW = "REVIEW"       # 자동으로 확정할 수 없어 사람이 확인해야 함 (예: RAG 본문 관련성). PASS로 세지 않는다
STATUSES = (PASS, FAIL, NA, NOT_RUN, ENV_ERROR, REVIEW)
APPROVED_MODEL = "qwen2.5:3b"   # 승인된 A기능 모델. 실측 설정이 다르면 측정하지 않는다 (모델을 바꾸지 않음)
FAKE_DEVICE = tod.FAKE_INDEX.device["requested"]     # 가상 색인의 장치 이름표 (실제 연산 없음, 실제 장치 선택과 무관)
SUCCESS_WORDS = ("저장했어요", "등록했어요", "기록했어요", "취소했어요", "바꿨어요")
INDICATORS = ("graph_ok", "intent_ok", "judgment_ok", "proposal_ok", "decision_ok", "saved_ok", "requery_ok",
              "persist_ok", "safe_fallback", "external_ok", "device_ok")
INDICATOR_NAMES = {"graph_ok": "그래프 실행", "intent_ok": "LLM 의도 분류", "judgment_ok": "LLM 상황 판단",
                   "proposal_ok": "제안·안내 결과(기대 카드 또는 기대 안내)", "decision_ok": "승인·거절 반영",
                   "saved_ok": "실제 저장", "requery_ok": "재조회", "persist_ok": "재실행 후 유지",
                   "safe_fallback": "안전한 대체 응답", "external_ok": "외부 API·RAG", "device_ok": "실제 장치"}
CAUSES = ("프로그램 결함", "모델 판단 한계", "환경 오류", "원인 미확정")
REPEATS = 3
WEATHER_SOURCE = "Open-Meteo (CC BY 4.0) · https://open-meteo.com/"     # 1단계 예보 응답의 출처 표기 (운영 read_weather 계약과 같음)
MAX_OBSERVED_RUN_SECONDS = 142.0      # 5-4 3차 실측: 대표 흐름 승인 회차의 LLM 시간 최댓값
MAX_OBSERVED_TURN_SECONDS = 31.0      # 5-4 3차 실측: 3단계 + 일정 이해 턴의 최댓값
TIME_CAP_FACTOR = 3
ENV_ERROR_STREAK_LIMIT = 2
SCHEDULE_ROUTES: set[str] = set(sg.SCHEDULE_ROUTES)
SID = ts.SID
RAIN = ts.RAIN
UNUSED_STORE_NAME = "unused_operational_store.json"


def iso(day: int, hour: int, minute: int = 0) -> str:
    """2026-11-{day} {hour}:{minute} KST의 저장 형식 문자열"""
    return ss.iso(ts.at(day, hour, minute))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# %% 네트워크 차단 장치 (기본 M·I층 전용)
# 핵심 요약: 기본 실행에서 실제 모델·외부 API 호출이나 모델 다운로드가 일어나면 바로 실패하게 한다.
# socket 연결을 막고 Hugging Face 오프라인 환경변수를 테스트 범위에서만 켠다. (끝나면 원래대로 돌아간다)
class NetworkBlocked(OSError):
    """기본 테스트에서 네트워크 연결을 시도했다는 표시"""


@contextlib.contextmanager
def offline_guard():
    """socket.connect·create_connection을 막고 HF 오프라인 모드를 켠다. (with 범위 안에서만)"""
    def blocked(*args, **kwargs):
        raise NetworkBlocked("5-6 기본 테스트: 네트워크 연결이 차단되어 있습니다.")
    with mock.patch.object(socket.socket, "connect", blocked), mock.patch.object(socket, "create_connection", blocked), \
            mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
        yield


class OfflineCase(unittest.TestCase):
    """M·I층 기본 클래스: 테스트마다 네트워크 차단 장치를 켠다."""

    def setUp(self):
        stack = contextlib.ExitStack()
        stack.enter_context(offline_guard())
        self.addCleanup(stack.close)


# %% 보호 대상 상태 기록
# 핵심 요약: 운영 저장소·기존 실측 로그·공개 측정 자료의 상태(SHA-256)를 측정 전후로 비교하기 위해 읽는다. (쓰지 않음)
def operational_state() -> dict[str, str | None]:
    """운영 저장소와 표시 파일의 상태 (없으면 None)"""
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name: ts.file_state(p) for p in ts.OPERATIONAL_FILES}


def protected_records(log_dir: Path = LOG_DIR) -> dict[str, str | None]:
    """5-4 원본 로그 4개와 docs/test_results 파일의 SHA-256 (측정이 이 파일들을 바꾸지 않았는지 확인용)"""
    paths = sorted(log_dir.glob("5-4_live_flow*.txt")) + sorted((ROOT / "docs" / "test_results").glob("*"))
    return {p.name: ts.file_state(p) for p in paths if p.is_file()}


def safe_env(folder: Path, *, live: bool = False, **extra: str) -> dict[str, str]:
    """테스트 프로세스용 환경변수. 운영 저장소 경로 변수는 임시 폴더의 미사용 경로로 둔다.
    (실수로 기본 저장소를 쓰는 코드가 있으면 운영 폴더가 아닌 임시 폴더에 생겨서 바로 드러난다)
    - 가상 측정(live=False): 사용자 PC의 OLLAMA_*가 가상 결과를 바꾸지 않도록 지운다(기존 규칙)
    - 실측(live=True): 사용자가 정한 OLLAMA_* 설정을 그대로 둔다. 준비 확인·실제 호출·별도 프로세스가 같은 서버·모델을 쓴다."""
    env = dict(os.environ) if live else tod.clean_env()
    env.pop("SAFEMATE_EMBED_DEVICE", None)       # 장치는 인자로만 넘긴다 (환경변수로 조용히 바뀌지 않게)
    env[ss.STORE_PATH_ENV] = str(folder / UNUSED_STORE_NAME)
    env.update(extra)
    return env


def ollama_key() -> dict[str, Any]:
    """지금 환경의 Ollama 서버 주소와 모델 이름 (준비 확인과 실제 호출이 같은 설정인지 비교용)"""
    settings = od.ollama_settings()
    server = settings.get("server")
    geturl = getattr(server, "geturl", None)
    return {"server": geturl() if callable(geturl) else str(server), "model": settings.get("model")}


def approved_model_problem(key: dict[str, Any]) -> str | None:
    """승인된 모델(qwen2.5:3b)이 아니면 측정하지 않는 이유 문장"""
    if key.get("model") != APPROVED_MODEL:
        return f"승인된 모델은 {APPROVED_MODEL}인데 현재 설정은 {key.get('model')}입니다. 설정을 바꾸지 않고 측정하지 않습니다."
    return None


def git_info() -> dict[str, Any]:
    """측정 대상 커밋과 작업 트리 변경 여부 (기록용, git이 없으면 None)"""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout
        return {"commit": head or None, "dirty_paths": [line[3:] for line in dirty.splitlines()]}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty_paths": None}


def package_version(name: str) -> str | None:
    """패키지 버전 (import 없이 메타데이터만 읽음)"""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sanitize(text: str, extra: tuple[str, ...] = ()) -> str:
    """공개용 사본을 만들 때 프로젝트 절대 경로·사용자 폴더·사용자 이름·지정한 문자열(서버 주소 등)을 가린다."""
    pairs = [(str(ROOT), "<프로젝트 폴더>"), (str(ROOT).replace("\\", "/"), "<프로젝트 폴더>"),
             (str(Path.home()), "<사용자 폴더>"), (str(Path.home()).replace("\\", "/"), "<사용자 폴더>")]
    user = os.environ.get("USERNAME") or os.environ.get("USER")
    for value, label in pairs:
        text = text.replace(value, label)
    for value in extra:
        if value:
            text = text.replace(value, "<가림>")
    if user and len(user) >= 3:
        text = text.replace(user, "<사용자>")
    return text


# %% LLM·도구 호출 기록기와 재현기
# 핵심 요약: 실제(또는 가상) LLM의 원문 출력·단계·시간을 순서대로 남기고(RecordingLLM, 기존 CountingLLM 확장),
# 날씨·RAG 호출의 인자·응답·시간을 남긴다(TimedCall). 실패 회차는 이 기록으로 같은 순서를 다시 재생해 원인을 가른다.
class RecordingLLM(ts.CountingLLM):
    """기존 CountingLLM(단계·시간)에 원문 출력·모델 이름·오류를 더해 기록한다. 응답은 바꾸지 않는다."""

    def __call__(self, messages, schema, settings, *, stage, max_tokens=None, temperature=None):
        started = time.perf_counter()
        try:
            raw, stats = self.real(messages, schema, settings, stage=stage, max_tokens=max_tokens, temperature=temperature)
        except Exception as exc:
            self.calls.append({"stage": stage, "seconds": round(time.perf_counter() - started, 1), "raw": None,
                               "error": {"type": type(exc).__name__, "status": getattr(exc, "status", None),
                                         "code": getattr(exc, "code", None), "message": str(exc)[:200]}})
            raise
        # (D17 보완) 운영 코드가 이 호출에 보낸 스키마와 단계별 운영 검증 함수로 출력 규격을 바로 검사해 결과만 남긴다(스키마 원문은 저장 안 함).
        self.calls.append({"stage": stage, "seconds": round(time.perf_counter() - started, 1), "raw": raw,
                           "model": (stats or {}).get("model"), "format": format_problems(stage, raw, schema)})
        return raw, stats


class ReplayedError(RuntimeError):
    """기록에 남은 예외를 재현할 때 내는 예외 (원래 종류·메시지를 담는다)"""


class ReplayLLM:
    """기록된 LLM 호출을 같은 순서로 돌려준다. 단계가 다르거나 기록이 모자라면 divergence에 남기고 오류를 낸다.
    재현이 끝난 뒤 remaining()이 0이 아니면 기록보다 적게 호출된 것이다(이것도 어긋남)."""

    def __init__(self, calls: list[dict[str, Any]]):
        self.recorded = [dict(c) for c in calls]
        self.position = 0
        self.divergence: list[str] = []
        self.calls: list[dict[str, Any]] = []

    def remaining(self) -> int:
        return len(self.recorded) - self.position

    def __call__(self, messages, schema, settings, *, stage, max_tokens=None, temperature=None):
        # 1. 남은 기록이 없거나 단계가 다르면 재현이 어긋난 것이다. (모델 문제라고 단정할 근거가 아님)
        if self.position >= len(self.recorded):
            self.divergence.append(f"기록보다 호출이 많음({stage})")
            raise od.DecisionError("llm_unavailable", "replay_exhausted", "재현 기록이 끝났습니다.")
        item = self.recorded[self.position]
        self.position += 1
        self.calls.append({"stage": stage, "seconds": 0.0, "error": item.get("error")})
        if item.get("stage") != stage:
            self.divergence.append(f"{self.position}번째 호출 단계 {item.get('stage')} → {stage}")
        # 2. 기록된 오류는 같은 오류로, 기록된 원문은 그대로 돌려준다.
        error = item.get("error")
        if error:
            raise od.DecisionError(error.get("status") or "llm_unavailable", error.get("code") or "replay",
                                   error.get("message") or "")
        return item["raw"], {"stage": stage, "model": item.get("model") or settings["model"], "eval_count": 0,
                             "prompt_eval_count": 0, "total_duration_ms": 0}


def call_args(args: tuple, kwargs: dict) -> tuple[list[Any], dict[str, Any]]:
    """기록·비교용 인자: 문자열·숫자·None은 값 그대로, 그 밖의 객체(색인 등)는 타입 이름으로 남긴다."""
    def plain(value: Any) -> Any:
        return value if isinstance(value, (str, int, float, bool, type(None))) else f"<{type(value).__name__}>"
    return [plain(a) for a in args], {k: plain(v) for k, v in kwargs.items()}


class TimedCall:
    """날씨·RAG 함수를 감싸 인자(위치·키워드)·응답·예외·걸린 시간을 기록한다. 응답은 바꾸지 않고 예외는 그대로 다시 낸다."""

    def __init__(self, fn: Callable, kind: str):
        self.fn, self.kind, self.calls = fn, kind, []

    def __call__(self, *args, **kwargs):
        started = time.perf_counter()
        record_args, record_kwargs = call_args(args, kwargs)
        try:
            result = self.fn(*args, **kwargs)
        except Exception as exc:
            self.calls.append({"args": record_args, "kwargs": record_kwargs, "result": None,
                               "error": {"type": type(exc).__name__, "message": str(exc)[:300]},
                               "seconds": round(time.perf_counter() - started, 2)})
            raise
        self.calls.append({"args": record_args, "kwargs": record_kwargs,
                           "result": copy.deepcopy(result) if isinstance(result, dict) else None,
                           "seconds": round(time.perf_counter() - started, 2)})
        return result


class ReplayCalls:
    """기록된 날씨·RAG 응답(또는 예외)을 같은 순서로 돌려준다. 위치·키워드 인자가 다르면 divergence에 남긴다."""

    def __init__(self, calls: list[dict[str, Any]], kind: str):
        self.recorded, self.kind, self.position, self.divergence = list(calls), kind, 0, []

    def remaining(self) -> int:
        return len(self.recorded) - self.position

    def __call__(self, *args, **kwargs):
        if self.position >= len(self.recorded):
            self.divergence.append(f"{self.kind}: 기록보다 호출이 많음")
            return {"status": "api_error", "message": "재현 기록 없음", "weather": None} if self.kind == "weather" \
                else {"status": "search_error", "message": "재현 기록 없음", "count": 0, "results": []}
        item = self.recorded[self.position]
        self.position += 1
        now_args, now_kwargs = call_args(args, kwargs)
        if item.get("args", []) != now_args or item.get("kwargs", {}) != now_kwargs:
            self.divergence.append(f"{self.kind} {self.position}번째 인자 {item.get('args')}·{item.get('kwargs')} → "
                                   f"{now_args}·{now_kwargs}")
        if item.get("error"):
            raise ReplayedError(f"{item['error'].get('type')}: {item['error'].get('message')}")
        return copy.deepcopy(item["result"])


class ScheduleTap:
    """그래프에 주입하는 일정 상담 함수: 기본 run_schedule_request를 같은 인자로 그대로 부르고 결과만 기록한다.
    (A가 날씨로 넘겨받는 턴에도 일정 이해 결과를 잃지 않기 위함. 동작은 기본 함수와 같다)"""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def __call__(self, message, **kwargs):
        result = sa.run_schedule_request(message, **kwargs)
        llm = result.get("llm") or {}
        decision = result.get("decision") or {}
        raw_weather = result.get("weather")
        weather = raw_weather if isinstance(raw_weather, list) else []
        self.calls.append({"message": message, "understanding": copy.deepcopy(llm.get("understanding")),
                           "rules": list(llm.get("rules_applied") or []), "status": result.get("status"),
                           "route": result.get("route"), "kind": decision.get("kind"),
                           "missing": list(result.get("missing_fields") or []),
                           "weather": [{"at": c.get("at"), "status": c.get("status")} for c in weather if isinstance(c, dict)],
                           "embedding_device": copy.deepcopy(result.get("embedding_device")),
                           "card": result.get("proposed_action") is not None})
        return result


# %% 시나리오 정의 도우미
# 핵심 요약: 시나리오는 JSON으로 기록할 수 있는 값만 담는다(턴 문장·준비 작업 이름·가상 예보·기대값).
# a·s는 사전 검증용 '올바른 이해'(3단계·일정 이해)다. 실측에서는 쓰지 않고 실제 모델 출력과 비교하는 기준으로만 쓴다.
def scenario(sid: str, suite: str, turns: list[str], *, basis: str, title: str = "", setup: str | None = None,
             forecast: dict[str, Any] | None = None, a: dict[str, dict] | None = None, s: dict[str, dict] | None = None,
             choices: list[dict] | None = None, intents: dict[int, list[str]] | None = None,
             understanding: dict[int, dict] | None = None, checks: list[tuple[int, dict]] | None = None,
             repeats: int = REPEATS, note: str | None = None) -> dict[str, Any]:
    """시나리오 하나 (turns: 사용자 발화 순서, checks: [(턴 번호, 기대)], -1은 마지막 턴)
    note: 판단이 어긋났을 때 원인 후보에 덧붙일 사정 (예: AG41 문장의 '분류 명세 모호'). 원인을 확정하지 않는다."""
    return {"id": sid, "suite": suite, "title": title or sid, "basis": basis, "turns": list(turns), "setup": setup,
            "note": note,
            "forecast": forecast or {}, "a": a or {}, "s": s or {}, "choices": choices or [],
            "intents": {int(k): list(v) for k, v in (intents or {}).items()},
            "understanding": {int(k): v for k, v in (understanding or {}).items()},
            "checks": [[i, c] for i, c in (checks or [])], "repeats": repeats}


def A(intent: str, **values: Any) -> dict[str, Any]:
    """3단계 이해 단계의 올바른 출력(사전 검증용)"""
    return dict(intent=intent, **values)


def card(kind: str, *, schedule_id: str | None = None, start: str | None = None, city: list | None = None,
         title: str | None = None) -> dict[str, Any]:
    """기대 승인 카드 (city는 허용 값 목록, None 포함 가능)"""
    return {"type": kind, "schedule_id": schedule_id, "start": start, "city": city, "title": title}


SCHED: list[str] = sorted(SCHEDULE_ROUTES)


# %% 준비 작업 (기존 I층 테스트와 같은 일정 추가)
# 핵심 요약: 기존 테스트(AG35·AG37·AG42~44·AG47·GR17)가 쓰는 추가 일정을 같은 방식으로 만든다. 임시 저장소에만 쓴다.
def setup_walk(store: ss.ScheduleStore, now: datetime) -> None:
    """지역이 저장되지 않은 실외 일정 '공원 걷기' (내일 10:00~11:00, AG42~44·AG47과 같음)"""
    made = sv.propose_create(store.load(), title="공원 걷기", start=ts.at(26, 10), minutes=60, now=now, city=None,
                             category="outing")
    sv.apply_schedule_action(ts.approved(ts.got(made, "action"), now), store)


def setup_reading(day: int, hour: int) -> Callable[[ss.ScheduleStore, datetime], None]:
    """'독서 모임' 일정(60분, 실내·지역 없음)을 추가하는 준비 작업"""
    def run(store: ss.ScheduleStore, now: datetime) -> None:
        store.transact(uuid.uuid4().hex, "c" * 64, ts.add_schedule("독서 모임", ts.at(day, hour), 60, now))
    return run


SETUPS: dict[str, Callable[[ss.ScheduleStore, datetime], None]] = {
    "walk": setup_walk, "reading_26_10": setup_reading(26, 10), "reading_26_14": setup_reading(26, 14),
    "reading_27_14": setup_reading(27, 14)}


def make_weather(spec: dict[str, Any]) -> tod.FakeWeather:
    """가상 예보: rain에 적은 시각은 비(강수 2mm·80%), wind에 적은 시각은 강풍(12m/s), fail이면 모든 조회가 그 오류"""
    by_hour = {int(h): dict(RAIN) for h in spec.get("rain", [])}
    for h in spec.get("wind", []):
        by_hour.setdefault(int(h), {}).update(ts.WINDY)
    return tod.FakeWeather(by_hour=by_hour, fail=spec.get("fail"))


def contract_meta(city: str | None, day: str, clock: str) -> dict[str, Any]:
    """(D25 보완) 합성 예보 기록에 넣을 1단계 응답 계약 메타데이터: 지역·출발 시각·정시 구간(KST)·출처·종류 (FakeWeather와 같은 형식)"""
    at = datetime.fromisoformat(f"{day}T{clock}").replace(tzinfo=KST)
    start, end = od.departure_hour_window(at)
    return {"data_kind": "forecast", "source": WEATHER_SOURCE, "city": city, "outing_at": at.isoformat(timespec="minutes"),
            "hour_start": start.isoformat(timespec="minutes"), "hour_end": end.isoformat(timespec="minutes")}


# %% 4.2~4.7 시나리오 (실측 전에 고정. 기대값의 근거는 basis의 기존 테스트 또는 이 파일의 I층 사전 검증)
DELEGATE = ts.I_AG5_WeatherAndDelegation.DELEGATE
DELEGATE_U = ts.I_AG5_WeatherAndDelegation.DELEGATE_U
WALK_U = {"operation": "update", "target_text": "공원 걷기", "preferences": ["avoid_rain"]}
CITY_U = {"answers_previous_question": True, "city_text": "인천"}
AG41_MSG = "내일 오후 두 시에 비 안 오면 산책 등록해 줘"
AG41_U = {"operation": "create", "title": "산책", "date_text": "내일", "time_text": "오후 두 시",
          "preferences": ["avoid_rain"], "category_guess": "outing"}
EVENING_MSG = "내일 저녁 비 안 오는 시간에 산책 등록해 줘"
EVENING_U = {"operation": "create", "title": "산책", "date_text": "내일", "time_text": "저녁", "delegate_choice": False,
             "preferences": ["avoid_rain"], "category_guess": "outing"}
CONFLICT_MSG = "시장 비 안 오는 시간으로 오후 네 시로 바꿔 줘"
CONFLICT_U = {"operation": "update", "target_text": "시장", "new_time_text": "오후 네 시", "preferences": ["avoid_rain"]}
SEVEN = "시장 시간 저녁 일곱 시로 바꿔 줘"
SEVEN_U = {"operation": "update", "target_text": "시장", "new_time_text": "저녁 일곱 시"}
GWANGJU_MSG = "모레 오후 세 시에 광주에서 친구 만나기로 했어."
GWANGJU_U = {"operation": "create", "title": "친구 만나기", "date_text": "모레", "time_text": "오후 세 시",
             "city_text": "광주", "category_guess": "outing"}
GG_CITY = ["경기도 광주", "광주, 경기도"]       # 사용자가 '경기도 광주'라고 했을 때 인정하는 지역 표기
WALK_GWANGJU_MSG = "공원 걷기 광주 기준으로 비 안 오게 오후 다섯 시로 바꿔 줘"
AMPM_MSG = "내일 두 시 일정 취소해 줘"

LV2 = [
    scenario("LV2-delegate", "LV2", [DELEGATE], basis="AG22", title="선택 위임 (13·14시 비 → 15:00)",
             forecast={"rain": [13, 14]}, a={DELEGATE: A("schedule_change")}, s={DELEGATE: DELEGATE_U},
             choices=[{"choice": "C3", "reasons": ["less_rain", "no_conflict"]}], intents={0: SCHED},
             understanding={0: {"operation": "update", "delegate_choice": True, "preferences": ["avoid_rain"]}},
             checks=[(0, {"card": card("update_schedule", schedule_id=SID["시장"], start=iso(26, 15)),
                          "texts": ["안전을 보장한다는 뜻은 아니에요"]})]),
    scenario("LV2-delegate-unknown", "LV2", [DELEGATE], basis="AG24", title="선택 위임 + 예보 확인 불가 (network_error)",
             forecast={"fail": "network_error"}, a={DELEGATE: A("schedule_change")}, s={DELEGATE: DELEGATE_U},
             intents={0: SCHED},
             understanding={0: {"operation": "update", "delegate_choice": True, "preferences": ["avoid_rain"]}},
             checks=[(0, {"card": None, "texts": ["비 신호가 없다고 확인된 빈 시간을 찾지 못해 바꾸지 않았어요."]})]),
    scenario("LV2-period", "LV2", ["시장 저녁쯤 비 안 오는 시간으로 바꿔 줘"], basis="AG37", setup="reading_26_10",
             title="시간대 지정 (13·14·18시 비 → 19:00)", forecast={"rain": [13, 14, 18]},
             a={"시장 저녁쯤 비 안 오는 시간으로 바꿔 줘": A("schedule_change")},
             s={"시장 저녁쯤 비 안 오는 시간으로 바꿔 줘": {"operation": "update", "target_text": "시장", "new_time_text": "저녁쯤",
                                                 "preferences": ["avoid_rain"]}}, intents={0: SCHED},
             understanding={0: {"operation": "update", "preferences": ["avoid_rain"], "new_time_text": "~저녁"}},
             checks=[(0, {"card": card("update_schedule", schedule_id=SID["시장"], start=iso(26, 19))})]),
    scenario("LV2-direct-rain", "LV2", [AG41_MSG, "인천"], basis="AG41", title="직접 시각 등록 + 지역 질문 (15시 비 → 제안 없음)",
             note="분류 명세 모호(비 안 오면 ~ 등록: request+avoid_rain 기대, conditional 표기는 프롬프트에 규칙 없음)",
             forecast={"rain": [15]}, a={AG41_MSG: A("schedule_create")},
             s={AG41_MSG: AG41_U, "인천": dict(CITY_U, operation="create")}, intents={0: SCHED},
             understanding={0: {"operation": "create", "request_mode": "request", "preferences": ["avoid_rain"]},
                            1: {"city_text": "~인천"}},
             checks=[(0, {"card": None, "missing": ["city"], "texts": ["대체 후보로 골라 예보를 확인해요"],
                          "not_texts": ["오후 3시로 제안해요"]}),
                     (1, {"card": None, "texts": ["비 예보가 있어 제안하지 않았어요"]})]),
    scenario("LV2-direct-clear", "LV2", [AG41_MSG, "인천"], basis="AG41", title="직접 시각 등록 + 지역 질문 (맑음 → 15:00)",
             note="분류 명세 모호(비 안 오면 ~ 등록: request+avoid_rain 기대, conditional 표기는 프롬프트에 규칙 없음)",
             forecast={}, a={AG41_MSG: A("schedule_create")},
             s={AG41_MSG: AG41_U, "인천": dict(CITY_U, operation="create")}, intents={0: SCHED},
             understanding={0: {"operation": "create", "request_mode": "request", "preferences": ["avoid_rain"]},
                            1: {"city_text": "~인천"}},
             checks=[(0, {"card": None, "missing": ["city"]}),
                     (1, {"card": card("create_schedule", start=iso(26, 15), city=["인천"]),
                          "texts": ["프로젝트 기준 위험 신호 없음(안전 보장 아님)"]})]),
    scenario("LV2-relative", "LV2", ["공원 걷기 비 안 오게 한 시간 늦춰 줘", "인천"], basis="AG43 첫 턴 + I층 사전 검증",
             setup="walk", title="상대 시각 + 지역 질문 (17시 비, 11:00 확인)", forecast={"rain": [17]},
             a={"공원 걷기 비 안 오게 한 시간 늦춰 줘": A("schedule_change")},
             s={"공원 걷기 비 안 오게 한 시간 늦춰 줘": dict(WALK_U, shift_text="한 시간 늦춰", shift_base="schedule"),
                "인천": dict(CITY_U, operation="update")}, intents={0: SCHED},
             understanding={0: {"operation": "update", "preferences": ["avoid_rain"], "shift_text": "*"},
                            1: {"city_text": "~인천"}},
             checks=[(0, {"card": None, "missing": ["city"]}),
                     (1, {"card": card("update_schedule", start=iso(26, 11), city=[None])})]),
    scenario("LV2-conflict-rain", "LV2", [CONFLICT_MSG], basis="AG43", title="충돌 대체 시각 (17시 비 → 제안 없음)",
             forecast={"rain": [17]}, a={CONFLICT_MSG: A("schedule_change")}, s={CONFLICT_MSG: CONFLICT_U},
             intents={0: SCHED}, understanding={0: {"operation": "update", "preferences": ["avoid_rain"]}},
             checks=[(0, {"card": None, "texts": ["그래서 비어 있는 오후 5시을(를) 대체 후보로 골라 예보를 확인해요",
                                                  "오후 5시에는 비 예보가 있어 제안하지 않았어요"],
                          "not_texts": ["오후 5시로 제안해요"]})]),
    scenario("LV2-conflict-clear", "LV2", [CONFLICT_MSG], basis="I층 사전 검증", title="충돌 대체 시각 (맑음 → 17:00)",
             forecast={}, a={CONFLICT_MSG: A("schedule_change")}, s={CONFLICT_MSG: CONFLICT_U}, intents={0: SCHED},
             understanding={0: {"operation": "update", "preferences": ["avoid_rain"]}},
             checks=[(0, {"card": card("update_schedule", schedule_id=SID["시장"], start=iso(26, 17))})]),
]

LV3 = [
    scenario("LV3-create-evening", "LV3", [EVENING_MSG, "인천"], basis="AG45", title="저녁·비 조건 등록 → 지역 질문 → 인천",
             forecast={"rain": [18]}, a={EVENING_MSG: A("schedule_create")},
             s={EVENING_MSG: EVENING_U, "인천": dict(CITY_U, operation="create")}, intents={0: SCHED},
             understanding={0: {"operation": "create", "preferences": ["avoid_rain"], "time_text": "~저녁"},
                            1: {"city_text": "~인천"}},
             checks=[(0, {"card": None, "missing": ["city"]}),
                     (1, {"card": card("create_schedule", start=iso(26, 19), city=["인천"]), "no_city_question": True})]),
    scenario("LV3-date-correction", "LV3", [EVENING_MSG, "인천, 모레로"], basis="AG46", title="지역 답변 + 날짜만 정정 (시간대 유지)",
             forecast={"rain": [18]}, a={EVENING_MSG: A("schedule_create")},
             s={EVENING_MSG: EVENING_U, "인천, 모레로": dict(CITY_U, operation="create", date_text="모레")}, intents={0: SCHED},
             understanding={0: {"operation": "create", "preferences": ["avoid_rain"]},
                            1: {"city_text": "~인천", "date_text": "~모레"}},
             checks=[(1, {"card": card("create_schedule", start=iso(27, 19), city=["인천"]), "no_city_question": True})]),
    scenario("LV3-time-correction", "LV3", [EVENING_MSG, "인천, 아침 아홉 시로"], basis="AG46", title="지역 답변 + 시각 정정 (새 시각 우선)",
             forecast={"rain": [18]}, a={EVENING_MSG: A("schedule_create")},
             s={EVENING_MSG: EVENING_U, "인천, 아침 아홉 시로": dict(CITY_U, operation="create", time_text="아침 아홉 시")},
             intents={0: SCHED},
             understanding={0: {"operation": "create", "preferences": ["avoid_rain"]},
                            1: {"city_text": "~인천", "time_text": "~아홉"}},
             checks=[(1, {"card": card("create_schedule", start=iso(26, 9), city=["인천"]), "no_city_question": True})]),
    scenario("LV3-update-evening", "LV3", ["공원 걷기 저녁 비 안 오는 시간으로 바꿔 줘", "인천"], basis="AG47", setup="walk",
             title="지역 없는 실외 일정 변경 → 지역 질문 → 인천 (일정 지역 불변)", forecast={"rain": [18]},
             a={"공원 걷기 저녁 비 안 오는 시간으로 바꿔 줘": A("schedule_change")},
             s={"공원 걷기 저녁 비 안 오는 시간으로 바꿔 줘": {"operation": "update", "target_text": "공원 걷기",
                                                  "new_time_text": "저녁", "preferences": ["avoid_rain"]},
                "인천": dict(CITY_U, operation="update")}, intents={0: SCHED},
             understanding={0: {"operation": "update", "preferences": ["avoid_rain"]}, 1: {"city_text": "~인천"}},
             checks=[(0, {"card": None, "missing": ["city"]}),
                     (1, {"card": card("update_schedule", start=iso(26, 19), city=[None]), "no_city_question": True})]),
]

LV4 = [
    scenario("LV4-V1", "LV4", ["내일 일정을 바꿔 줘.", "첫 번째"], basis="AG11", title="V1 대상이 여러 개 → 번호 질문 → 첫 번째",
             a={"내일 일정을 바꿔 줘.": A("schedule_change")},
             s={"내일 일정을 바꿔 줘.": {"operation": "update", "date_text": "내일"}, "첫 번째": {"operation": "other"}},
             intents={0: SCHED}, understanding={0: {"operation": "update", "request_mode": "request"}},
             checks=[(0, {"card": None, "texts": ["1) 내일(11월 26일 목요일) 오후 2시~오후 3시 시장 장보기"]}),
                     (1, {"card": None, "texts": ["'시장 장보기' 일정은 지금"]})]),
    scenario("LV4-V2", "LV4", ["시장 시간 바꿔 줘", "내일 일정 바꾸지 마."], basis="AG12", title="V2 부정 (진행 중인 초안 폐기)",
             a={"시장 시간 바꿔 줘": A("schedule_change"), "내일 일정 바꾸지 마.": A("schedule_change")},
             s={"시장 시간 바꿔 줘": {"operation": "update", "target_text": "시장"},
                "내일 일정 바꾸지 마.": {"operation": "update", "request_mode": "negated", "date_text": "내일"}},
             intents={0: SCHED}, understanding={1: {"request_mode": "negated"}},
             checks=[(1, {"card": None, "texts": ["알겠어요."], "draft_none": True})]),
    scenario("LV4-V3", "LV4", ["시간을 바꿀 필요는 없어."], basis="AG12", title="V3 부정 (제안 없음)",
             a={"시간을 바꿀 필요는 없어.": A("schedule_change")},
             s={"시간을 바꿀 필요는 없어.": {"operation": "update", "request_mode": "negated"}},
             understanding={0: {"request_mode": "negated"}},
             checks=[(0, {"card": None, "texts": ["알겠어요."]})]),
    scenario("LV4-V4-rain", "LV4", ["비가 오면 시장 시간을 바꿀까?"], basis="AG13", title="V4 조건부 변경 (13·14시 비 → 15:00)",
             forecast={"rain": [13, 14]}, a={"비가 오면 시장 시간을 바꿀까?": A("schedule_change")},
             s={"비가 오면 시장 시간을 바꿀까?": {"operation": "update", "request_mode": "conditional", "target_text": "시장",
                                         "preferences": ["avoid_rain"], "check_weather": True}},
             choices=[{"choice": "C3", "reasons": ["less_rain", "no_conflict"]}], intents={0: SCHED},
             understanding={0: {"operation": "update", "request_mode": "conditional"}},
             checks=[(0, {"card": card("update_schedule", schedule_id=SID["시장"], start=iso(26, 15)),
                          "texts": ["비 예보가 있어요"]})]),
    scenario("LV4-V4-clear", "LV4", ["비가 오면 시장 시간을 바꿀까?"], basis="AG13", title="V4 조건부 변경 (맑음 → 그대로)",
             forecast={}, a={"비가 오면 시장 시간을 바꿀까?": A("schedule_change")},
             s={"비가 오면 시장 시간을 바꿀까?": {"operation": "update", "request_mode": "conditional", "target_text": "시장",
                                         "preferences": ["avoid_rain"], "check_weather": True}}, intents={0: SCHED},
             understanding={0: {"operation": "update", "request_mode": "conditional"}},
             checks=[(0, {"card": None, "kind": "schedule_condition_not_met"})]),
    scenario("LV4-V5", "LV4", [SEVEN, "아니, 원래 시간에 갈래."], basis="AG14", title="V5 제안 뒤 '원래 시간에 갈래' → 초안 폐기",
             a={SEVEN: A("schedule_change"), "아니, 원래 시간에 갈래.": A("schedule_change")},
             s={SEVEN: SEVEN_U, "아니, 원래 시간에 갈래.": {"operation": "update", "request_mode": "negated"}},
             intents={0: SCHED}, understanding={0: {"operation": "update"}, 1: {"request_mode": "negated"}},
             checks=[(0, {"card": card("update_schedule", schedule_id=SID["시장"], start=iso(26, 19))}),
                     (1, {"card": None, "draft_none": True,
                          "texts": ["'시장 장보기' 일정은 원래대로 내일(11월 26일 목요일) 오후 2시에 그대로 둘게요."]})]),
    scenario("LV4-V6", "LV4", ["내일 오후 일정 바꿔 줘", "첫 번째", "아까 말한 일정 말고 다른 일정이야."], basis="AG15",
             title="V6 '다른 일정이야' → 직전 대상 제외 (남은 복지관은 변경 제한)",
             a={"내일 오후 일정 바꿔 줘": A("schedule_change")},
             s={"내일 오후 일정 바꿔 줘": {"operation": "update", "date_text": "내일", "time_text": "오후"},
                "첫 번째": {"operation": "update", "answers_previous_question": True},
                "아까 말한 일정 말고 다른 일정이야.": {"operation": "update", "request_mode": "correction", "corrects_field": "target",
                                           "exclude_previous_target": True}},
             intents={0: SCHED}, understanding={2: {"request_mode": "correction", "exclude_previous_target": True}},
             checks=[(2, {"card": None, "texts": ["'복지관 방문'은(는) 변경 제한 일정"]})]),
    scenario("LV4-V7", "LV4", [SEVEN, "저녁 일곱 시 아니고 여덟 시야."], basis="AG16", title="V7 시각 정정 → 20:00",
             a={SEVEN: A("schedule_change"), "저녁 일곱 시 아니고 여덟 시야.": A("schedule_change")},
             s={SEVEN: SEVEN_U, "저녁 일곱 시 아니고 여덟 시야.": {"operation": "update", "request_mode": "correction",
                                                        "corrects_field": "time", "new_time_text": "여덟 시"}},
             intents={0: SCHED}, understanding={1: {"request_mode": "correction", "corrects_field": "time"}},
             checks=[(1, {"card": card("update_schedule", schedule_id=SID["시장"], start=iso(26, 20)),
                          "texts": ["앞에서 말씀하신 시간대를 이어 오후 8시로 계산했어요."]})]),
    scenario("LV4-V8", "LV4", ["시장 가는 시간 좀 바꿔 줘", "오늘 인천 날씨 어때?", "우유가 떨어졌어"], basis="AG17·GR10",
             title="V8 상담 중 날씨·생활 지원 전환 → A·B 미연결 안내, 초안 보존",
             a={"시장 가는 시간 좀 바꿔 줘": A("schedule_change"), "우유가 떨어졌어": A("living_support"),
                "오늘 인천 날씨 어때?": A("weather_question", needs_weather=True, city_text="인천", date_text="오늘")},
             s={"시장 가는 시간 좀 바꿔 줘": ts.FLOW_S["시장 가는 시간 좀 바꿔 줘"], "오늘 인천 날씨 어때?": {"operation": "other"},
                "우유가 떨어졌어": {"operation": "other"}},
             intents={0: SCHED, 1: ["weather_question"], 2: ["living_support"]},
             understanding={0: {"operation": "update"}, 1: {"operation": "other"}, 2: {"operation": "other"}},
             checks=[(1, {"card": None, "feature": "A", "draft_kept": True}),
                     (2, {"card": None, "feature": "B", "texts": [sg.B_UNAVAILABLE], "draft_kept": True})]),
]

LV5 = [
    scenario("LV5-create", "LV5", ["비 오면 내일 아침 여덟 시에 산책 등록해 줘"], basis="AG34", title="조건부 등록 (미지원)",
             forecast={"rain": [14]}, a={"비 오면 내일 아침 여덟 시에 산책 등록해 줘": A("schedule_create")},
             s={"비 오면 내일 아침 여덟 시에 산책 등록해 줘": {"operation": "create", "request_mode": "conditional", "title": "산책",
                                                 "date_text": "내일", "time_text": "아침 여덟 시"}}, intents={0: SCHED},
             understanding={0: {"operation": "create", "request_mode": "conditional"}},
             checks=[(0, {"card": None, "texts": ["기능은 아직 없어서 일정은 바꾸지 않았어요"], "weather_calls": 0})]),
    scenario("LV5-cancel", "LV5", ["비 오면 내일 장보기 취소해 줘"], basis="AG34", title="조건부 취소 (미지원)",
             forecast={"rain": [14]}, a={"비 오면 내일 장보기 취소해 줘": A("schedule_cancel")},
             s={"비 오면 내일 장보기 취소해 줘": {"operation": "cancel", "target_text": "장보기", "date_text": "내일",
                                          "preferences": ["avoid_rain"]}}, intents={0: SCHED},
             understanding={0: {"operation": "cancel", "_any": [{"request_mode": "conditional"}, {"preferences": ["avoid_rain"]}]}},
             checks=[(0, {"card": None, "texts": ["기능은 아직 없어서 일정은 바꾸지 않았어요"], "weather_calls": 0})]),
    scenario("LV5-complete", "LV5", ["비 오면 산책 완료로 해 줘"], basis="AG34", title="조건부 완료 (미지원)",
             forecast={"rain": [14]}, a={"비 오면 산책 완료로 해 줘": A("schedule_complete")},
             s={"비 오면 산책 완료로 해 줘": {"operation": "complete", "request_mode": "conditional", "target_text": "산책"}},
             intents={0: SCHED},
             understanding={0: {"operation": "complete", "_any": [{"request_mode": "conditional"}, {"preferences": ["avoid_rain"]}]}},
             checks=[(0, {"card": None, "texts": ["기능은 아직 없어서 일정은 바꾸지 않았어요"], "weather_calls": 0})]),
]

AMPM_U = {"operation": "cancel", "date_text": "내일", "time_text": "두 시"}
LV6 = [
    scenario("LV6-ampm-pm", "LV6", [AMPM_MSG, "오후 두 시"], basis="AG35 논리 + I층 사전 검증", title="모호한 시각 → 오후 두 시 (독립 초기 상태)",
             a={AMPM_MSG: A("schedule_cancel")},
             s={AMPM_MSG: AMPM_U, "오후 두 시": {"operation": "cancel", "answers_previous_question": True, "time_text": "오후 두 시"}},
             intents={0: SCHED}, understanding={0: {"operation": "cancel"}, 1: {"time_text": "~오후"}},
             checks=[(0, {"card": None, "missing": ["target_time"], "texts": ["오전인지 오후인지"]}),
                     (1, {"card": card("cancel_schedule", schedule_id=SID["시장"])})]),
    scenario("LV6-ampm-am", "LV6", [AMPM_MSG, "오전 두 시"], basis="AG35 논리 + I층 사전 검증", title="모호한 시각 → 오전 두 시 (독립 초기 상태)",
             a={AMPM_MSG: A("schedule_cancel")},
             s={AMPM_MSG: AMPM_U, "오전 두 시": {"operation": "cancel", "answers_previous_question": True, "time_text": "오전 두 시"}},
             intents={0: SCHED}, understanding={0: {"operation": "cancel"}, 1: {"time_text": "~오전"}},
             checks=[(0, {"card": None, "missing": ["target_time"]}), (1, {"card": None})]),
    scenario("LV6-ag35-pm", "LV6", ["모레 두 시 일정 취소해 줘", "오후 두 시"], basis="AG35", setup="reading_27_14",
             title="AG35 원문 → 오후 두 시 → 독서 모임 취소 카드",
             a={"모레 두 시 일정 취소해 줘": A("schedule_cancel")},
             s={"모레 두 시 일정 취소해 줘": {"operation": "cancel", "date_text": "모레", "time_text": "두 시"},
                "오후 두 시": {"operation": "cancel", "answers_previous_question": True, "time_text": "오후 두 시"}},
             intents={0: SCHED}, understanding={0: {"operation": "cancel"}, 1: {"time_text": "~오후"}},
             checks=[(0, {"card": None, "missing": ["target_time"]}), (1, {"card": card("cancel_schedule", schedule_id="schedule_005")})]),
    scenario("LV6-city-gg", "LV6", [GWANGJU_MSG, "경기도 광주"], basis="AG38 + I층 사전 검증", title="'광주' → 지역 질문 → '경기도 광주'",
             a={GWANGJU_MSG: A("schedule_create")},
             s={GWANGJU_MSG: GWANGJU_U, "경기도 광주": {"operation": "create", "answers_previous_question": True,
                                                    "city_text": "경기도 광주"}},
             intents={0: SCHED}, understanding={0: {"operation": "create", "city_text": "광주"}, 1: {"city_text": "~경기도"}},
             checks=[(0, {"card": None, "missing": ["city"], "texts": ["'광주광역시' 또는 '광주, 경기도'"]}),
                     (1, {"card": card("create_schedule", start=iso(27, 15), city=GG_CITY, title="친구")})]),
    scenario("LV6-city-metro", "LV6", [GWANGJU_MSG, "광주광역시"], basis="AG38 + I층 사전 검증", title="'광주' → 지역 질문 → '광주광역시'",
             a={GWANGJU_MSG: A("schedule_create")},
             s={GWANGJU_MSG: GWANGJU_U, "광주광역시": {"operation": "create", "answers_previous_question": True,
                                                   "city_text": "광주광역시"}},
             intents={0: SCHED}, understanding={0: {"operation": "create", "city_text": "광주"}, 1: {"city_text": "~광역시"}},
             checks=[(0, {"card": None, "missing": ["city"]}),
                     (1, {"card": card("create_schedule", start=iso(27, 15), city=["광주광역시"], title="친구")})]),
    scenario("LV6-city-said-gg", "LV6", ["모레 오후 세 시에 경기도 광주에서 친구 만나기로 했어"], basis="AG38 + I층 사전 검증",
             title="처음부터 '경기도 광주' → 광주광역시로 바꾸지 않음",
             a={"모레 오후 세 시에 경기도 광주에서 친구 만나기로 했어": A("schedule_create")},
             s={"모레 오후 세 시에 경기도 광주에서 친구 만나기로 했어": dict(GWANGJU_U, city_text="경기도 광주")},
             intents={0: SCHED}, understanding={0: {"operation": "create", "city_text": "~경기도"}},
             checks=[(0, {"card": card("create_schedule", start=iso(27, 15), city=GG_CITY)})]),
    scenario("LV6-city-said-metro", "LV6", ["모레 오후 세 시에 광주광역시에서 친구 만나기로 했어"], basis="AG38 + I층 사전 검증",
             title="처음부터 '광주광역시' → 그대로 유지",
             a={"모레 오후 세 시에 광주광역시에서 친구 만나기로 했어": A("schedule_create")},
             s={"모레 오후 세 시에 광주광역시에서 친구 만나기로 했어": dict(GWANGJU_U, city_text="광주광역시")},
             intents={0: SCHED}, understanding={0: {"operation": "create", "city_text": "광주광역시"}},
             checks=[(0, {"card": card("create_schedule", start=iso(27, 15), city=["광주광역시"])})]),
    scenario("LV6-update-city", "LV6", [WALK_GWANGJU_MSG, "경기도 광주"], basis="D1 회귀(AG51) + I층 사전 검증", setup="walk",
             title="변경: '광주' 기준 비 조건 → 지역 질문 → '경기도 광주' (대상·새 시각 유지, 일정 지역 불변)",
             a={WALK_GWANGJU_MSG: A("schedule_change")},
             s={WALK_GWANGJU_MSG: {"operation": "update", "target_text": "공원 걷기", "new_time_text": "오후 다섯 시",
                                   "city_text": "광주", "preferences": ["avoid_rain"]},
                "경기도 광주": {"operation": "update", "answers_previous_question": True, "city_text": "경기도 광주"}},
             intents={0: SCHED}, understanding={0: {"operation": "update", "city_text": "광주"}, 1: {"city_text": "~경기도"}},
             checks=[(0, {"card": None, "missing": ["city"]}),
                     (1, {"card": card("update_schedule", schedule_id="schedule_005", start=iso(26, 17), city=[None])})]),
]

OUT_UPDATE = "내일 오후 2시에 인천 시장 가는데 비 오면 어떡해? 오후 다섯 시로 늦춰도 괜찮아."
OUT_REFUSED = "내일 오후 2시에 인천 시장 가는데 비 오면 어떡해? 시간은 바꿀 수 있어."
OUT_CREATE = "모레 오전 10시에 인천 공원 산책 가려는데 비 오면 어떡해? 시간은 바꿀 수 있어."
OUT_AMBIG = "내일 오후 2시에 인천에서 친구 만나는데 비 오면 어떡해? 오후 다섯 시로 늦춰도 괜찮아."
OUT_A: dict[str, Any] = dict(intent="outing_consultation", needs_weather=True, city_text="인천", can_reschedule="yes")
LV7 = [
    scenario("LV7-update", "LV7", [OUT_UPDATE], basis="GR15 + I층 사전 검증(실제 3단계 A)", title="A 제안 → 저장된 시장 일정 변경 카드",
             forecast={"rain": [14]},
             a={OUT_UPDATE: A(**OUT_A, date_text="내일", time_text="오후 2시", purpose="시장", alternative_time_text="오후 다섯 시")},
             intents={0: ["outing_consultation"]},
             checks=[(0, {"card": card("update_schedule", schedule_id=SID["시장"], start=iso(26, 17)), "outing_link": "update",
                          "texts": ["저장된 '시장 장보기' 일정과 연결해"]})]),
    scenario("LV7-ambiguous", "LV7", [OUT_AMBIG, "1번"], basis="GR17 + I층 사전 검증(실제 3단계 A)", setup="reading_26_14",
             title="A 제안 → 겹치는 일정 2개 → 되묻기 → 1번",
             forecast={"rain": [14]},
             a={OUT_AMBIG: A(**OUT_A, date_text="내일", time_text="오후 2시", purpose="친구 만나기",
                             alternative_time_text="오후 다섯 시")},
             s={"1번": {"operation": "update", "answers_previous_question": True}},
             intents={0: ["outing_consultation"]},
             checks=[(0, {"card": None, "outing_link": "ambiguous", "texts": ["어느 것을 바꿀지 확인이 필요해요"]}),
                     (1, {"card": card("update_schedule", schedule_id=SID["시장"], start=iso(26, 17))})]),
    scenario("LV7-create", "LV7", [OUT_CREATE], basis="GR16 + I층 사전 검증(실제 3단계 A)", title="A 제안 → 저장되지 않은 외출 → 새 일정 등록 카드",
             forecast={"rain": [10]},
             a={OUT_CREATE: A(**OUT_A, date_text="모레", time_text="오전 10시", purpose="공원 산책")},
             intents={0: ["outing_consultation"]},
             checks=[(0, {"card": card("create_schedule", start=iso(27, 12)), "outing_link": "create",
                          "texts": ["새 일정으로 등록하는 것을 제안해요"]})]),
    scenario("LV7-refused", "LV7", [OUT_REFUSED], basis="GR18 + I층 사전 검증(실제 3단계 A)", title="A 제안 시각이 복지관과 겹침 → 카드 없이 이유 안내",
             forecast={"rain": [14]},
             a={OUT_REFUSED: A(**OUT_A, date_text="내일", time_text="오후 2시", purpose="시장")},
             intents={0: ["outing_consultation"]},
             checks=[(0, {"card": None, "outing_link": "refused",
                          "texts": ["[일정 확인] 오후 4시에는 다른 일정(오후 4시 복지관 방문)이 있어요."]})]),
]

GRAPH_SCENARIOS = LV2 + LV3 + LV4 + LV5 + LV6 + LV7
SUITES = {"LV2": LV2, "LV3": LV3, "LV4": LV4, "LV5": LV5, "LV6": LV6, "LV7": LV7}


# %% 4.9-B 일정 이해 측정 문장 21개 (실측 전 고정)
# 핵심 요약: 문장마다 '모델 판단으로 인정할 필수 칸'(matcher)을 정한다. s는 사전 검증용 올바른 출력이다.
# _any는 '둘 중 하나면 인정'이다(예: 조건부 취소는 conditional 또는 avoid_rain). 정정·문맥 문장은 4.4 흐름에서 잰다.
def item(n: int, text: str, expect: dict, s: dict, outcome: dict | None = None) -> dict[str, Any]:
    return {"id": f"LV9-{n:02d}", "text": text, "expect": expect, "s": s, "outcome": outcome or {}, "repeats": REPEATS}


LV9 = [
    item(1, "내일 시장 가는 시간 좀 바꿔 줘", {"operation": "update", "request_mode": "request"},
         {"operation": "update", "target_text": "시장 가는", "date_text": "내일"}),
    item(2, "내일 일정을 바꿔 줘.", {"operation": "update", "request_mode": "request"}, {"operation": "update", "date_text": "내일"}),
    item(3, "내일 일정 바꾸지 마.", {"request_mode": "negated"},
         {"operation": "update", "request_mode": "negated", "date_text": "내일"}, {"kind": "schedule_kept"}),
    item(4, "시간을 바꿀 필요는 없어.", {"request_mode": "negated"}, {"operation": "update", "request_mode": "negated"},
         {"kind": "schedule_kept"}),
    item(5, "비가 오면 시장 시간을 바꿀까?", {"operation": "update", "request_mode": "conditional"},
         {"operation": "update", "request_mode": "conditional", "target_text": "시장", "preferences": ["avoid_rain"],
          "check_weather": True}),
    item(6, "시간은 네가 알아서 정해 줘", {"operation": "update", "delegate_choice": True},
         {"operation": "update", "delegate_choice": True}),
    item(7, "늦잠 자서 시장 가는 거 한 시간 늦출게", {"operation": "update", "shift_text": "*", "shift_base": "schedule"},
         {"operation": "update", "target_text": "시장 가는", "shift_text": "한 시간 늦출게", "shift_base": "schedule",
          "reason_text": "늦잠 자서"}),
    item(8, "시장 비 안 오는 시간으로 알아서 바꿔 줘", {"operation": "update", "delegate_choice": True, "preferences": ["avoid_rain"]},
         {"operation": "update", "target_text": "시장", "delegate_choice": True, "preferences": ["avoid_rain"]}),
    item(9, "내가 내일 뭐 하기로 했지?", {"operation": "query", "query_scope": "date"},
         {"operation": "query", "query_scope": "date", "date_text": "내일"}),
    item(10, "시장 가는 거 몇 시였지?", {"operation": "query", "query_scope": "target"},
         {"operation": "query", "query_scope": "target", "target_text": "시장 가는"}),
    item(11, "다음 약속이 뭐야?", {"operation": "query", "query_scope": "next"}, {"operation": "query", "query_scope": "next"}),
    item(12, "오늘 남은 일정 있어?", {"operation": "query", "query_scope": "remaining"},
         {"operation": "query", "query_scope": "remaining", "date_text": "오늘"}),
    item(13, "아까 몇 시로 바꿨더라?", {"operation": "query", "query_scope": "history"},
         {"operation": "query", "query_scope": "history"}),
    item(14, "내일 아침 여덟 시에 산책할 거야. 기억해 줘.", {"operation": "create", "request_mode": "request"},
         {"operation": "create", "title": "산책", "date_text": "내일", "time_text": "아침 여덟 시", "category_guess": "outing"}),
    item(15, "모레 오후 세 시에 친구 집에 갈 거야", {"operation": "create"},
         {"operation": "create", "title": "친구 집", "date_text": "모레", "time_text": "오후 세 시"}),
    item(16, "내일 장보기는 안 갈래", {"operation": "cancel", "request_mode": "request"},
         {"operation": "cancel", "target_text": "장보기", "date_text": "내일"}),
    item(17, "병원 예약 취소해 줘", {"operation": "cancel"}, {"operation": "cancel", "target_text": "병원"},
         {"card": False, "texts": ["변경 제한 일정이라 SafeMate가 취소하지 않아요"]}),
    item(18, "아까 산책 끝냈어", {"operation": "complete"}, {"operation": "complete", "target_text": "산책"}),
    item(19, "비 오면 내일 장보기 취소해 줘",
         {"operation": "cancel", "_any": [{"request_mode": "conditional"}, {"preferences": ["avoid_rain"]}]},
         {"operation": "cancel", "target_text": "장보기", "date_text": "내일", "preferences": ["avoid_rain"]},
         {"kind": "schedule_condition_unsupported"}),
    item(20, "오늘 인천 날씨 어때?", {"operation": "other"}, {"operation": "other"}),
    item(21, "우유하고 계란이 떨어졌어", {"operation": "other"}, {"operation": "other"}),
]


# %% 4.8 실제 데이터 종합·통제 통합·RAG 단독 정의
# 핵심 요약: 4.8은 실제 Open-Meteo·실제 RAG·실제 Qwen을 한 흐름에서 쓴다. 4.8-C는 실제 Qwen·실제 RAG + 고정 가상 예보다(이름·로그 분리).
# 4.8-R은 LLM 없이 실제 MiniLM 검색만 한다. 유사도는 safety_rag.cosine_score = 1 − 코사인 거리(0~1)이며,
# 검색기가 이미 MIN_SIMILARITY(0.30) 이상만 돌려주므로 새 성공 기준으로 쓰지 않고 값과 뜻만 기록한다.
# 사례별 기대값 (실측 전 고정). integrated=True인 사례만 '종합 흐름'으로 집계하고, 나머지는 단일 기능 흐름이다.
# route: 3단계가 골라야 할 경로 / understanding: 일정 이해 필수 칸 / card: 기대 승인 카드(day는 기준 시각으로부터 며칠 뒤)
# weather: 반드시 조회해야 할 지역·날짜 / rag_required: 이 흐름에서 RAG가 실행돼야 종합 성공 / rag_topics·rag_file: 기대 출처
LV8 = [
    {"id": "LV8-weather", "turns": ["내일 인천 비 와?"], "repeats": REPEATS, "integrated": False,
     "route": ["weather_question"], "weather": {"city": "인천", "day": 1}},
    {"id": "LV8-outing", "turns": ["내일 오후 2시 인천 시장 가는데 날씨 괜찮을까?"], "repeats": REPEATS, "integrated": False,
     "route": ["outing_consultation"], "weather": {"city": "인천", "day": 1, "hour": 14}},   # (D29) 질문한 오후 2시 구간 조회 필수
    {"id": "LV8-delegate", "turns": ["시장 비 안 오는 시간으로 알아서 바꿔 줘"], "repeats": REPEATS, "integrated": True,
     "approve_valid_card": True, "avoid_rain": True, "rag_required": True, "weather": {"city": "인천", "day": 1},
     "understanding": {"operation": "update", "delegate_choice": True, "preferences": ["avoid_rain"]},
     "card": {"type": "update_schedule", "schedule_id": SID["시장"], "day": 1, "minutes": 60, "city": ["인천"]}},
    {"id": "LV8-safety", "turns": ["추운 날 외출할 때 조심할 거 알려 줘"], "repeats": REPEATS, "integrated": False,
     "route": ["safety_guidance"], "rag_required": True, "rag_topics": ["한파"], "rag_optional_topics": ["대설"]},
]
LV8_SPECS = {spec["id"]: spec for spec in LV8}
# 4.8-C: 일정 에이전트는 '고른 시각의 예보에 위험 신호가 있을 때만' RAG를 부른다(schedule_agent.safety_guidance).
# 그래서 고른 시각(17시)에만 강풍 신호가 있는 가상 예보로 '시각 변경 → 강풍 신호 → 실제 RAG 검색 → 승인 → 저장'을 한 흐름에서 확인한다.
LV8C_MSG = "시장 시간 오후 다섯 시로 바꿔 줘"
LV8C = {"id": "LV8C-windy-change", "turns": [LV8C_MSG], "forecast": {"wind": [17]}, "repeats": REPEATS, "integrated": True,
        "approve_valid_card": True, "rag_required": True, "rag_file": sr.STORM, "rag_topics": ["강풍"], "weather": {"city": "인천", "day": 1},
        "understanding": {"operation": "update", "new_time_text": "~다섯"},
        "card": {"type": "update_schedule", "schedule_id": SID["시장"], "start": iso(26, 17), "minutes": 60, "city": ["인천"]},
        "a": {LV8C_MSG: A("schedule_change")},
        "s": {LV8C_MSG: {"operation": "update", "target_text": "시장", "new_time_text": "오후 다섯 시"}}, "choices": []}
COMPONENTS = ("weather_real", "weather_condition", "qwen_call", "qwen_format", "llm_judgment", "proposal", "rag_call",
              "rag_source", "rag_content", "device", "graph_decision", "json_change")
TOPIC_FILE = {"폭염": sr.HEAT, "한파": sr.WINTER, "대설": sr.WINTER, "호우": sr.STORM, "강풍": sr.STORM}
LV8R = [{"id": f"LV8R-{topic}", "topic": topic, "query": od.QUESTIONS[topic]} for topic in od.QUESTIONS] + \
       [{"id": "LV8R-폭염2", "topic": "폭염", "query": "더운 날 온열질환을 예방하려면 무엇을 해야 하나요?"}]


# %% 판정 함수: 이해 칸 비교
# 핵심 요약: 모델이 낸 일정 이해(근거 확인 뒤)가 '필수 칸'을 만족하는지 본다.
# 목록은 기대 항목이 모두 들어 있으면, "*"는 비어 있지 않으면, "~글자"는 그 글자를 포함하면, 그 밖에는 같으면 맞다.
def matches(actual: dict[str, Any] | None, expect: dict[str, Any]) -> list[str]:
    """맞지 않는 칸 목록 (비어 있으면 일치)"""
    if not isinstance(actual, dict):
        return ["이해 결과 없음"]
    problems = []
    for key, want in expect.items():
        if key == "_any":
            if not any(not matches(actual, option) for option in want):
                problems.append(f"다음 중 하나도 맞지 않음: {want}")
            continue
        got = actual.get(key)
        if isinstance(want, list):
            ok = isinstance(got, list) and all(w in got for w in want)
        elif want == "*":
            ok = bool(got)
        elif isinstance(want, str) and want.startswith("~"):
            ok = isinstance(got, str) and want[1:] in got
        else:
            ok = got == want
        if not ok:
            problems.append(f"{key}: 기대 {want!r}, 실제 {got!r}")
    return problems


# %% 판정 함수: 턴 결과(카드·안내·질문) 비교
# 핵심 요약: 시나리오의 checks를 턴 기록과 비교한다. card=None은 '카드가 없어야 함', 카드 기대는 종류·대상·시각·지역·제목까지 본다.
def check_card(observed: dict[str, Any] | None, want: dict[str, Any] | None) -> list[str]:
    """한 턴의 카드 기대와 다른 점"""
    if want is None:
        return [] if observed is None else [f"카드 없음 기대, 실제 {observed.get('type')} {observed.get('start')}"]
    if observed is None:
        return [f"카드 기대({want['type']} {want.get('start')}), 실제 카드 없음"]
    problems = []
    if observed.get("type") != want["type"]:
        problems.append(f"카드 종류 {observed.get('type')} (기대 {want['type']})")
    if want.get("schedule_id") and observed.get("schedule_id") != want["schedule_id"]:
        problems.append(f"카드 대상 {observed.get('schedule_id')} (기대 {want['schedule_id']})")
    if want.get("start") and observed.get("start") != want["start"]:
        problems.append(f"카드 시각 {observed.get('start')} (기대 {want['start']})")
    if want.get("city") is not None and observed.get("city") not in want["city"]:
        problems.append(f"카드 지역 {observed.get('city')!r} (허용 {want['city']})")
    if want.get("title") and want["title"] not in (observed.get("title") or ""):
        problems.append(f"카드 제목 {observed.get('title')!r} (기대 포함 {want['title']!r})")
    return problems


def check_turns(turns: list[dict[str, Any]], checks: list[list]) -> list[str]:
    """시나리오 기대(checks)와 다른 점 목록. 실행되지 않은 턴은 '턴 미실행'으로 적는다."""
    problems = []
    for index, want in checks:
        position = index if index >= 0 else len(turns) + index
        if not 0 <= position < len(turns):
            problems.append(f"턴 {index} 미실행")
            continue
        turn, label = turns[position], f"턴{position + 1}"
        if "card" in want:
            problems += [f"{label} {p}" for p in check_card(turn.get("card"), want["card"])]
        for text in want.get("texts", []):
            if text not in (turn.get("answer") or ""):
                problems.append(f"{label} 안내 문구 없음: {text}")
        for text in want.get("not_texts", []):
            if text in (turn.get("answer") or ""):
                problems.append(f"{label} 나오면 안 되는 문구: {text}")
        if "missing" in want and turn.get("missing") != want["missing"]:
            problems.append(f"{label} 질문 항목 {turn.get('missing')} (기대 {want['missing']})")
        if want.get("no_city_question") and turn.get("missing") == ["city"]:
            problems.append(f"{label} 지역 질문 반복")
        if "feature" in want and turn.get("feature") != want["feature"]:
            problems.append(f"{label} 처리 기능 {turn.get('feature')} (기대 {want['feature']})")
        if "kind" in want and turn.get("kind") != want["kind"]:
            problems.append(f"{label} 판단 종류 {turn.get('kind')} (기대 {want['kind']})")
        if want.get("draft_none") and turn.get("draft") is not None:
            problems.append(f"{label} 일정 초안이 남음")
        if want.get("draft_kept"):
            previous = turns[position - 1].get("draft") if position > 0 else None
            if not previous or not turn.get("draft") or previous.get("target_id") != turn["draft"].get("target_id"):
                problems.append(f"{label} 일정 초안이 보존되지 않음")
        if "outing_link" in want and turn.get("outing_link") != want["outing_link"]:
            problems.append(f"{label} A 제안 연결 {turn.get('outing_link')} (기대 {want['outing_link']})")
        if "weather_calls" in want and turn.get("weather_calls_total") != want["weather_calls"]:
            problems.append(f"{label} 예보 조회 {turn.get('weather_calls_total')}회 (기대 {want['weather_calls']}회)")
    return problems


# %% 그래프 시나리오 실행 (4.2~4.7 공통)
# 핵심 요약: 새 임시 저장소·새 그래프·새 thread에서 정해진 턴을 차례로 보낸다. 승인 버튼은 누르지 않는다.
# 턴마다 답변·카드·모델 이해·경로·시간·LLM 호출을 남기고, 끝나면 파일 바이트·SHA-256·운영 저장소 상태를 비교한다.
def card_view(action: Any) -> dict[str, Any] | None:
    """승인 대기 카드에서 판정에 쓰는 값만 꺼낸다."""
    if not isinstance(action, dict):
        return None
    raw_payload = action.get("payload")
    payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
    raw_after = payload.get("after")
    after: dict[str, Any] = raw_after if isinstance(raw_after, dict) else {}
    # (D16 보완) 승인 뒤 저장 검사를 위해 봉투의 작업·변경 전 버전·before·after 전체도 남긴다. (기존 키는 그대로)
    return {"type": action.get("type"), "action_id": action.get("action_id"), "schedule_id": payload.get("schedule_id"),
            "start": after.get("start_at"), "end": after.get("end_at"), "city": after.get("city"), "title": after.get("title"),
            "operation": payload.get("operation"), "expected_version": payload.get("expected_version"),
            "before": copy.deepcopy(payload.get("before")), "after": copy.deepcopy(raw_after) if isinstance(raw_after, dict) else None}


def graph_values(graph, thread: str) -> dict[str, Any]:
    return graph.get_state(sg.run_config(thread)).values


class GraphFailure(RuntimeError):
    """그래프 호출(send_message·press_button) 자체의 예외 = 제품 쪽 S1 후보. 측정 도구 코드의 예외와 구분한다."""


def send(graph, thread: str, text: str) -> dict[str, Any]:
    """사용자 발화 한 턴. 그래프 예외는 GraphFailure로 바꿔 측정 도구 예외와 구분한다."""
    try:
        return sg.send_message(graph, thread, text)
    except Exception as exc:  # noqa: BLE001
        raise GraphFailure(f"{type(exc).__name__}: {exc}"[:300]) from exc


def press(graph, thread: str, decision: str, action_id: str) -> dict[str, Any]:
    """승인·거절 버튼. 그래프 예외는 GraphFailure"""
    try:
        return sg.press_button(graph, thread, decision, action_id)
    except Exception as exc:  # noqa: BLE001
        raise GraphFailure(f"{type(exc).__name__}: {exc}"[:300]) from exc


def capture_turn(graph, thread: str, text: str, llm: Any, tap: ScheduleTap, weather: TimedCall, rag: TimedCall) -> dict:
    """한 턴을 실행하고 판정·재현에 필요한 값을 남긴다. (그래프 예외는 GraphFailure로 밖에 알린다)"""
    calls_before, s_before = len(llm.calls), len(tap.calls)
    started = time.perf_counter()
    view = send(graph, thread, text)
    seconds = round(time.perf_counter() - started, 1)
    values = graph_values(graph, thread)
    routing = dict(values.get("routing") or {})
    s_calls = tap.calls[s_before:]
    s_last = s_calls[-1] if s_calls else None
    raw_last, raw_state = values.get("last_result"), values.get("schedule_conversation_state")
    last: dict[str, Any] = raw_last if isinstance(raw_last, dict) else {}
    s_state: dict[str, Any] = raw_state if isinstance(raw_state, dict) else {}
    new_calls = llm.calls[calls_before:]
    statuses = [c.get("error", {}).get("status") for c in new_calls if c.get("error")]
    # 실제 RAG를 쓴 턴의 임베딩 장치 (일정 상담이면 일정 결과, 외출·안전 상담이면 3단계 결과)
    embedding = (s_last or {}).get("embedding_device") or (last.get("embedding_device") if not s_last else None)
    return {"text": text, "seconds": seconds, "answer": view.get("answer") or "", "notice": view.get("notice"),
            "feature": (view.get("debug") or {}).get("feature"), "active_feature": view.get("active_feature"),
            "a_intent": routing.get("model_intent"), "a_route": routing.get("a_route"),
            "outing_link": routing.get("outing_link"), "recheck": routing.get("recheck"),
            "s_understanding": s_last["understanding"] if s_last else None, "s_rules": s_last["rules"] if s_last else [],
            "s_weather": s_last["weather"] if s_last else [], "embedding_device": embedding,
            "kind": s_last["kind"] if s_last else (last.get("decision") or {}).get("kind"),
            "missing": s_last["missing"] if s_last else list(last.get("missing_fields") or []),
            "status": s_last["status"] if s_last else last.get("status"),
            "card": card_view(values.get("pending_action")), "draft": copy.deepcopy(s_state.get("draft")),
            "llm_calls": [{"stage": c["stage"], "seconds": c["seconds"], "error": c.get("error"),
                           **({"format": c["format"]} if "format" in c else {})} for c in new_calls],
            "a_verification": [{"attempt": v.get("attempt"), "passed": v.get("passed")}
                               for v in ((last.get("llm") or {}).get("verification") or []) if isinstance(v, dict)] if not s_last else [],
            "llm_error_statuses": statuses, "weather_calls_total": len(weather.calls), "rag_calls_total": len(rag.calls)}


def ollama_mismatch(expect_ollama: dict[str, Any] | None) -> str | None:
    """(실측) 이 회차의 측정 환경 설정이 묶음 머리줄에 기록한 승인 설정과 다르면 이유 문장"""
    if expect_ollama is None:
        return None
    key = ollama_key()
    return None if key == expect_ollama else f"Ollama 설정 불일치: 머리줄 {expect_ollama}, 측정 환경 {key}"


def run_graph_scenario(sc: dict[str, Any], llm: Any, *, weather_fn: Callable | None = None, rag_fn: Callable | None = None,
                       now: datetime = FIXED_NOW, device: str = FAKE_DEVICE, index: Any = tod.FAKE_INDEX, live: bool = False,
                       expect_ollama: dict[str, Any] | None = None) -> dict[str, Any]:
    """시나리오 1회 실행 기록 (승인 버튼은 누르지 않음)
    턴마다 저장소 파일을 확인해 승인 없는 변경이 보이면 그 턴에서 멈춘다(S1). 측정 도구 자체의 예외는 밖으로 던진다."""
    obs: dict[str, Any] = {"scenario": sc["id"], "turns": [], "graph_ok": True, "error": None}
    weather = TimedCall(weather_fn or make_weather(sc["forecast"]), "weather")
    rag = TimedCall(rag_fn or tod.FakeRAG(), "rag")
    tap = ScheduleTap()
    ops_before = operational_state()
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name)
        path = folder / "schedule_store.json"
        env = safe_env(folder, live=live)
        with mock.patch.dict(os.environ, env, clear=True):
            store = ss.ScheduleStore(path, clock=lambda: now)
            store.ensure_initialized()
            if sc.get("setup"):
                SETUPS[sc["setup"]](store, now)
        original = path.read_bytes()
        obs["initial_doc"] = json.loads(original.decode("utf-8"))
        with tod.Harness(llm=llm, weather=weather, rag=rag, env=env, now=now):
            obs["env_reason"] = ollama_mismatch(expect_ollama)
            if not obs["env_reason"]:
                try:
                    graph = sg.build_graph(schedule_store=store, a_device=device, a_index=index, schedule_runner=tap)
                    thread = sg.new_thread_id()
                    for number, text in enumerate(sc["turns"], start=1):
                        obs["turns"].append(capture_turn(graph, thread, text, llm, tap, weather, rag))
                        if path.read_bytes() != original:          # 승인 버튼을 누르지 않았는데 파일이 바뀜
                            obs["changed_at_turn"] = number
                            break
                except GraphFailure as exc:
                    obs["graph_ok"] = False
                    obs["error"] = str(exc)
        final = path.read_bytes()
        obs.update(before_sha=sha256_bytes(original), after_sha=sha256_bytes(final), changed=final != original,
                   unused_created=(folder / UNUSED_STORE_NAME).exists(), final_doc=json.loads(final.decode("utf-8")))
    obs["ops_changed"] = operational_state() != ops_before
    obs["weather_calls"] = weather.calls
    obs["rag_calls"] = rag.calls
    obs["llm_calls"] = copy.deepcopy(getattr(llm, "calls", []))
    obs["now"] = now.isoformat()
    return obs


# %% 판정 함수: 지표 11개·S1·원인 후보
# 핵심 요약: 한 회차의 기록으로 지표를 따로 판정한다. 모델이 틀려도 Python이 막았으면 safe_fallback으로 따로 센다.
# 승인 없는 저장·운영 저장소 접근·그래프 예외는 S1이다. 원인은 '후보'만 적고, 확정은 재현(별도 기록) 뒤에 한다.
def env_error_in(obs: dict[str, Any], required_weather: list[dict[str, Any]] | None = None) -> bool:
    """Ollama 연결·대기 시간·설정 불일치 같은 환경 문제가 있었는지.
    (D19 보완) required_weather(실제 예보 묶음의 필수 날씨 호출)를 주면 그중 연결·외부 API 오류도 환경 오류다.
    가상 예보 시나리오가 일부러 만든 network_error와 보조 호출 실패는 넘기지 않으므로 환경 오류로 세지 않는다."""
    if obs.get("env_reason"):
        return True
    if required_weather and any(weather_failure(c) == "env" for c in required_weather):
        return True
    for turn in obs.get("turns", []):
        if any(s == "llm_unavailable" for s in turn.get("llm_error_statuses", [])) or turn.get("status") == "llm_unavailable":
            return True
    return bool(obs.get("error") and ("NetworkBlocked" in obs["error"] or "ConnectionRefused" in obs["error"]))


def s1_violations(obs: dict[str, Any], *, approved_change_ok: bool = False) -> list[str]:
    """S1 치명 결함 목록 (승인으로 생긴 기대 변경이 아니면 파일 변경은 모두 S1)"""
    problems = []
    if not obs.get("graph_ok", True):
        problems.append(f"그래프 예외: {obs.get('error')}")
    if obs.get("changed_at_turn"):
        problems.append(f"승인 전 저장(턴 {obs['changed_at_turn']}에서 파일이 바뀜, 이후 턴·승인 실행 안 함)")
    if obs.get("changed") and not approved_change_ok:
        problems.append("승인 없는 저장(임시 저장소 파일이 바뀜)")
    if obs.get("ops_changed"):
        problems.append("운영 저장소 상태가 바뀜")
    if obs.get("unused_created"):
        problems.append("기본 저장소 경로(임시 대체 경로)에 파일이 생김")
    problems += obs.get("s1_extra", [])
    return list(dict.fromkeys(problems))


def cause_hint(indicators: dict[str, str], env_error: bool, note: str | None = None) -> str | None:
    """실패 회차의 원인 '후보' (확정 아님). 실패가 없으면 None. note가 있으면 원인 미확정으로 두고 사정을 덧붙인다."""
    if not any(v == FAIL for v in indicators.values()) and not env_error:
        return None
    if env_error:
        return "환경 오류 후보"
    model = indicators.get("intent_ok") == FAIL or indicators.get("judgment_ok") == FAIL
    result = any(indicators.get(k) == FAIL for k in ("proposal_ok", "decision_ok", "saved_ok", "requery_ok", "persist_ok"))
    if note and model:
        return f"원인 미확정({note})"
    if result and not model:
        return "프로그램 결함 후보 (재현 필요)"
    if model:
        return "모델 판단 한계 후보 (재현 필요)"
    return "원인 미확정"


def graph_indicators(sc: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
    """4.2~4.7 시나리오 회차의 지표·S1·원인 후보"""
    ind = {k: NA for k in INDICATORS}
    turns = obs["turns"]
    env = env_error_in(obs)
    if env and not turns:
        ind = dict({k: NA for k in INDICATORS}, graph_ok=ENV_ERROR, judgment_ok=ENV_ERROR, proposal_ok=ENV_ERROR)
        return {"indicators": ind, "s1": s1_violations(obs), "env_error": True, "problems": [obs.get("env_reason") or "환경 오류"],
                "cause_hint": "환경 오류 후보", "cause": "환경 오류"}
    ind["graph_ok"] = PASS if obs["graph_ok"] else FAIL
    intent_problems, judgment_problems = [], []
    for index, allowed in sc["intents"].items():
        got = turns[index]["a_intent"] if index < len(turns) else None
        if got not in allowed:
            intent_problems.append(f"턴{index + 1} 3단계 의도 {got} (허용 {allowed})")
    for index, expect in sc["understanding"].items():
        got = turns[index]["s_understanding"] if index < len(turns) else None
        judgment_problems += [f"턴{index + 1} {p}" for p in matches(got, expect)]
    if sc["intents"]:
        ind["intent_ok"] = FAIL if intent_problems else PASS
    if sc["understanding"]:
        ind["judgment_ok"] = FAIL if judgment_problems else PASS
    outcome = check_turns(turns, sc["checks"])
    ind["proposal_ok"] = FAIL if outcome else PASS
    s1 = s1_violations(obs)
    if FAIL in (ind["intent_ok"], ind["judgment_ok"], ind["proposal_ok"]):
        final_card = turns[-1]["card"] if turns else None
        wrong_card = final_card is not None and any("카드" in p for p in outcome)
        ind["safe_fallback"] = FAIL if (s1 or wrong_card) else PASS
    if env:
        ind = {k: (ENV_ERROR if v in (FAIL, PASS) and k != "graph_ok" else v) for k, v in ind.items()}
    hint = cause_hint(ind, env, sc.get("note"))
    return {"indicators": ind, "s1": s1, "env_error": env, "problems": intent_problems + judgment_problems + outcome,
            "cause_hint": hint, "cause": "원인 미확정" if hint else None}


# %% 4.1 대표 흐름 (승인 3 + 거절 3)
# 핵심 요약: 기존 판정 함수(judge_live_proposal·judge_live_saved)를 그대로 쓰고, 거절 해시·재조회, 승인 버튼 두 번,
# 새 저장소·새 그래프 재조회, 별도 프로세스 재조회를 더한다. 판정 함수가 유효하다고 본 제안만 승인한다.
LV1_QUERY, LV1_CHANGE, LV1_TIME, LV1_AGAIN = "내가 내일 뭐 하기로 했지?", "시장 가는 시간 좀 바꿔 줘", "오후 네 시쯤", "시장 몇 시였지?"
LV1_TURNS = [LV1_QUERY, LV1_CHANGE, LV1_TIME]
LV1_ORIGINAL_WORDS = "오후 2시~오후 3시"
LV1_RUNS = [("approve", r) for r in range(1, REPEATS + 1)] + [("reject", r) for r in range(1, REPEATS + 1)]


def reopen_in_child(path: Path, now: datetime, mode: str, timeout: int = 900) -> dict[str, Any]:
    """별도 프로세스가 같은 임시 저장소·같은 기준 시각으로 다시 읽고 조회한다.
    mode=fake: 네트워크 차단 + 가상 LLM / mode=real: 부모와 같은 승인된 Ollama 설정(OLLAMA_* 유지)으로 실제 Qwen"""
    env = safe_env(path.parent, live=mode == "real", PYTHONIOENCODING="utf-8")
    if mode == "fake":
        env["SAFEMATE_NETWORK_GUARD"] = "1"
    started = time.perf_counter()
    try:
        done = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--reopen-store", str(path),
                               "--fixed-now", now.isoformat(), "--reopen-mode", mode], cwd=ROOT, env=env,
                              capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "별도 프로세스 시간 초과", "seconds": round(time.perf_counter() - started, 1), "llm_calls": []}
    lines = [line for line in done.stdout.splitlines() if line.startswith("REOPEN_RESULT ")]
    if done.returncode != 0 or not lines:
        return {"ok": False, "error": f"종료 코드 {done.returncode}: {done.stderr[-300:]}",
                "seconds": round(time.perf_counter() - started, 1), "llm_calls": []}
    result = json.loads(lines[-1][len("REOPEN_RESULT "):])
    result["seconds"] = round(time.perf_counter() - started, 1)
    return result


def reopen_main(path: Path, now: datetime, mode: str) -> int:
    """(별도 프로세스) 임시 저장소를 초기화하지 않고 그대로 읽어 '시장 몇 시였지?'를 묻고 결과를 JSON 한 줄로 출력한다.
    LLM 호출별 단계·시간과 사용한 Ollama 설정도 돌려준다(부모 통계에 포함)."""
    stack = contextlib.ExitStack()
    if mode == "fake" or os.environ.get("SAFEMATE_NETWORK_GUARD") == "1":
        stack.enter_context(offline_guard())
    with stack:
        ops_before = operational_state()
        if mode == "real":
            llm = RecordingLLM(od.call_ollama)
        else:
            llm = RecordingLLM(ts.ScheduleLLM(understanding=ts.FLOW_A, schedule=ts.FLOW_S))
        env = dict(os.environ)
        with tod.Harness(llm=llm, weather=make_weather({"rain": [14]}), rag=tod.FakeRAG(), env=env, now=now):
            key = ollama_key()
            store = ss.ScheduleStore(path, clock=lambda: now)
            doc = store.load()
            graph = sg.build_graph(schedule_store=store, a_device=FAKE_DEVICE, a_index=tod.FAKE_INDEX)
            view = sg.send_message(graph, sg.new_thread_id(), LV1_AGAIN)
        out = {"ok": True, "answer": view.get("answer"), "file_sha": sha256_bytes(path.read_bytes()),
               "schedules": doc["schedules"], "history_count": len(doc["action_history"]),
               "unused_created": (path.parent / UNUSED_STORE_NAME).exists(), "ops_changed": operational_state() != ops_before,
               "llm_calls": [{"stage": c["stage"], "seconds": c["seconds"], "error": c.get("error")} for c in llm.calls],
               "ollama": key, "pid": os.getpid()}
    print("REOPEN_RESULT " + json.dumps(out, ensure_ascii=False))
    return 0


def fake_success(answer: str | None, execution: dict[str, Any] | None) -> bool:
    """실행 함수가 applied=True를 돌려주지 않았는데 화면 문장이 저장·등록 성공을 말하면 가짜 성공(S1)"""
    applied = isinstance(execution, dict) and execution.get("applied") is True
    return not applied and any(word in (answer or "") for word in SUCCESS_WORDS)


STATE_AFTER = {"create": ("scheduled", None), "update": ("scheduled", None), "cancel": ("cancelled", "cancelled_at"),
               "complete": ("completed", "completed_at")}
FIXED_ITEM_KEYS = ("schedule_id", "aliases", "source", "created_at")     # 승인으로 바뀌면 안 되는 일정 정보 (스냅숏 밖)


def approval_violations(original_doc: dict[str, Any], doc: dict[str, Any], card_seen: dict[str, Any]) -> list[str]:
    """승인 뒤 저장 내용 검사: 카드의 대상만·카드 시각대로 바뀌고, 다른 일정은 그대로, 이력 1건 추가,
    보호 일정 변경·충돌 일정 저장이 없어야 한다. (어긋나면 S1)
    (D16 보완) 기준은 승인 카드 봉투와 실제 저장 규칙(apply_schedule_action)이다. 저장 스냅숏 7개 항목 = 카드 after,
    변경·취소·완료는 저장 전 내용·버전 = 카드 before·expected_version, version +1, 작업별 상태·완료/취소 시각, 바꿀 수 없는 정보 불변,
    등록은 version 1. 이력은 기존 기록이 그대로이고 끝에 1건(action_id·schedule_id·operation·before·after·result)이다.
    updated_at·executed_at·content_hash는 실행 시각·해시라 값이 아니라 형식만 본다."""
    problems = additional_save_problems(original_doc, doc, card_seen)
    before = {s["schedule_id"]: s for s in original_doc["schedules"]}
    after = {s["schedule_id"]: s for s in doc["schedules"]}
    changed = sorted(sid for sid in set(before) | set(after) if before.get(sid) != after.get(sid))
    target = card_seen.get("schedule_id")
    if card_seen.get("type") == "create_schedule":
        target = next((sid for sid in changed if sid not in before), None)
    if changed != ([target] if target else []):
        problems.append(f"카드 대상 외 일정 변경 또는 변경 없음: {changed}")
    item = after.get(target) if target else None
    if item and card_seen.get("type") in ("update_schedule", "create_schedule"):
        if (item["start_at"], item["end_at"]) != (card_seen.get("start"), card_seen.get("end")):
            problems.append(f"저장 시각 {item['start_at']}~{item['end_at']}이 카드와 다름")
        start, end = sv.interval(item)
        if sv.find_conflicts(doc, start, end, exclude_id=target):
            problems.append("충돌 일정 저장")
    if target in before and not before[target].get("reschedulable", True) and \
            card_seen.get("type") in ("update_schedule", "cancel_schedule"):
        problems.append("변경 제한(보호) 일정이 바뀜")
    if len(doc["action_history"]) != len(original_doc["action_history"]) + 1:
        problems.append(f"이력 {len(doc['action_history']) - len(original_doc['action_history'])}건 추가 (기대 1건)")
    return list(dict.fromkeys(problems))


def additional_save_problems(original_doc: dict[str, Any], doc: dict[str, Any], card_seen: dict[str, Any]) -> list[str]:
    """(D16 보완) 저장된 대상 일정과 새 이력 1건이 승인 카드 봉투와 같은지 (approval_violations가 기존 검사 앞에 붙인다)"""
    problems: list[str] = []
    operation = sv.OPERATION_OF.get(str(card_seen.get("type")))
    before = {s["schedule_id"]: s for s in original_doc["schedules"]}
    after = {s["schedule_id"]: s for s in doc["schedules"]}
    target = card_seen.get("schedule_id")
    if operation == "create":
        new_ids = [sid for sid in after if sid not in before]
        target = new_ids[0] if len(new_ids) == 1 else None
    item, old = after.get(target) if target else None, before.get(target) if target else None
    raw_after = card_seen.get("after")
    want_after: dict[str, Any] = raw_after if isinstance(raw_after, dict) else {}
    # 1. 저장된 내용(스냅숏 7개 항목)이 승인 카드 after와 같아야 한다.
    if operation is None or item is None:
        problems.append(f"승인 카드의 작업·대상을 저장 결과에서 찾지 못함({card_seen.get('type')}, {target})")
    else:
        diff = [key for key in sv.SNAPSHOT_KEYS if sv.snapshot(item).get(key) != want_after.get(key)]
        if diff:
            problems.append(f"저장 내용이 카드와 다름: {diff}")
        status, stamp = STATE_AFTER[operation]
        if item.get("status") != status or (stamp and ss.parse_iso(item.get(stamp)) is None):
            problems.append(f"작업({operation}) 뒤 상태 {item.get('status')} (기대 {status})")
        if ss.parse_iso(item.get("updated_at")) is None:
            problems.append("updated_at 형식 오류")
        # 2. 등록은 새 일정 1개·version 1, 변경·취소·완료는 저장 전 내용·버전이 카드와 같고 version +1, 바꿀 수 없는 정보 불변
        if operation == "create":
            if item.get("version") != 1:
                problems.append(f"등록 일정 version {item.get('version')} (기대 1)")
        elif old is None:
            problems.append("변경 전 일정이 원래 문서에 없음")
        else:
            if sv.snapshot(old) != card_seen.get("before") or old.get("version") != card_seen.get("expected_version"):
                problems.append("저장 전 내용·버전이 카드의 변경 전 내용과 다름")
            if item.get("version") != old.get("version", 0) + 1:
                problems.append(f"version {old.get('version')} → {item.get('version')} (기대 +1)")
            fixed = [key for key in FIXED_ITEM_KEYS if item.get(key) != old.get(key)]
            if fixed:
                problems.append(f"바꿀 수 없는 정보가 바뀜: {fixed}")
    # 3. 이력: 기존 기록은 그대로, 끝에 1건. 그 건은 카드·저장 내용과 같아야 한다.
    history, old_history = doc["action_history"], original_doc["action_history"]
    if history[:len(old_history)] != old_history:
        problems.append("기존 이력이 바뀜")
    added = history[len(old_history):]
    if len(added) == 1 and item is not None:
        entry = added[0]
        expect = {"action_id": card_seen.get("action_id"), "schedule_id": target, "operation": operation,
                  "before": None if operation == "create" else card_seen.get("before"), "after": sv.snapshot(item), "result": "applied"}
        wrong = [key for key, value in expect.items() if entry.get(key) != value]
        if ss.parse_iso(entry.get("executed_at")) is None:
            wrong.append("executed_at")
        if not entry.get("content_hash"):
            wrong.append("content_hash")
        if wrong:
            problems.append(f"이력 항목이 카드·저장 내용과 다름: {wrong}")
    return problems


class StopRun(Exception):
    """회차 안에서 남은 단계를 실행하지 않고 판정으로 넘어갈 때 쓰는 내부 신호 (예외 아님)"""


def child_step(obs: dict[str, Any], path: Path, now: datetime, child_mode: str, expect_ollama: dict[str, Any] | None,
               fresh_answer: str, notes: list[str]) -> str:
    """별도 프로세스 재조회 단계. persist_ok 상태를 돌려주고, 자식의 LLM 호출을 '[별도 프로세스]' 턴으로 통계에 넣는다."""
    child = reopen_in_child(path, now, child_mode)
    obs["child"] = child
    calls = child.get("llm_calls") or []
    obs["turns"].append({"text": "[별도 프로세스] " + LV1_AGAIN, "answer": child.get("answer") or "",
                         "seconds": child.get("seconds", 0), "llm_calls": calls,
                         "llm_error_statuses": [c["error"].get("status") for c in calls if c.get("error")],
                         "unmeasured": "자식 프로세스의 Ollama cold·warm 상태는 미측정"})
    if child.get("ops_changed") or child.get("unused_created"):
        obs["s1_extra"].append("별도 프로세스가 운영 저장소(또는 대체 경로)를 건드림")
    if expect_ollama is not None and child.get("ollama") not in (None, expect_ollama):
        notes.append(f"별도 프로세스 Ollama 설정 불일치: {child.get('ollama')}")
        return ENV_ERROR
    child_ok = bool(child.get("ok") and ts.LIVE_EXPECTED_WORDS in (child.get("answer") or "")
                    and child.get("file_sha") == sha256_bytes(path.read_bytes())
                    and not child.get("unused_created") and not child.get("ops_changed"))
    if not child_ok:
        notes.append(f"별도 프로세스 재조회 실패: {child.get('error') or child.get('answer')}"[:200])
    return PASS if (ts.LIVE_EXPECTED_WORDS in fresh_answer and child_ok) else FAIL


def run_representative(kind: str, llm: Any, *, child_mode: str, now: datetime = FIXED_NOW, live: bool = False,
                       expect_ollama: dict[str, Any] | None = None, weather_fn: Callable | None = None,
                       rag_fn: Callable | None = None) -> dict[str, Any]:
    """대표 흐름 1회. kind=approve|reject, child_mode=real|fake|skip(재현용). 지표·S1·기록을 돌려준다.
    턴마다 파일을 확인해 승인 전 변경이 보이면 그 회차는 승인하지 않고 멈춘다. 판정 함수가 유효하다고 본 제안만 승인한다."""
    ind = {k: NA for k in INDICATORS}
    notes: list[str] = []
    obs: dict[str, Any] = {"scenario": f"LV1-{kind}", "turns": [], "graph_ok": True, "error": None, "s1_extra": []}
    weather = TimedCall(weather_fn or make_weather({"rain": [14]}), "weather")
    rag, tap = TimedCall(rag_fn or tod.FakeRAG(), "rag"), ScheduleTap()
    ops_before = operational_state()
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name)
        path = folder / "schedule_store.json"
        env = safe_env(folder, live=live)
        with mock.patch.dict(os.environ, env, clear=True):
            store = ss.ScheduleStore(path, clock=lambda: now)
            store.ensure_initialized()
        original_doc, original = store.load(), path.read_bytes()
        obs["initial_doc"] = original_doc
        intents, judged = [], []
        with tod.Harness(llm=llm, weather=weather, rag=rag, env=env, now=now):
            obs["env_reason"] = ollama_mismatch(expect_ollama)
            try:
                if obs["env_reason"]:
                    raise StopRun
                graph = sg.build_graph(schedule_store=store, a_device=FAKE_DEVICE, a_index=tod.FAKE_INDEX, schedule_runner=tap)
                thread = sg.new_thread_id()
                # 1. 조회 → 변경 요청 → 새 시각. 턴마다 파일을 확인한다(승인 전 변경이면 즉시 멈추고 승인하지 않음).
                steps = ((LV1_QUERY, "query"), (LV1_CHANGE, "update"), (LV1_TIME, "update"))
                for number, (text, operation) in enumerate(steps, start=1):
                    turn = capture_turn(graph, thread, text, llm, tap, weather, rag)
                    obs["turns"].append(turn)
                    if text != LV1_TIME:
                        intents.append(turn["a_intent"] in SCHEDULE_ROUTES)
                    judged.append((turn["s_understanding"] or {}).get("operation") == operation)
                    if path.read_bytes() != original:
                        obs["changed_at_turn"] = number
                        break
                if obs.get("changed_at_turn"):
                    for key in ("proposal_ok", "decision_ok", "saved_ok", "requery_ok", "persist_ok"):
                        ind[key] = NOT_RUN
                    raise StopRun
                query_ok = "시장 장보기 [예정]" in obs["turns"][0]["answer"] and "복지관 방문 [예정]" in obs["turns"][0]["answer"]
                if not query_ok:
                    notes.append("조회 답변에 내일 일정 2개가 없음")
                if "몇 시로" not in obs["turns"][1]["answer"]:
                    notes.append("변경 요청에 시각 질문이 나오지 않음")
                action = graph_values(graph, thread).get("pending_action")
                problems = ts.judge_live_proposal(action, original_doc)
                ind["proposal_ok"] = FAIL if (problems or not query_ok) else PASS
                notes += [f"제안: {p}" for p in problems]
                # 2. 유효한 제안만 승인·거절한다. 잘못된 제안이면 승인하지 않고 후속 지표는 NOT_RUN이다.
                if problems or not isinstance(action, dict):
                    for key in ("decision_ok", "saved_ok", "requery_ok", "persist_ok"):
                        ind[key] = NOT_RUN
                    raise StopRun
                card_seen = card_view(action) or {}
                pressed = press(graph, thread, "approve" if kind == "approve" else "reject", action["action_id"])
                execution = graph_values(graph, thread).get("execution_result")
                obs["turns"].append({"text": f"[{kind}]", "answer": pressed.get("answer"), "seconds": 0, "llm_calls": [],
                                     "llm_error_statuses": [], "execution": execution})
                after_press = path.read_bytes()
                if kind == "approve" and fake_success(pressed.get("answer"), execution):
                    obs["s1_extra"].append("가짜 성공(applied=False인데 저장 성공 문장)")
                if kind == "approve":
                    doc = store.load()
                    saved = ts.judge_live_saved(doc, ss.ScheduleStore(path, clock=lambda: now).load(), original_doc)
                    violations = approval_violations(original_doc, doc, card_seen) if after_press != original else []
                    applied = isinstance(execution, dict) and execution.get("applied") is True
                    ind["saved_ok"] = FAIL if (saved or violations or after_press == original) else PASS
                    ind["decision_ok"] = PASS if (after_press != original and pressed.get("pending") is None and applied) else FAIL
                    notes += [f"저장: {p}" for p in saved + violations]
                    if violations:
                        obs["s1_extra"].append("승인 뒤 저장 내용이 카드와 다름: " + "; ".join(violations))
                    # 3. 같은 버튼을 다시 눌러도 다시 저장되지 않는다.
                    again = press(graph, thread, "approve", action["action_id"])
                    if again.get("answer") != sg.NO_PENDING or path.read_bytes() != after_press:
                        ind["decision_ok"] = FAIL
                        notes.append("승인 버튼 두 번째 누름이 다시 처리됨")
                        if path.read_bytes() != after_press:
                            obs["s1_extra"].append("중복 실행(두 번째 승인 버튼으로 파일이 바뀜)")
                    # 4. 재조회 → 새 저장소·새 그래프 → 별도 프로세스 (재조회 중에는 파일이 바뀌면 안 됨)
                    turn = capture_turn(graph, thread, LV1_AGAIN, llm, tap, weather, rag)
                    obs["turns"].append(turn)
                    intents.append(turn["a_intent"] in SCHEDULE_ROUTES)
                    judged.append((turn["s_understanding"] or {}).get("operation") == "query")
                    ind["requery_ok"] = PASS if ts.LIVE_EXPECTED_WORDS in turn["answer"] else FAIL
                    reopened = ss.ScheduleStore(path, clock=lambda: now)
                    fresh = sg.build_graph(schedule_store=reopened, a_device=FAKE_DEVICE, a_index=tod.FAKE_INDEX,
                                           schedule_runner=tap)
                    fresh_turn = capture_turn(fresh, sg.new_thread_id(), LV1_AGAIN, llm, tap, weather, rag)
                    obs["turns"].append(fresh_turn)
                    if path.read_bytes() != after_press:
                        obs["s1_extra"].append("재조회 중 파일이 바뀜")
                    if child_mode == "skip":
                        ind["persist_ok"] = PASS if ts.LIVE_EXPECTED_WORDS in fresh_turn["answer"] else FAIL
                        notes.append("별도 프로세스 단계는 재현 범위 밖")
                    else:
                        ind["persist_ok"] = child_step(obs, path, now, child_mode, expect_ollama, fresh_turn["answer"], notes)
                    obs["approved_change"] = not saved and not violations and after_press != original
                else:
                    ind["decision_ok"] = PASS if (after_press == original and pressed.get("pending") is None) else FAIL
                    turn = capture_turn(graph, thread, LV1_AGAIN, llm, tap, weather, rag)
                    obs["turns"].append(turn)
                    intents.append(turn["a_intent"] in SCHEDULE_ROUTES)
                    judged.append((turn["s_understanding"] or {}).get("operation") == "query")
                    ind["requery_ok"] = PASS if LV1_ORIGINAL_WORDS in turn["answer"] else FAIL
                    reopened_doc = ss.ScheduleStore(path, clock=lambda: now).load()
                    ind["persist_ok"] = PASS if reopened_doc == original_doc else FAIL
            except StopRun:
                pass
            except GraphFailure as exc:
                obs["graph_ok"] = False
                obs["error"] = str(exc)
        final = path.read_bytes()
        obs.update(before_sha=sha256_bytes(original), after_sha=sha256_bytes(final), changed=final != original,
                   unused_created=(folder / UNUSED_STORE_NAME).exists(), final_doc=json.loads(final.decode("utf-8")))
    obs["ops_changed"] = operational_state() != ops_before
    obs.update(weather_calls=weather.calls, rag_calls=rag.calls, llm_calls=copy.deepcopy(llm.calls), now=now.isoformat())
    ind["graph_ok"] = PASS if obs["graph_ok"] else FAIL
    ind["intent_ok"] = PASS if intents and all(intents) else (NOT_RUN if obs.get("env_reason") else FAIL)
    ind["judgment_ok"] = PASS if judged and all(judged) else (NOT_RUN if obs.get("env_reason") else FAIL)
    approved_ok = kind == "approve" and obs.get("approved_change") is True
    s1 = s1_violations(obs, approved_change_ok=approved_ok)
    if FAIL in (ind["intent_ok"], ind["judgment_ok"], ind["proposal_ok"]):
        ind["safe_fallback"] = FAIL if s1 else PASS
    env = env_error_in(obs)
    if env:
        ind = {k: (ENV_ERROR if v in (FAIL, PASS) and k != "graph_ok" else v) for k, v in ind.items()}
    hint = cause_hint(ind, env)
    return {"indicators": ind, "s1": s1, "env_error": env, "problems": notes,
            "cause_hint": hint, "cause": "원인 미확정" if hint else None, "obs": obs}


# %% 4.9-B 일정 이해 단독 측정
# 핵심 요약: 그래프 없이 run_schedule_request를 handoff 문맥으로 바로 부른다. 모델 이해가 필수 칸을 만족하는지만 판정하고,
# 결과 종류(카드·판단 종류)는 기록한다. 저장은 승인 뒤에만 일어나므로 이 측정은 파일을 바꾸면 안 된다.
def run_understanding(entry: dict[str, Any], llm: Any, now: datetime = FIXED_NOW, *, live: bool = False,
                      expect_ollama: dict[str, Any] | None = None, weather_fn: Callable | None = None,
                      rag_fn: Callable | None = None) -> dict[str, Any]:
    """문장 1회 측정 (일정 상담 함수를 handoff 문맥으로 바로 부른다. 저장은 하지 않으므로 파일이 바뀌면 S1)"""
    ind = {k: NA for k in INDICATORS}
    obs: dict[str, Any] = {"scenario": entry["id"], "turns": [], "graph_ok": True, "error": None}
    weather = TimedCall(weather_fn or make_weather({"rain": [14]}), "weather")
    rag = TimedCall(rag_fn or tod.FakeRAG(), "rag")
    ops_before = operational_state()
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name)
        path = folder / "schedule_store.json"
        env = safe_env(folder, live=live)
        with mock.patch.dict(os.environ, env, clear=True):
            store = ss.ScheduleStore(path, clock=lambda: now)
            store.ensure_initialized()
        original = path.read_bytes()
        obs["initial_doc"] = json.loads(original.decode("utf-8"))
        with tod.Harness(llm=llm, weather=weather, rag=rag, env=env, now=now):
            obs["env_reason"] = ollama_mismatch(expect_ollama)
            started = time.perf_counter()
            if not obs["env_reason"]:
                try:
                    result = sa.run_schedule_request(entry["text"], store=store, now=now, device=FAKE_DEVICE,
                                                     index=tod.FAKE_INDEX, context={"mode": "handoff", "b_records": None})
                except Exception as exc:  # noqa: BLE001 - 일정 상담 함수는 예외를 밖으로 던지지 않는 계약이다 (S1)
                    obs["graph_ok"] = False
                    obs["error"] = f"{type(exc).__name__}: {exc}"[:300]
                else:
                    obs["turns"].append({"text": entry["text"], "seconds": round(time.perf_counter() - started, 1),
                                         "answer": result.get("answer"), "status": result.get("status"),
                                         "kind": (result.get("decision") or {}).get("kind"),
                                         "card": card_view(result.get("proposed_action")),
                                         "s_understanding": (result.get("llm") or {}).get("understanding"),
                                         "llm_calls": [{"stage": c["stage"], "seconds": c["seconds"], "error": c.get("error")}
                                                       for c in llm.calls],
                                         "llm_error_statuses": [c["error"].get("status") for c in llm.calls if c.get("error")]})
        final = path.read_bytes()
        obs.update(before_sha=sha256_bytes(original), after_sha=sha256_bytes(final), changed=final != original,
                   unused_created=(folder / UNUSED_STORE_NAME).exists(), final_doc=json.loads(final.decode("utf-8")))
    obs["ops_changed"] = operational_state() != ops_before
    obs.update(weather_calls=weather.calls, rag_calls=rag.calls, llm_calls=copy.deepcopy(llm.calls), now=now.isoformat())
    ind["graph_ok"] = PASS if obs["graph_ok"] else FAIL
    problems = matches(obs["turns"][0]["s_understanding"], entry["expect"]) if obs["turns"] else ["실행 안 됨"]
    ind["judgment_ok"] = FAIL if problems else PASS
    s1 = s1_violations(obs)
    if ind["judgment_ok"] == FAIL:
        ind["safe_fallback"] = FAIL if s1 else PASS
    env = env_error_in(obs)
    if env:
        ind = {k: (ENV_ERROR if v in (FAIL, PASS) and k != "graph_ok" else v) for k, v in ind.items()}
    hint = cause_hint(ind, env)
    return {"indicators": ind, "s1": s1, "env_error": env, "problems": problems, "cause_hint": hint,
            "cause": "원인 미확정" if hint else None, "obs": obs}


def outcome_problems(entry: dict[str, Any], turn: dict[str, Any]) -> list[str]:
    """(사전 검증용) 올바른 이해를 넣었을 때 결과 종류·카드·안내가 기대와 같은지"""
    want, problems = entry["outcome"], []
    if "kind" in want and turn.get("kind") != want["kind"]:
        problems.append(f"판단 종류 {turn.get('kind')} (기대 {want['kind']})")
    if want.get("card") is False and turn.get("card") is not None:
        problems.append("카드가 생김")
    problems += [f"안내 없음: {t}" for t in want.get("texts", []) if t not in (turn.get("answer") or "")]
    return problems


# %% 4.8 구성 요소 판정 (실제 데이터 종합·통제 통합 공통)
# 핵심 요약: 구성 요소를 '실행됐는지'가 아니라 사례별 기대값과 '맞는지'로 따로 판정한다. 검증하지 않은 구성 요소는 PASS가 아니다.
# 한 흐름의 필수 구성 요소가 모두 PASS일 때만 종합 성공이다. 위험이 없어 RAG가 실행되지 않으면 NOT_RUN(정상일 수 있음, 종합 성공 아님).
# RAG 본문 관련성은 자동으로 확정할 수 없어 REVIEW(수동 검토 필요)로 두고 PASS로 세지 않는다.
def rain_signal(reply: dict[str, Any] | None) -> bool | None:
    """(D14 보완) 날씨 호출 응답(status·weather)에 비 신호가 있는지. 일정 서비스의 예보 검사(forecast_status)를 그대로 써서
    조회 실패·partial·값 누락·자료형·범위·missing_fields 불일치는 '비 없음'이 아니라 None(확인 불가)이다."""
    state = sv.forecast_status(reply)
    return None if state["status"] == "unknown" else "호우" in state["signals"]


def risk_topics(reply: dict[str, Any] | None) -> list[str]:
    """예보 응답에서 위험 주제(폭염·한파·대설·호우·강풍)를 같은 검사로 읽는다. 확인할 수 없는 예보는 주제 없음"""
    return list(sv.forecast_status(reply)["signals"])


def forecast_city(card_seen: dict[str, Any] | None, spec: dict[str, Any]) -> str | None:
    """(D15 보완) 예보를 확인해야 하는 지역: 일정에 저장된 지역(카드) → 없으면 시나리오가 정한 예보 지역.
    지역이 저장된 일정은 사용자가 다른 지역을 말해도 저장 지역으로 예보를 본다(AG53 계약)."""
    return (card_seen or {}).get("city") or (spec.get("weather") or {}).get("city")


def span_buckets(start: datetime, end: datetime) -> list[tuple[str, int]]:
    """(D21 보완) 카드 [시작, 종료)와 겹치는 정시 구간 (날짜, 시). 운영 코드 schedule_agent.span_hours와 같은 기준(자정을 넘기면 다음 날 0시 포함)"""
    hours = [start] + [h for h in (start.replace(minute=0, second=0, microsecond=0) + timedelta(hours=n) for n in range(1, 49)) if h < end]
    return [(h.date().isoformat(), h.hour) for h in hours]


def bucket_calls(obs: dict[str, Any], day: str, hour: int, city: str | None = None) -> list[dict[str, Any]]:
    """그 정시 구간의 날씨 호출 기록 (조회 시각의 '시'가 같으면 같은 구간: 14:00·14:30 조회는 모두 14시 구간)"""
    found = []
    for call in obs.get("weather_calls", []):
        kwargs = call.get("kwargs") or {}
        clock = str(kwargs.get("outing_time") or "")
        if kwargs.get("outing_date") == day and clock[:2] == f"{hour:02d}" and (city is None or kwargs.get("city") == city):
            found.append(call)
    return found


def card_span(card_seen: dict[str, Any] | None) -> tuple[datetime, datetime] | None:
    """카드의 시작·종료 시각 (종료가 없으면 기존처럼 시작 구간 하나만 본다)"""
    start = ss.parse_iso((card_seen or {}).get("start"))
    if start is None:
        return None
    end = ss.parse_iso((card_seen or {}).get("end"))
    return start, end if end is not None and end > start else start + timedelta(minutes=1)


def rain_at(obs: dict[str, Any], card_seen: dict[str, Any] | None, spec: dict[str, Any]) -> tuple[str, str, list[str]]:
    """(D14·D15·D21 보완) 카드 [시작, 종료)와 겹치는 모든 정시 구간에서, 날짜·시·지역이 맞는 예보로 '비 신호 없음'이 확인됐는지.
    (상태, 이유, 경고) 날씨 조건 판정과 승인 전 검사가 같이 쓴다. 한 구간이라도 기록이 없거나, 확인할 수 없는 예보가 있거나,
    기록끼리 상충하거나, 비가 있으면 FAIL(고르지 않음). 운영 코드(schedule_agent.fetch_span)와 같은 기준이다."""
    span = card_span(card_seen)
    if span is None:
        return NOT_RUN, "카드 없음", []
    city = forecast_city(card_seen, spec)
    if not city:
        return FAIL, "예보를 확인할 지역을 정할 수 없음(미검증)", []
    warnings: list[str] = []
    checked = []
    for day, hour in span_buckets(*span):
        hits = bucket_calls(obs, day, hour, city)
        others = [c for c in bucket_calls(obs, day, hour) if c not in hits]
        if others:
            warnings.append(f"{day} {hour}시 구간의 다른 지역 예보 {len(others)}건은 판정에 쓰지 않음")
        if not hits:
            return FAIL, f"{city} {day} {hour}시 구간 예보 기록 없음(일정 전체 구간 미확인)", warnings
        # (D25 보완) 응답이 요청과 다른 지역·시각·구간이거나 계약 형식이 아니면 그 기록으로 '비 없음'을 확인할 수 없다.
        broken = [p for c in hits for p in weather_contract_problems(c)]
        if broken:
            return FAIL, f"{city} {day} {hour}시 구간 응답이 요청과 맞지 않음(원인 미확정): {'; '.join(broken)[:160]}", warnings
        states = [rain_signal(c.get("result")) for c in hits]
        if None in states:
            return FAIL, f"{city} {day} {hour}시 구간 확인할 수 없는 예보 {states.count(None)}건(누락·형식·조회 실패는 비 없음으로 보지 않음)", warnings
        if len(set(states)) > 1:
            return FAIL, f"{city} {day} {hour}시 구간 예보 기록끼리 상충 {states}", warnings
        if states[0]:
            return FAIL, f"{city} {day} {hour}시 구간 비 신호 있음", warnings
        checked.append(f"{hour}시")
    return PASS, f"{city} 일정 전체 구간({', '.join(checked)}) 비 신호 없음 확인", warnings


def calls_at(obs: dict[str, Any], day: str, clock: str | None = None, city: str | None = None) -> list[dict[str, Any]]:
    """기록된 날씨 호출 중 날짜·시각·지역이 맞는 것"""
    found = []
    for call in obs.get("weather_calls", []):
        kwargs = call.get("kwargs") or {}
        if kwargs.get("outing_date") != day or (clock and kwargs.get("outing_time") != clock):
            continue
        if city and kwargs.get("city") != city:
            continue
        found.append(call)
    return found


def judge_card(card_seen: dict[str, Any] | None, want: dict[str, Any], original_doc: dict[str, Any], now: datetime) -> list[str]:
    """기대 카드와 다른 점: 종류·대상·날짜(day)·시각(start)·소요 시간·지역, 그리고 저장된 일정과 시각 외 항목(제목·지역) 비교"""
    if not card_seen:
        return ["카드 없음"]
    problems = check_card(card_seen, {"type": want["type"], "schedule_id": want.get("schedule_id"), "start": want.get("start"),
                                      "city": want.get("city"), "title": want.get("title")})
    start, end = ss.parse_iso(card_seen.get("start")), ss.parse_iso(card_seen.get("end"))
    if start is None or end is None:
        return problems + ["카드 시각 형식 오류"]
    if "day" in want and start.date() != now.date() + timedelta(days=want["day"]):
        problems.append(f"카드 날짜 {start.date()} (기대 {now.date() + timedelta(days=want['day'])})")
    if want.get("minutes") and end - start != timedelta(minutes=want["minutes"]):
        problems.append(f"소요 시간 {int((end - start).total_seconds() // 60)}분 (기대 {want['minutes']}분)")
    stored = next((s for s in original_doc["schedules"] if s["schedule_id"] == card_seen.get("schedule_id")), None)
    if stored is not None:
        for key in ("title", "city"):
            if card_seen.get(key) != stored.get(key):
                problems.append(f"시각 외 항목 {key}가 바뀜: {stored.get(key)!r} → {card_seen.get(key)!r}")
    return problems


NETWORK_ERROR_TYPES = ("URLError", "TimeoutError", "timeout", "ConnectionError", "ConnectionRefusedError", "ConnectionResetError",
                       "ConnectionAbortedError", "RemoteDisconnected", "gaierror", "NetworkBlocked")
WEATHER_FAILURE_WORDS = {"env": "연결·외부 API 오류(환경)", "contract": "요청·응답 계약 불일치(원인 미확정)",
                         "range": "예보 범위 밖", "input": "입력 부족·모호",
                         "data": "외부 응답 형식 오류(원인 미확정)", "unknown": "원인을 알 수 없는 오류(원인 미확정)"}


SERVICE_HTTP_RE = re.compile(r"HTTP (?:429|5[0-9]{2})")      # 외부 서비스 쪽 문제라는 근거: 호출 한도 초과·서버 오류
NONNEGATIVE_WEATHER = ("wind_speed_mps", "precipitation_mm", "precipitation_probability_pct")


def weather_contract_problems(call: dict[str, Any]) -> list[str]:
    """(D25 보완) 성공(ok·partial) 응답이 요청과 같은 예보인지, 운영 od.read_weather와 같은 1단계 계약을 측정 도구가 독립적으로 본다.
    지역, data_kind, 출처, outing_at=요청 날짜·시각, hour_start·hour_end=그 정시 구간, +09:00(KST), missing_fields·status 일관성,
    수치 형식. 문제가 없으면 빈 목록. (측정 도구는 1단계 원래 응답을 기록하므로 운영 코드의 검증 결과를 그대로 믿지 않는다)"""
    result = call.get("result") or {}
    if result.get("status") not in ("ok", "partial"):
        return []                      # 오류 상태는 weather_failure가 따로 분류한다
    kwargs, weather = call.get("kwargs") or {}, result.get("weather")
    if not isinstance(weather, dict):
        return ["성공 응답에 예보 자료가 없음"]
    # 1. 요청 시각과 응답의 출발 시각·정시 구간·시간대
    problems = []
    try:
        at = datetime.fromisoformat(f"{kwargs.get('outing_date')}T{kwargs.get('outing_time')}").replace(tzinfo=KST)
        start, end = od.departure_hour_window(at)
        stamps = [datetime.fromisoformat(str(weather.get(key))) for key in ("outing_at", "hour_start", "hour_end")]
    except (TypeError, ValueError):
        return ["요청 또는 응답 시각 형식 오류"]
    if any(stamp.utcoffset() != timedelta(hours=9) for stamp in stamps):
        problems.append("응답 시각이 한국 시각(+09:00)이 아님")
    if stamps[0] != at:
        problems.append(f"outing_at {weather.get('outing_at')} (요청 {at.isoformat(timespec='minutes')})")
    if stamps[1:] != [start, end]:
        problems.append(f"정시 구간 {weather.get('hour_start')}~{weather.get('hour_end')} (요청 구간 {start:%H:%M}~{end:%H:%M})")
    # 2. 지역·종류·출처
    if weather.get("city") != kwargs.get("city"):
        problems.append(f"지역 {weather.get('city')!r} (요청 {kwargs.get('city')!r})")
    if weather.get("data_kind") != "forecast":
        problems.append(f"data_kind {weather.get('data_kind')!r}")
    if weather.get("source") != WEATHER_SOURCE:
        problems.append("출처 표기가 계약과 다름")
    # 3. 필수 항목 키·누락 항목 목록·상태·수치 형식. (D28 보완) 키가 있고 값이 None인 것(정상 partial)과 키 자체가 없는 것(계약 위반)을 나눈다.
    absent = [key for key in od.FIELDS if key not in weather]
    if absent:
        problems.append(f"필수 예보 항목 키 없음 {absent}")
    missing = [key for key in od.FIELDS if key in weather and weather[key] is None]
    listed = weather.get("missing_fields")
    if not isinstance(listed, list) or sorted(str(x) for x in listed) != sorted(missing) or len(missing) == len(od.FIELDS):
        problems.append("missing_fields가 실제 누락 항목과 다름")
    if result.get("status") != ("partial" if missing else "ok"):
        problems.append(f"status {result.get('status')} (누락 {len(missing)}개와 맞지 않음)")
    for key in od.FIELDS:
        value = weather.get(key)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)
                                  or key in NONNEGATIVE_WEATHER and value < 0
                                  or key == "precipitation_probability_pct" and value > 100):
            problems.append(f"{key} 값 형식 오류")
    return problems


def weather_failure(call: dict[str, Any]) -> str | None:
    """(D19 보완) 날씨 호출 1건의 실패 종류. None(ok·partial), env(network_error·연결 예외·외부 서비스 오류 근거가 있는 api_error),
    range(out_of_range), input(needs_input), data(data_unavailable: 외부 응답 형식이지만 프로그램 원인일 수도 있어 환경 오류로 합치지 않음),
    unknown(그 밖). (5-6 최종 보완) api_error는 1단계 메시지에 HTTP 429·5xx가 있을 때만 env이고,
    HTTP 4xx·'요청을 거절'·메시지 없음은 요청 인자(프로그램) 문제일 수 있어 unknown(원인 미확정)이다."""
    error = call.get("error")
    if error:
        return "env" if error.get("type") in NETWORK_ERROR_TYPES else "unknown"
    result = call.get("result") or {}
    status = result.get("status")
    if status in ("ok", "partial") and weather_contract_problems(call):
        return "contract"        # (D25) 응답이 요청과 다르거나 계약 형식이 아님: 환경 오류가 아니라 원인 미확정
    if status == "api_error":
        return "env" if SERVICE_HTTP_RE.search(str(result.get("message") or "")) else "unknown"
    kinds: dict[str, str | None] = {"ok": None, "partial": None, "network_error": "env", "out_of_range": "range",
                                    "needs_input": "input", "data_unavailable": "data"}
    return kinds.get(str(status), "unknown")


def required_weather_calls(obs: dict[str, Any], spec: dict[str, Any], card_seen: dict[str, Any] | None,
                           now: datetime) -> tuple[list[dict[str, Any]], str] | None:
    """흐름의 필수 예보 호출(카드가 있으면 카드 전체 구간·지역, 없으면 시나리오가 정한 지역·날짜)과 설명. 예보 대상이 아니면 None
    (D21 보완) 카드 구간 중 기록이 하나도 없는 정시 구간이 있으면 빈 목록을 돌려 '필수 예보 호출 없음'으로 판정되게 한다."""
    want = spec.get("weather") or {}
    span = card_span(card_seen)
    if span is not None:
        city = forecast_city(card_seen, spec)
        start = span[0]
        groups = [(f"{hour}시", bucket_calls(obs, day, hour, city)) for day, hour in span_buckets(*span)]
        missing = [name for name, calls in groups if not calls]
        label = f"카드 구간 {start:%m-%d %H:%M}~{span[1]:%H:%M} {city}"
        if missing:
            return [], f"{label}, {', '.join(missing)} 구간 기록 없음"
        return [c for _, calls in groups for c in calls], label
    if want:
        day = (now.date() + timedelta(days=want.get("day", 0))).isoformat()
        if want.get("hour") is not None:
            # (D29 보완) 시각을 말한 질문은 그 정시 구간의 조회 기록이 있어야 한다(같은 날 다른 시각 조회로 통과하지 않음).
            label = f"요청 지역·날짜·시각 {want.get('city')} +{want.get('day', 0)}일 {want['hour']}시 구간"
            return bucket_calls(obs, day, int(want["hour"]), want.get("city")), label
        return calls_at(obs, day, None, want.get("city")), f"요청 지역·날짜 {want.get('city')} +{want.get('day', 0)}일"
    return None


def judge_weather(obs: dict[str, Any], spec: dict[str, Any], card_seen: dict[str, Any] | None, now: datetime,
                  real_weather: bool) -> tuple[str, str, list[str]]:
    """필수 예보 호출이 있고 성공했는지, 보조 호출 실패를 숨기지 않았는지. (상태, 이유, 경고)
    (D19 보완) 필수 호출의 연결·외부 API 오류만 ENV_ERROR이고, 예보 범위 밖·입력·외부 응답 형식·원인 모를 오류는 FAIL(사유 표시)"""
    if not real_weather:
        return NA, "가상 예보 (실제 Open-Meteo 검증 대상 아님)", []
    found = required_weather_calls(obs, spec, card_seen, now)
    if found is None:
        return (NA, "이 흐름은 예보 조회 대상이 아님", []) if not obs.get("weather_calls") else (PASS, "보조 조회만 있음", [])
    required, label = found
    if not required:
        return FAIL, f"필수 예보 호출 없음({label})", []
    kinds = [weather_failure(c) for c in required]
    if "env" in kinds:
        return ENV_ERROR, f"필수 예보 호출 환경 오류({label}: {WEATHER_FAILURE_WORDS['env']})", []
    if any(kinds):
        return FAIL, f"필수 예보 호출 실패({label}: {', '.join(sorted({WEATHER_FAILURE_WORDS[k] for k in kinds if k}))})", []
    warnings: list[str] = []
    summaries = [w for t in obs.get("turns", []) for w in (t.get("s_weather") or [])]
    for call in obs.get("weather_calls", []):
        if call in required:
            continue
        kwargs = call.get("kwargs") or {}
        if (call.get("result") or {}).get("status") in ("ok", "partial"):
            if weather_contract_problems(call):       # (D25) 보조 호출의 응답 불일치도 숨기지 않는다
                warnings.append(f"보조 예보 응답이 요청과 맞지 않음({kwargs.get('outing_date')} {kwargs.get('outing_time')})")
            continue
        at = f"{kwargs.get('outing_date')}T{kwargs.get('outing_time')}"
        reported = [w.get("status") for w in summaries if str(w.get("at") or "").startswith(at)]
        if any(r != "unknown" for r in reported):
            return FAIL, f"보조 예보 호출 실패({at})를 {reported}로 보고함 (오류 은폐)", warnings
        warnings.append(f"보조 예보 호출 실패({at}) → {'unknown으로 처리' if reported else '사용 여부 기록 없음'}")
    return PASS, f"필수 예보 호출 성공({label})", warnings


def judge_device(selection: dict[str, Any] | None, devices: list[dict[str, Any]]) -> tuple[str, str]:
    """RAG를 실행한 턴의 실제 임베딩 장치를 장치 선택 결과와 비교한다.
    requested는 장치 종류(selected), weights·token_input·token_output·embedding은 장치 번호까지(device), 차원 384"""
    if not devices:
        return NOT_RUN, "RAG를 실행한 턴 없음"
    if not selection or not selection.get("device") or not selection.get("selected"):
        return FAIL, "장치 선택 기록 없음"
    problems = []
    for used in devices:
        if used.get("requested") != selection["selected"]:
            problems.append(f"requested {used.get('requested')} (선택 {selection['selected']})")
        for key in ("weights", "token_input", "token_output", "embedding"):
            if used.get(key) != selection["device"]:
                problems.append(f"{key} {used.get(key)} (선택 {selection['device']})")
        if used.get("dimensions") != 384:
            problems.append(f"dimensions {used.get('dimensions')}")
    return (FAIL, "; ".join(problems)) if problems else (PASS, f"실제 장치 {selection['device']}, 384차원")


# (D17 보완) LLM 출력 규격 검사: 운영 코드가 호출마다 보낸 스키마와 단계별 운영 검증 함수를 그대로 쓴다. (패키지 설치 없음)
SCHEMA_RULES = {"type", "enum", "properties", "required", "additionalProperties", "items", "maxItems", "minItems",
                "maxLength", "minLength", "minimum", "maximum", "description", "title"}
JSON_TYPES: dict[str, Any] = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int,
                              "number": (int, float), "null": type(None)}
STAGE_VALIDATORS: dict[str, Callable[[Any], Any]] = {
    "understanding": od.validate_understanding,
    "schedule_understanding": lambda data: sa.validate_schedule_understanding(data, []),
}
SCHEMA_ONLY_STAGES = ("answer", "schedule_choice")    # 운영 코드도 스키마(허용 후보 id 포함)로 검사하고, 답변 내용 검증은 따로 기록


def schema_problems(data: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """JSON 스키마의 자료형·필수 키·추가 키·허용값·목록·길이 규칙만 확인하는 최소 검사기. 모르는 규칙이 있으면 통과로 보지 않는다."""
    unknown = sorted(set(schema) - SCHEMA_RULES)
    if unknown:
        return [f"{path} 검사하지 못한 스키마 규칙 {unknown}"]
    kinds = schema.get("type")
    allowed = kinds if isinstance(kinds, list) else ([kinds] if kinds else [])

    def fits(kind: str) -> bool:
        return isinstance(data, JSON_TYPES[kind]) and not (kind in ("integer", "number") and isinstance(data, bool))
    if allowed and not any(fits(k) for k in allowed if k in JSON_TYPES):
        return [f"{path} 자료형 {type(data).__name__} (기대 {allowed})"]
    problems = []
    if "enum" in schema and data not in schema["enum"]:
        problems.append(f"{path} 허용값 아님: {data!r}"[:120])
    if isinstance(data, str) and not schema.get("minLength", 0) <= len(data) <= schema.get("maxLength", len(data)):
        problems.append(f"{path} 길이 {len(data)}")
    if isinstance(data, (int, float)) and not isinstance(data, bool) and \
            not schema.get("minimum", data) <= data <= schema.get("maximum", data):
        problems.append(f"{path} 범위 밖 {data}")
    if isinstance(data, dict):
        properties = schema.get("properties") or {}
        missing = [k for k in schema.get("required", []) if k not in data]
        extra = [k for k in data if k not in properties] if schema.get("additionalProperties") is False else []
        if missing:
            problems.append(f"{path} 필수 키 없음 {missing}")
        if extra:
            problems.append(f"{path} 규격에 없는 키 {extra}")
        for key, value in data.items():
            if key in properties:
                problems += schema_problems(value, properties[key], f"{path}.{key}")
    if isinstance(data, list):
        if not schema.get("minItems", 0) <= len(data) <= schema.get("maxItems", len(data)):
            problems.append(f"{path} 항목 수 {len(data)}")
        if isinstance(schema.get("items"), dict):
            for number, value in enumerate(data):
                problems += schema_problems(value, schema["items"], f"{path}[{number}]")
    return problems


def format_problems(stage: str, raw: Any, schema: Any) -> list[str]:
    """LLM 출력 1건의 규격 문제 목록(빈 목록이면 통과). JSON 문법(운영 parse_model_json) → JSON 객체 → 운영 스키마 → 단계별 운영 검증 함수.
    검증 기준이 없는 단계는 통과로 보지 않는다."""
    try:
        data = od.parse_model_json(raw)
    except (TypeError, ValueError) as exc:
        return [f"JSON 문법 오류: {exc}"[:120]]
    if not isinstance(data, dict):
        return ["JSON 객체가 아님"]
    problems = schema_problems(data, schema) if isinstance(schema, dict) and schema else ["운영 스키마 기록 없음"]
    validator = STAGE_VALIDATORS.get(stage)
    if validator is not None:
        try:
            validator(copy.deepcopy(data))
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"운영 검증 함수 거부: {exc}"[:160])
    elif stage not in SCHEMA_ONLY_STAGES:
        problems.append(f"검증 기준이 없는 단계 {stage}")
    return problems


LLM_ENV_CODES = ("connection_error", "connection_refused", "host_not_found", "model_missing", "configuration")


def llm_error_kind(call: dict[str, Any]) -> str | None:
    """(D33 보완) LLM 호출 1건의 오류 종류. env: 3단계 llm_unavailable 중 연결·시간 초과·호스트 없음·모델 미설치·설정·HTTP 5xx,
    또는 연결 예외. invalid: llm_invalid_output(응답 형식). unknown: HTTP 4xx·KeyError 같은 원인 미확정 예외. 오류가 없으면 None"""
    error = call.get("error")
    if not error:
        return None
    status, code, kind = error.get("status"), str(error.get("code") or ""), error.get("type")
    if status == "llm_unavailable" and (code in LLM_ENV_CODES or code.endswith("_timeout") or re.fullmatch(r"http_5[0-9]{2}", code)):
        return "env"
    if kind in NETWORK_ERROR_TYPES:
        return "env"
    return "invalid" if status == "llm_invalid_output" else "unknown"


def judge_llm_calls(llm_calls: list[dict[str, Any]]) -> tuple[str, str]:
    """(D33 보완) qwen_call 판정: 호출 없음 NOT_RUN, 오류 없음 PASS, 환경 근거가 있는 오류만 있으면 ENV_ERROR,
    응답 형식 오류·원인 미확정 오류가 하나라도 있으면 FAIL(환경 오류가 함께 있어도 사유에 따로 남긴다). 형식(qwen_format)·판단(llm_judgment)은 따로 본다."""
    if not llm_calls:
        return NOT_RUN, "LLM 호출 없음"
    kinds = [llm_error_kind(c) for c in llm_calls]
    errors = [(k, c["error"]) for k, c in zip(kinds, llm_calls) if k]
    if not errors:
        return PASS, f"LLM 호출 {len(llm_calls)}건 모두 응답"
    words = {"env": "환경 오류", "invalid": "응답 형식 오류", "unknown": "원인 미확정 오류"}
    detail = "; ".join(f"{words[k]}({e.get('status') or e.get('type')}/{e.get('code') or e.get('type')})" for k, e in errors)[:300]
    if all(k == "env" for k, _ in errors):
        return ENV_ERROR, f"LLM 호출 {len(llm_calls)}건 중 {detail}"
    return FAIL, f"LLM 호출 {len(llm_calls)}건 중 {detail}"


def judge_format(turns: list[dict[str, Any]], llm_calls: list[dict[str, Any]]) -> tuple[str, str, list[str]]:
    """(D17 보완) 턴·단계별로 마지막 출력이 운영 스키마·검증을 통과해야 PASS. 앞 출력의 형식 오류 뒤 재요청 성공은 PASS이지만
    '최초 형식 오류'로 따로 남긴다. 검사 기록이 없는 호출은 검증하지 않은 것이므로 NOT_RUN. 답변 단계의 운영 내용 검증 결과는 경고로 남긴다.
    (D23 보완) 호출한 단계 중 출력 없이 실패한 단계가 있거나(정상 출력 뒤 다른 필수 단계 실패), 검증할 출력이 하나도 없으면(모든 호출 실패)
    전체 형식 검증이 끝나지 않은 것이므로 NOT_RUN이다. 출력의 스키마 준수 수와 단계 검증 완료 수를 사유에 따로 적는다.
    (호출 성공 여부는 qwen_call, 모델 판단은 llm_judgment가 따로 판정한다)"""
    if not llm_calls:
        return NOT_RUN, "LLM 호출 없음", []
    per_turn = [t["llm_calls"] for t in turns if t.get("llm_calls")]
    final_bad, first_bad, retried, unchecked = [], 0, 0, 0
    outputs, valid_outputs, missing_stages, called_stages = 0, 0, [], 0
    for calls in per_turn or [llm_calls]:
        by_stage: dict[str, list[dict[str, Any]]] = {}
        stages_called = list(dict.fromkeys(str(call.get("stage")) for call in calls))
        for call in calls:
            if not call.get("error"):
                by_stage.setdefault(str(call.get("stage")), []).append(call)
                outputs += 1
                valid_outputs += 1 if call.get("format") == [] else 0
        called_stages += len(stages_called)
        missing_stages += [stage for stage in stages_called if stage not in by_stage]     # 출력 없이 호출 실패만 있는 단계
        for stage, sequence in by_stage.items():
            if any("format" not in c for c in sequence):
                unchecked += 1
                continue
            retried += len(sequence) - 1
            if sequence[-1]["format"]:
                final_bad.append(f"{stage}: {'; '.join(sequence[-1]['format'])}"[:160])
            else:
                first_bad += sum(1 for c in sequence[:-1] if c["format"])
    warnings = [f"최초 형식 오류 {first_bad}건 → 재요청 성공"] if first_bad else []
    for turn in turns:
        checks = turn.get("a_verification") or []
        if checks and not checks[-1].get("passed"):
            warnings.append(f"답변 단계 운영 검증 실패(대체 답변 사용): {turn.get('text')}"[:120])
        elif len(checks) > 1:
            warnings.append(f"답변 단계 운영 검증 {len(checks) - 1}회 실패 뒤 재요청 통과: {turn.get('text')}"[:120])
    invalid_turns = [t.get("text") for t in turns if t.get("status") == "llm_invalid_output"]
    counts = (f"출력 {outputs}건 중 운영 스키마·검증 통과 {valid_outputs}건 / 호출한 단계 {called_stages}개 중 "
              f"출력 없이 실패 {len(missing_stages)}개")
    if final_bad or invalid_turns:
        return FAIL, f"최종 출력 규격 오류 {final_bad}, 운영 코드가 규격 오류로 끝낸 턴 {invalid_turns} ({counts})", warnings
    if not outputs:
        return NOT_RUN, f"검증할 LLM 출력 없음: 모든 호출 실패({counts}, 호출 실패는 qwen_call에서 판정)", warnings
    if missing_stages:
        return NOT_RUN, (f"필수 단계 {sorted(set(missing_stages))}의 출력이 없어 전체 형식 검증 미완료({counts}, "
                         "호출 실패는 qwen_call에서 판정)"), warnings
    if unchecked:
        return NOT_RUN, f"형식 검사 기록이 없는 단계 {unchecked}개(미검증, {counts})", warnings
    return PASS, f"모든 단계의 최종 출력이 운영 스키마·검증 통과(재요청 {retried}건, {counts})", warnings


# (D18 보완) RAG 검색 결과: 필수 검색(기대 주제의 공식 질문으로 시작)과 보조 검색을 나눈다.
def rag_topic(call: dict[str, Any]) -> str | None:
    """검색 질의가 3단계 공식 질문(od.QUESTIONS)으로 시작하면 그 주제. (D26 보완) 공식 질문이 아닌 질의는 주제 낱말이
    하나만 들어 있을 때만 그 주제로 본다(운영 코드는 항상 공식 질문으로 검색한다)."""
    query = (call.get("args") or [None])[0]
    if not isinstance(query, str):
        return None
    official = next((t for t, q in od.QUESTIONS.items() if query.startswith(q)), None)
    if official:
        return official
    words = [t for t in od.QUESTIONS if t in query]
    return words[0] if len(words) == 1 else None


def split_rag_calls(calls: list[dict[str, Any]], topics: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(필수, 보조). 기대 주제를 정할 수 없거나 질의로 주제를 알 수 있는 호출이 하나도 없으면 모든 호출을 필수로 본다."""
    known = [rag_topic(c) for c in calls]
    if not topics or not any(known):
        return list(calls), []
    required = [c for c, topic in zip(calls, known) if topic in topics]
    return required, [c for c, topic in zip(calls, known) if topic not in topics]


# (5-6 최종 보완) RAG에서 환경 문제라는 근거가 있는 예외: 연결·파일(모델 캐시)·메모리 부족. search_error는 임베딩·검색 예외를
# 감싼 상태라 원인을 알 수 없으므로 환경 오류로 합치지 않는다(원인 미확정).
RAG_ENV_ERROR_TYPES = NETWORK_ERROR_TYPES + ("OSError", "FileNotFoundError", "PermissionError", "MemoryError", "OutOfMemoryError")


def rag_outcome(call: dict[str, Any]) -> str:
    """검색 1건: ok(결과 있음), env_error(연결·파일·메모리 예외), error(그 밖 예외·search_error: 원인 미확정),
    insufficient(근거 부족), invalid(그 밖)"""
    result = call.get("result") or {}
    error = call.get("error")
    if error:
        return "env_error" if error.get("type") in RAG_ENV_ERROR_TYPES else "error"
    if result.get("status") == "search_error":
        return "error"
    if result.get("status") == "ok" and result.get("results"):
        return "ok"
    return "insufficient" if result.get("status") == "insufficient_evidence" else "invalid"


def judge_rag_calls(required: list[dict[str, Any]], topics: set[str] | None = None) -> tuple[str, str]:
    """필수 검색이 없으면 FAIL, 환경 근거가 있는 오류면 ENV_ERROR, 원인 모를 오류(search_error·일반 예외)는 FAIL(원인 미확정),
    근거 부족·형식 오류면 FAIL, 모두 정상일 때만 PASS.
    (D26 보완) topics(필수 주제)를 주면 주제마다 그 주제의 검색이 1건 이상 정상이어야 한다. 빠진 주제가 있으면 FAIL"""
    if not required:
        return FAIL, "필수 검색 없음(기대 주제의 공식 질문으로 검색하지 않음)"
    outcomes = [rag_outcome(c) for c in required]
    if "env_error" in outcomes:
        return ENV_ERROR, f"필수 검색 환경 오류 {outcomes.count('env_error')}건 / {len(outcomes)}건"
    if "error" in outcomes:
        return FAIL, f"필수 검색 오류 {outcomes.count('error')}건(원인 미확정: 환경 문제라는 근거 없음) / {len(outcomes)}건"
    if any(o != "ok" for o in outcomes):
        return FAIL, f"필수 검색 결과 {outcomes}"
    missing = sorted((topics or set()) - {rag_topic(c) for c in required})
    if missing:
        return FAIL, f"필수 주제 {missing} 검색 없음 (검색한 주제 {sorted({str(rag_topic(c)) for c in required})})"
    return PASS, f"필수 검색 {len(outcomes)}건 정상" + (f" (주제 {sorted(topics or set())} 모두 확인)" if topics else "")


def judge_components(obs: dict[str, Any], spec: dict[str, Any], *, real_weather: bool, real_rag: bool, real_llm: bool,
                     selection: dict[str, Any] | None = None) -> dict[str, Any]:
    """구성 요소별 {status, reasons, warnings}. real_*=False인 실제 연동 항목은 N/A(가상)이다."""
    turns = obs.get("turns", [])
    first = turns[0] if turns else {}
    card_seen = next((t.get("card") for t in turns if t.get("card")), None)
    now = datetime.fromisoformat(obs["now"]) if obs.get("now") else FIXED_NOW
    status: dict[str, str] = {}
    reasons: dict[str, str] = {}
    # 1. 실제 예보 호출과 날씨 조건
    status["weather_real"], reasons["weather_real"], warnings = judge_weather(obs, spec, card_seen, now, real_weather)
    if spec.get("avoid_rain"):
        # (D14·D15 보완) 카드 날짜·시각·지역이 모두 맞는 예보로, 확인할 수 있는 '비 없음'만 PASS (승인 전 검사와 같은 함수)
        status["weather_condition"], reasons["weather_condition"], extra = rain_at(obs, card_seen, spec)
        warnings = warnings + extra
    else:
        status["weather_condition"] = NA
    # 2. LLM: 호출 성공 / 출력 규격 / 의미 판단을 나눈다.
    llm_calls = obs.get("llm_calls", [])
    if not real_llm:
        status["qwen_call"] = NA
    else:
        status["qwen_call"], reasons["qwen_call"] = judge_llm_calls(llm_calls)
    status["qwen_format"], reasons["qwen_format"], extra = judge_format(turns, llm_calls)
    warnings = warnings + extra
    if spec.get("understanding"):
        problems = matches(first.get("s_understanding"), spec["understanding"])
        status["llm_judgment"] = FAIL if problems else PASS
        reasons["llm_judgment"] = "; ".join(problems)
    elif spec.get("route"):
        route = first.get("a_route")
        status["llm_judgment"] = PASS if route in spec["route"] else FAIL
        reasons["llm_judgment"] = f"3단계 경로 {route} (기대 {spec['route']})"
    else:
        status["llm_judgment"] = NA
    # 3. 승인 카드: 기대 카드와 내용 비교. 조건에 맞는 후보가 없다는 정상 안내는 NOT_RUN(정상)
    if spec.get("card"):
        if card_seen:
            problems = judge_card(card_seen, spec["card"], obs.get("initial_doc") or {"schedules": []}, now)
            status["proposal"] = FAIL if problems else PASS
            reasons["proposal"] = "; ".join(problems)
        else:
            summary = [w.get("status") for t in turns for w in (t.get("s_weather") or [])]
            no_candidate = any("찾지 못해" in (t.get("answer") or "") for t in turns) and summary and \
                all(s in ("signal", "unknown") for s in summary)
            status["proposal"] = NOT_RUN if no_candidate else FAIL
            reasons["proposal"] = "조건에 맞는 후보 없음(정상 안내)" if no_candidate else "기대 카드가 없음"
    else:
        status["proposal"] = NA
    # 4. RAG: 호출 성공 / 출처 일치 / 본문 관련성(수동 검토)
    rag_calls = obs.get("rag_calls", [])
    if not rag_calls:
        for key in ("rag_call", "rag_source", "rag_content"):
            status[key] = NOT_RUN
        reasons["rag_call"] = "RAG 미실행(위험 신호 없음 또는 해당 없음)"
    else:
        # (D26 보완) 기대 검색 주제와 기대 출처 파일을 따로 정한다. 같은 PDF를 쓰는 한파·대설, 호우·강풍도 서로 다른 주제다.
        # 주제: 시나리오의 rag_topics(필수, 모두 검색돼야 함)·rag_optional_topics(검색했다면 성공해야 함)
        # → 시나리오에 없으면 카드 전체 구간의 실제 예보 기록에서 위험 주제를 읽는다(D21, 두 번째 구간만의 강풍도 포함).
        topics = set(spec.get("rag_topics", []))
        optional = set(spec.get("rag_optional_topics", [])) - topics
        if not topics and card_seen:
            span = card_span(card_seen)
            city = forecast_city(card_seen, spec)
            hits = [c for day, hour in span_buckets(*span) for c in bucket_calls(obs, day, hour, city)] if span else []
            topics = {t for h in hits for t in risk_topics(h.get("result")) if t in TOPIC_FILE}
        expected = ({spec["rag_file"]} if spec.get("rag_file") else set()) | {TOPIC_FILE[t] for t in topics}
        # (D18 보완) 필수 검색(기대 주제의 공식 질문)과 보조 검색을 나눈다. 보조 검색 성공으로 필수 검색 실패를 덮지 않는다.
        required, auxiliary = split_rag_calls(rag_calls, topics | optional)
        status["rag_call"], reasons["rag_call"] = judge_rag_calls(required, topics)
        for call in auxiliary:
            if rag_outcome(call) != "ok":
                warnings = warnings + [f"보조 검색 실패({rag_outcome(call)}): {str((call.get('args') or [''])[0])[:40]}"]
        # (D30 보완) 출처는 주제마다 따로 본다: 주제를 아는 검색(필수·검색된 선택 주제)은 그 주제의 공식 PDF(TOPIC_FILE)가 상위 출처여야 한다.
        # 주제를 알 수 없는 질의만 기존처럼 기대 출처 집합(rag_file·기대 주제 파일)과 비교한다(호환). 보조 검색 출처는 판정에 쓰지 않는다.
        tops, wrong, known = [], [], False
        for call in required:
            found = (call.get("result") or {}).get("results")
            if not found:
                continue
            top = found[0].get("filename")
            tops.append(top)
            topic = rag_topic(call)
            known = known or topic in TOPIC_FILE
            want = {TOPIC_FILE[topic]} if topic in TOPIC_FILE else expected
            if want and top not in want:
                wrong.append(f"{topic or '주제 미상'}: {top} (기대 {sorted(want)})")
        if not tops:
            status["rag_source"] = FAIL
        elif not expected and not known:
            status["rag_source"] = REVIEW
        else:
            status["rag_source"] = FAIL if wrong else PASS
        reasons["rag_source"] = (f"주제별 출처 불일치 {wrong}" if wrong else f"상위 출처 {tops}") + f" (기대 출처 {sorted(expected)})"
        status["rag_content"] = REVIEW if status["rag_call"] == PASS else NOT_RUN
        reasons["rag_content"] = "본문 관련성은 자동 확정하지 않음 (로그의 본문 앞부분을 사람이 확인)"
    # 5. 실제 장치 (실제 RAG에서만), 6. LangGraph 승인·JSON 변경
    if real_rag:
        devices = [t["embedding_device"] for t in turns if isinstance(t.get("embedding_device"), dict)]
        status["device"], reasons["device"] = judge_device(selection, devices)
    else:
        status["device"] = NA
    status["graph_decision"] = obs.get("decision_status", NOT_RUN)
    status["json_change"] = obs.get("json_status", NOT_RUN)
    return {"status": status, "reasons": reasons, "warnings": warnings}


def required_components(spec: dict[str, Any], *, real_weather: bool) -> list[str]:
    """종합 흐름으로 설계된 사례의 필수 구성 요소. 가상 예보(통제 통합)는 실제 예보 항목을 필수에서 빼고 따로 표시한다."""
    if not spec.get("integrated"):
        return []
    required = ["qwen_call", "qwen_format", "llm_judgment", "proposal", "graph_decision", "json_change"]
    if real_weather:
        required.append("weather_real")
    if spec.get("avoid_rain"):
        required.append("weather_condition")
    if spec.get("rag_required"):
        required += ["rag_call", "rag_source", "rag_content", "device"]
    return required


def integrated_success(status: dict[str, str], required: list[str]) -> str:
    """종합 흐름 판정: 필수 구성 요소가 모두 PASS면 PASS. 본문 관련성만 REVIEW면 REVIEW(사람 확인 전에는 성공 아님).
    필수 요소의 N/A·NOT_RUN은 성공으로 바꾸지 않는다(NOT_RUN). 설계상 종합 흐름이 아니면 N/A"""
    if not required:
        return NA
    values = [status.get(key, NOT_RUN) for key in required]
    for outcome in (ENV_ERROR, FAIL):
        if outcome in values:
            return outcome
    if any(v in (NOT_RUN, NA) for v in values):
        return NOT_RUN
    return REVIEW if REVIEW in values else PASS


def run_data_flow(spec: dict[str, Any], llm: Any, *, weather_fn: Callable, rag_fn: Callable, now: datetime, device: str,
                  index: Any, real_weather: bool, real_rag: bool, real_llm: bool, selection: dict[str, Any] | None = None,
                  live: bool = False, expect_ollama: dict[str, Any] | None = None) -> dict[str, Any]:
    """4.8·4.8-C 흐름 1회. 기대 카드와 날씨 조건을 판정 함수로 확인한 뒤에만 승인하고, 저장 결과를 검사한다."""
    obs: dict[str, Any] = {"scenario": spec["id"], "turns": [], "graph_ok": True, "error": None, "s1_extra": [],
                           "now": now.isoformat()}
    weather, rag, tap = TimedCall(weather_fn, "weather"), TimedCall(rag_fn, "rag"), ScheduleTap()
    obs.update(weather_calls=weather.calls, rag_calls=rag.calls)     # 승인 전 카드 판정에서 바로 쓰도록 같은 목록을 연결
    ops_before = operational_state()
    notes: list[str] = []
    approved_ok = False
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name)
        path = folder / "schedule_store.json"
        env = safe_env(folder, live=live, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        with mock.patch.dict(os.environ, env, clear=True):
            store = ss.ScheduleStore(path, clock=lambda: now)
            store.ensure_initialized()
        original_doc, original = store.load(), path.read_bytes()
        obs["initial_doc"] = original_doc
        with tod.Harness(llm=llm, weather=weather, city="real" if real_weather else None, rag=rag, env=env, now=now):
            obs["env_reason"] = ollama_mismatch(expect_ollama)
            try:
                if obs["env_reason"]:
                    raise StopRun
                graph = sg.build_graph(schedule_store=store, a_device=device, a_index=index, schedule_runner=tap)
                obs["runtime"] = sg.runtime_report(graph)
                thread = sg.new_thread_id()
                for number, text in enumerate(spec["turns"], start=1):
                    obs["turns"].append(capture_turn(graph, thread, text, llm, tap, weather, rag))
                    if path.read_bytes() != original:
                        obs["changed_at_turn"] = number
                        raise StopRun
                seen = obs["turns"][-1]["card"]
                if spec.get("approve_valid_card") and seen:
                    problems = judge_card(seen, spec["card"], original_doc, now)
                    if spec.get("avoid_rain"):
                        # (D14·D15 보완) 날씨 조건 판정과 같은 함수로 확인한다(날짜·시각·지역 일치, 확인 불가·상충은 승인 안 함).
                        state, why, _ = rain_at(obs, seen, spec)
                        if state != PASS:
                            problems.append(f"날씨 조건 확인 실패: {why}")
                    notes += problems
                    if problems:
                        obs["decision_status"], obs["json_status"] = NOT_RUN, NOT_RUN
                        notes.append("유효하지 않은 카드라 승인하지 않음")
                    else:
                        pressed = press(graph, thread, "approve", seen["action_id"])
                        execution = graph_values(graph, thread).get("execution_result")
                        obs["turns"].append({"text": "[approve]", "answer": pressed.get("answer"), "seconds": 0, "llm_calls": [],
                                             "llm_error_statuses": [], "execution": execution})
                        if fake_success(pressed.get("answer"), execution):
                            obs["s1_extra"].append("가짜 성공(applied=False인데 저장 성공 문장)")
                        doc = store.load()
                        violations = approval_violations(original_doc, doc, seen) if path.read_bytes() != original else ["저장 안 됨"]
                        applied = isinstance(execution, dict) and execution.get("applied") is True
                        obs["decision_status"] = PASS if (pressed.get("pending") is None and applied) else FAIL
                        obs["json_status"] = FAIL if violations else PASS
                        notes += [f"저장: {v}" for v in violations]
                        approved_ok = not violations
                        if violations and path.read_bytes() != original:
                            obs["s1_extra"].append("승인 뒤 저장 내용이 카드와 다름: " + "; ".join(violations))
            except StopRun:
                pass
            except GraphFailure as exc:
                obs["graph_ok"] = False
                obs["error"] = str(exc)
        final = path.read_bytes()
        obs.update(before_sha=sha256_bytes(original), after_sha=sha256_bytes(final), changed=final != original,
                   unused_created=(folder / UNUSED_STORE_NAME).exists(), final_doc=json.loads(final.decode("utf-8")))
    obs["ops_changed"] = operational_state() != ops_before
    obs.update(llm_calls=copy.deepcopy(getattr(llm, "calls", [])), selection=selection)
    judged = judge_components(obs, spec, real_weather=real_weather, real_rag=real_rag, real_llm=real_llm, selection=selection)
    components = judged["status"]
    obs["rag_review"] = [{"query": (c.get("args") or [None])[0], "topic": rag_topic(c), "outcome": rag_outcome(c),
                          "error": c.get("error"),
                          "results": [{"filename": r.get("filename"), "pdf_page": r.get("pdf_page"),
                                       "similarity": r.get("similarity"), "content": (r.get("content") or "")[:160]}
                                      for r in ((c.get("result") or {}).get("results") or [])]} for c in rag.calls]
    ind = {k: NA for k in INDICATORS}
    ind["graph_ok"] = PASS if obs["graph_ok"] else FAIL
    ind["judgment_ok"] = components["llm_judgment"]
    ind["proposal_ok"] = components["proposal"]
    if "decision_status" in obs:
        ind["decision_ok"], ind["saved_ok"] = obs["decision_status"], obs["json_status"]
    external = [components[k] for k in ("weather_real", "rag_call") if components[k] != NA]
    ind["external_ok"] = NA if not external else next((o for o in (ENV_ERROR, FAIL, NOT_RUN) if o in external), PASS)
    ind["device_ok"] = components["device"]
    if components["device"] == FAIL:
        obs["s1_extra"].append(f"장치 오보고(RAG 장치가 선택 장치와 다름): {judged['reasons'].get('device')}")
    s1 = s1_violations(obs, approved_change_ok=approved_ok)
    # (D19 보완) 실제 예보 묶음에서는 필수 예보 호출의 연결·외부 API 오류도 회차 환경 오류(연속 중단 조건에 반영)
    card_seen = next((t.get("card") for t in obs["turns"] if t.get("card")), None)
    required_weather = (required_weather_calls(obs, spec, card_seen, now) or ([], ""))[0] if real_weather else None
    env = env_error_in(obs, required_weather=required_weather)
    if env and not obs["turns"]:
        ind = dict(ind, judgment_ok=ENV_ERROR, proposal_ok=ENV_ERROR)
    hint = cause_hint(ind, env)
    integrated = integrated_success(components, required_components(spec, real_weather=real_weather))
    return {"indicators": ind, "components": components, "component_reasons": judged["reasons"],
            "component_warnings": judged["warnings"], "integrated": integrated, "s1": s1, "env_error": env,
            "problems": notes, "cause_hint": hint, "cause": "원인 미확정" if hint else None, "obs": obs}


def judge_rag_probe(probe: dict[str, Any], result: dict[str, Any], index_device: dict[str, Any] | None = None,
                    selection: dict[str, Any] | None = None) -> dict[str, Any]:
    """4.8-R 판정: 검색 성공·상위 출처 파일이 주제의 공식 자료와 맞는지·색인 장치. 점수는 기록만 한다(새 임계값 없음).
    본문 관련성은 자동 확정하지 않고 REVIEW로 남긴다."""
    raw_results = result.get("results")
    results: list[dict[str, Any]] = [r for r in raw_results if isinstance(r, dict)] if isinstance(raw_results, list) else []
    expected = TOPIC_FILE[probe["topic"]]
    rows = [{"filename": r.get("filename"), "pdf_page": r.get("pdf_page"), "printed_page": r.get("printed_page"),
             "similarity": r.get("similarity"), "content": (r.get("content") or "")[:160]} for r in results]
    device_status, device_reason = judge_device(selection, [index_device]) if index_device is not None else (NA, "")
    base = {"expected_file": expected, "results": rows, "device": device_status, "device_reason": device_reason,
            "score_meaning": "similarity = 1 − 코사인 거리(0~1). 검색기가 0.30(MIN_SIMILARITY) 이상만 돌려줌"}
    if result.get("status") != "ok" or not rows:
        # (5-6 최종 보완) search_error는 임베딩·검색 예외를 감싼 상태라 환경 문제라는 근거가 없다 → FAIL(원인 미확정).
        # 장치·색인 준비 실패(모델 캐시 없음 등 환경)는 묶음 시작 조건에서 ENV_ERROR로 따로 기록된다(D8).
        reason = f"{result.get('status')} (원인 미확정)" if result.get("status") == "search_error" else result.get("status")
        return dict(base, status=FAIL, content=NOT_RUN, reason=reason)
    return dict(base, status=PASS if rows[0]["filename"] == expected else FAIL, content=REVIEW,
                all_from_expected=all(r["filename"] == expected for r in rows))


# %% 묶음 실행기 (재측정 사유·동시 실행 잠금·로그·중단 조건·요약)
# 핵심 요약: 묶음 하나를 실행할 때 ① 잠금 파일로 동시 실행을 막고 ② 잠금 안에서 이전 5-6 기록이 있으면 재측정 사유를 요구하며
# ③ 로그를 'x' 모드로 새로 만들어 덮어쓰지 않고 ④ S1·측정 도구 오류·환경 오류 연속·시간 상한에서 멈추고 남은 회차를 NOT_RUN으로 남긴다.
def stats(values: list[float]) -> dict[str, Any]:
    """표본 수·최솟값·중앙값·평균·최댓값 (표본이 없으면 n=0)"""
    if not values:
        return {"n": 0}
    return {"n": len(values), "min": min(values), "median": round(statistics.median(values), 2),
            "mean": round(statistics.mean(values), 2), "max": max(values)}


def summarize(rows: list[dict[str, Any]], planned: int) -> dict[str, Any]:
    """지표별 집계: 예정·실행·평가 가능·성공·실패·미실행·환경 오류·N/A·수동 검토. 성공률은 PASS/(PASS+FAIL), N/A·REVIEW 제외"""
    executed = [r for r in rows if not r.get("not_run")]
    out: dict[str, Any] = {"planned": planned, "executed": len(executed), "not_run_runs": len(rows) - len(executed),
                           "env_error_runs": sum(1 for r in rows if r.get("env_error")),
                           "s1_runs": sum(1 for r in rows if r.get("s1")),
                           "tool_error_runs": sum(1 for r in rows if r.get("tool_error")), "indicators": {}}
    for key in INDICATORS:
        counts = {status: sum(1 for r in rows if r["indicators"].get(key) == status) for status in STATUSES}
        evaluable = counts[PASS] + counts[FAIL]
        out["indicators"][key] = dict(counts, evaluable=evaluable,
                                      rate=round(counts[PASS] / evaluable, 3) if evaluable else None)
    judged = [r for r in rows if "integrated" in r]
    if judged:
        integrated: dict[str, Any] = {status: sum(1 for r in judged if r["integrated"] == status) for status in STATUSES}
        out["integrated"] = integrated
        integrated["note"] = "PASS만 종합 성공. REVIEW는 RAG 본문 관련성 사람 확인 전, NOT_RUN은 필수 구성 요소 미실행"
    return out


def latency(rows: list[dict[str, Any]], cold_first: bool) -> dict[str, Any]:
    """턴 시간·LLM 단계별 시간 통계 (cold_first: 묶음 시작 때 모델이 적재되지 않았으면 첫 LLM 호출을 cold로 분리)
    별도 프로세스 재조회 턴의 호출도 포함한다. (그 턴의 cold·warm은 미측정)"""
    turns, stages, cold = [], {}, []
    first = True
    for row in rows:
        for turn in (row.get("obs") or {}).get("turns", []):
            text = str(turn.get("text", ""))
            if turn.get("seconds") and (not text.startswith("[") or text.startswith("[별도 프로세스]")):
                turns.append(turn["seconds"])
            for call in turn.get("llm_calls", []):
                if first and cold_first:
                    cold.append(call["seconds"])
                else:
                    stages.setdefault(call["stage"], []).append(call["seconds"])
                first = False
    return {"turn_seconds": stats(turns), "llm_stage_seconds": {k: stats(v) for k, v in stages.items()},
            "cold_first_call_seconds": stats(cold)}


def file_identity(info: os.stat_result) -> tuple[int, int] | None:
    """(D20 보완) 파일 식별자(장치, inode·파일 번호). 파일 시스템이 주지 않으면(0) None → token 비교로만 판단"""
    return (info.st_dev, info.st_ino) if info.st_ino else None


class LiveSuite:
    """묶음 실행기. log_dir·clock·live를 바꿔 M층에서 오프라인으로 시험할 수 있다."""

    def __init__(self, name: str, planned: int, *, kinds: dict[str, str], per_run_seconds: float = MAX_OBSERVED_RUN_SECONDS,
                 log_dir: Path = LOG_DIR, live: bool = True, rerun_reason: str | None = None, clock=time.monotonic):
        self.name, self.planned, self.kinds = name, planned, kinds
        self.log_dir, self.live, self.clock = log_dir, live, clock
        self.rerun_reason = rerun_reason
        self.cap = planned * per_run_seconds * TIME_CAP_FACTOR
        self.run_id = f"{datetime.now(KST):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        self.rows: list[dict[str, Any]] = []
        self.lock_path = log_dir / LOCK_NAME
        self.token: str | None = None
        self.lock_id: tuple[int, int] | None = None
        self.txt: Any = None
        self.jsonl: Any = None
        self.stop_reason: str | None = None
        self.header: dict[str, Any] = {}
        self.protected_before: dict[str, str | None] = {}
        self.ops_before: dict[str, str | None] = {}
        self.started = 0.0

    def previous_runs(self) -> list[str]:
        return sorted(p.name for p in self.log_dir.glob(f"5-6_{self.name}_*.jsonl"))

    # 1. 시작: 잠금 → (잠금 안에서) 재측정 사유 확인 → 로그 생성 → 머리줄. 실패하면 내 잠금과 연 파일을 정리하고 다시 낸다.
    def start(self, extra_header: dict[str, Any] | None = None) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise unittest.SkipTest(f"[{self.name}] 다른 5-6 측정이 실행 중입니다({self.lock_path.name}). 동시에 실행하지 않습니다. "
                                    "실행 중인 측정이 없는데 남아 있으면 확인 뒤 사용자가 직접 지워 주세요.")
        self.token = json.dumps({"run_id": self.run_id, "pid": os.getpid(), "suite": self.name})
        try:
            # (D20 보완) 내가 만든 잠금 파일의 식별자를 먼저 기록한다. 파일 열기가 실패하면 fd를 바로 닫는다.
            # (내용 기록 전에 실패해도 식별자로 내 잠금임을 알 수 있어 정리된다. Windows에서 fd가 남으면 지울 수도 없음)
            try:
                self.lock_id = file_identity(os.fstat(fd))
                handle = os.fdopen(fd, "w", encoding="utf-8")
            except BaseException:
                os.close(fd)
                raise
            with handle:
                handle.write(self.token)
            previous = self.previous_runs()
            if previous and not self.rerun_reason:
                raise unittest.SkipTest(f"[{self.name}] 이전 5-6 측정 기록이 있습니다({', '.join(previous)}). "
                                        "다시 측정하려면 --rerun-reason \"사유\" 를 지정하세요. (이전 기록은 지우지 않습니다)")
            base = self.log_dir / f"5-6_{self.name}_{self.run_id}"
            self.txt = open(f"{base}.txt", "x", encoding="utf-8")
            self.jsonl = open(f"{base}.jsonl", "x", encoding="utf-8")
            self.protected_before = protected_records(self.log_dir)
            self.ops_before = operational_state()
            self.started = self.clock()
            self.header = {"type": "header", "run_id": self.run_id, "suite": self.name, "started_at": datetime.now(KST).isoformat(),
                           "git": git_info(), "python": platform.python_version(), "torch": package_version("torch"),
                           "langgraph": package_version("langgraph"), "ollama_settings": ollama_key(),
                           "kinds": self.kinds, "planned_runs": self.planned, "time_cap_seconds": self.cap,
                           "rerun_reason": self.rerun_reason, "previous_runs": previous,
                           "note": "L층 unittest 통과는 S1 안전 불변식 통과만 뜻한다. 모델 판단·종합 흐름 성공은 아래 요약으로 판단한다.",
                           **(extra_header or {})}
            self.write(self.header)
        except BaseException:
            self.close()
            raise

    def write(self, record: dict[str, Any]) -> None:
        assert self.jsonl is not None and self.txt is not None
        self.jsonl.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self.jsonl.flush()
        if record.get("type") == "header":
            self.txt.write(f"[5-6 {self.name}] run_id={record['run_id']} | 커밋 {record['git'].get('commit')} | "
                           f"종류 {record['kinds']} | 예정 {record['planned_runs']}회 | 시간 상한 {record['time_cap_seconds']:.0f}초\n")
        elif record.get("type") == "run":
            marks = " ".join(f"{k}={v}" for k, v in record["indicators"].items() if v != NA)
            self.txt.write(f"[{record['unit']}] {marks} | 종합 {record.get('integrated', '-')} | S1 {record.get('s1') or '없음'} | "
                           f"도구 오류 {record.get('tool_error') or '없음'} | 원인 후보 {record.get('cause_hint')} | "
                           f"{'; '.join(record.get('problems') or [])[:400]}\n")
        else:
            self.txt.write(json.dumps(record, ensure_ascii=False, default=str)[:4000] + "\n")
        self.txt.flush()

    # 2. 회차 실행: 중단 조건을 회차 사이에서만 검사한다. (느린 CPU 추론 회차를 도중에 끊지 않음)
    def should_stop(self) -> str | None:
        if any(r.get("s1") for r in self.rows):
            return "S1 발생"
        if any(r.get("tool_error") for r in self.rows):
            return "측정 도구 오류 (제품 S1 아님, 도구를 고친 뒤 재측정)"
        streak = 0
        for row in reversed(self.rows):
            if not row.get("env_error"):
                break
            streak += 1
        if streak >= ENV_ERROR_STREAK_LIMIT:
            return f"환경 오류 연속 {streak}회"
        if self.clock() - self.started > self.cap:
            return f"시간 상한 {self.cap:.0f}초 초과"
        return None

    def run(self, units: list[tuple[str, Callable[[], dict[str, Any]]]], precondition: Callable[[], Any] | None = None):
        """units: [(회차 이름, 실행 함수)]. precondition은 오류 문장(또는 (오류 문장, 기록))을 돌려준다.
        오류가 있으면 사전 준비 기록을 남기고 모든 회차를 ENV_ERROR로 기록한다."""
        problem, info = None, None
        if precondition:
            try:
                answer = precondition()
            except Exception as exc:  # noqa: BLE001 - 장치·색인 준비 실패도 로그에 남긴다
                answer = f"사전 준비 실패: {type(exc).__name__}: {exc}"[:400]
            problem, info = answer if isinstance(answer, tuple) else (answer, None)
            self.write({"type": "precondition", "problem": problem, "info": info})
        for unit, execute in units:
            if not problem and not self.stop_reason:
                self.stop_reason = self.should_stop()
            if problem or self.stop_reason:
                status = ENV_ERROR if problem else NOT_RUN
                row: dict[str, Any] = {"type": "run", "unit": unit, "not_run": True, "env_error": bool(problem),
                                       "indicators": {k: status for k in INDICATORS}, "s1": [],
                                       "problems": [problem or self.stop_reason], "cause_hint": "환경 오류 후보" if problem else None}
            else:
                started = time.perf_counter()
                try:
                    row = dict(execute(), type="run", unit=unit)
                except Exception as exc:  # noqa: BLE001 - 측정 도구 자체의 예외: 제품 S1과 구분해 묶음을 멈춘다
                    row = {"type": "run", "unit": unit, "indicators": {k: NOT_RUN for k in INDICATORS}, "s1": [],
                           "tool_error": f"{type(exc).__name__}: {exc}"[:300], "env_error": False, "problems": [],
                           "cause_hint": "측정 도구 결함"}
                row["seconds"] = round(time.perf_counter() - started, 1)
            row["kinds"] = self.kinds
            self.rows.append(row)
            self.write(row)
        return self.rows

    # 3. 끝: 요약·지연 시간·보호 파일·운영 저장소 확인 → 잠금 해제
    def finish(self, extra: dict[str, Any] | None = None, cold_first: bool = False) -> dict[str, Any]:
        summary = {"type": "summary", "run_id": self.run_id, "stop_reason": self.stop_reason,
                   "summary": summarize(self.rows, self.planned), "latency": latency(self.rows, cold_first),
                   "protected_unchanged": protected_records(self.log_dir) == self.protected_before,
                   "operational_unchanged": operational_state() == self.ops_before,
                   "finished_at": datetime.now(KST).isoformat(), **(extra or {})}
        self.write(summary)
        self.close()
        return summary

    def close(self) -> None:
        for handle in (self.txt, self.jsonl):
            if handle and not handle.closed:
                handle.close()
        if self.token is not None and self.lock_path.exists():
            try:
                # 내가 만든 잠금만 지운다: 같은 파일(식별자 일치)이거나 내 token이 기록된 경우. 다른 실행의 잠금은 지우지 않는다.
                mine = self.lock_id is not None and file_identity(os.stat(self.lock_path)) == self.lock_id
                if mine or self.lock_path.read_text(encoding="utf-8") == self.token:
                    self.lock_path.unlink()
            except OSError:
                pass
        self.token = None
        self.lock_id = None


def ollama_ready() -> str | None:
    """실측 시작 조건: 승인된 모델 설정, Ollama 서버·모델 준비 (준비되지 않았으면 이유 문장). 설정을 바꾸지 않는다."""
    problem = approved_model_problem(ollama_key())
    if problem:
        return problem
    report = od.ollama_status(od.ollama_settings())
    if not (report.get("server_running") and report.get("model_installed")):
        return f"Ollama 서버 또는 모델이 준비되지 않았습니다: {report}"
    return None


def ollama_snapshot() -> dict[str, Any]:
    """Ollama 적재 상태 (호출 없이 /api/ps만 읽음). cold·warm 구분과 CPU·GPU 메모리 기록용 (PyTorch 장치와 따로 기록)"""
    try:
        report = du.ollama_report(probe=False)
        return {"before": report.get("before"), "server": report.get("server"), "model": report.get("model")}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def row_from(result: dict[str, Any]) -> dict[str, Any]:
    """판정 결과를 로그 회차 기록으로 바꾼다. (obs는 재현 묶음으로 함께 남긴다)"""
    row = {k: v for k, v in result.items() if k != "obs"}
    row["obs"] = result.get("obs")
    return row


# %% 기존 L층 테스트 래퍼의 결과 분류 (D9)
# 핵심 요약: 기존 테스트의 unittest 결과를 PASS / FAIL(assert) / ENV_ERROR(Ollama 연결·미준비) / NOT_RUN(그 밖의 건너뜀)으로 나눈다.
# 기존 테스트 통과를 모델 판단 성공으로 쓰지 않는다(판단 지표는 N/A). 운영 저장소·보호 로그 변화가 확인되면 S1이고,
# 확인하지 못했으면 '미검증'으로 남긴다(위반 없음으로 단정하지 않음).
ENV_WORDS = ("llm_unavailable", "connection_refused", "Ollama 서버 또는 모델이 준비되지 않았습니다", "NetworkBlocked")


def classify_existing(outcome: unittest.TestResult, *, ops_changed: bool | None, protected_changed: bool | None) -> dict[str, Any]:
    """기존 테스트 1개의 결과 분류"""
    text = " ".join(str(e[1]) for e in outcome.errors + outcome.failures) + " ".join(s[1] for s in outcome.skipped)
    if outcome.errors:
        status = ENV_ERROR if any(word in text for word in ENV_WORDS) else FAIL
    elif outcome.failures:
        status = FAIL
    elif outcome.skipped:
        status = ENV_ERROR if any(word in text for word in ENV_WORDS) else NOT_RUN
    else:
        status = PASS
    s1 = []
    if ops_changed:
        s1.append("기존 테스트 실행 중 운영 저장소 상태가 바뀜")
    if protected_changed:
        s1.append("기존 테스트 실행 중 보호 로그·공개 자료가 바뀜")
    safety = "미검증" if ops_changed is None or protected_changed is None else ("위반" if s1 else "확인함(운영 저장소·보호 로그 불변)")
    return {"existing_test": status, "s1": s1, "safety": safety, "env_error": status == ENV_ERROR,
            "indicators": dict({k: NA for k in INDICATORS}, graph_ok=NA),
            "unmeasured": "모델 판단·경로 지표는 기존 테스트 출력에만 있음(미측정, unittest 통과를 모델 성공으로 쓰지 않음)",
            "cause_hint": "원인 미확정" if status == FAIL else None, "problems": [text[-600:]] if text.strip() else []}


def parse_intent_summary(printed: str) -> dict[str, int] | None:
    """L_OD_LiveIntent 출력 요약 줄에서 실행 수·모델 의도 일치·최종 경로 일치·재확인 신호를 읽는다. 못 읽으면 None(미측정)"""
    import re as _re
    found = _re.search(r"실행 (\d+)회 \| 모델 원래 의도 일치 (\d+) \| 최종 경로 일치 (\d+) \| 재확인 신호 (\d+)", printed or "")
    if not found:
        return None
    runs, model_ok, route_ok, recheck = (int(v) for v in found.groups())
    return {"runs": runs, "model_intent_ok": model_ok, "route_ok": route_ok, "recheck": recheck}


# %% 실측 실행 함수 (L층 클래스와 사전 검증이 함께 쓰는 진입점)
def real_llm() -> RecordingLLM:
    return RecordingLLM(od.call_ollama)


def scripted_llm(sc: dict[str, Any]) -> RecordingLLM:
    """사전 검증용 가상 LLM: 시나리오의 올바른 이해(a·s)와 시간 선택 답을 돌려준다."""
    return RecordingLLM(ts.ScheduleLLM(understanding={m: tod.u(**v) for m, v in sc["a"].items()}, schedule=sc["s"],
                                       choices=sc["choices"]))


def graph_unit(sc: dict[str, Any], llm_factory: Callable[[dict[str, Any]], Any], *, live: bool = False,
               expect_ollama: dict[str, Any] | None = None) -> Callable[[], dict[str, Any]]:
    def execute() -> dict[str, Any]:
        obs = run_graph_scenario(sc, llm_factory(sc), live=live, expect_ollama=expect_ollama)
        return row_from(dict(graph_indicators(sc, obs), obs=obs, scenario=sc["id"], basis=sc["basis"]))
    return execute


# %% 재현 도구 (D5)
# 핵심 요약: 기록된 회차를 같은 LLM 원문·같은 순서, 같은 날씨·RAG 응답(예외 포함), 같은 기준 시각으로 다시 실행한다.
# 초기 저장소가 기록과 다르거나, 호출 순서·인자가 다르거나, 기록이 남거나 모자라면 reproduced=False다.
# 재현 결과는 실측 성공률에 넣지 않고, 재현이 안 된다는 이유만으로 '모델 판단 한계'로 정하지 않는다.
def normalized_doc(doc: dict[str, Any] | None) -> Any:
    """비교용 저장 문서: 실행마다 새로 만드는 값(store_id·action_id)만 지운다."""
    if not isinstance(doc, dict):
        return doc
    clean = copy.deepcopy(doc)
    clean.pop("store_id", None)
    for entry in clean.get("action_history", []):
        entry.pop("action_id", None)
    return clean


def replay_row(row: dict[str, Any]) -> dict[str, Any]:
    """기록된 회차 재현. LV1(별도 프로세스 단계는 재현 범위 밖)·LV9·그래프 시나리오·LV8을 지원한다."""
    obs = row["obs"]
    sid = obs["scenario"]
    now = datetime.fromisoformat(obs["now"])
    llm = ReplayLLM(obs["llm_calls"])
    weather, rag = ReplayCalls(obs["weather_calls"], "weather"), ReplayCalls(obs["rag_calls"], "rag")
    sc = next((s for s in GRAPH_SCENARIOS if s["id"] == sid), None)
    entry = next((e for e in LV9 if e["id"] == sid), None)
    spec = LV8C if sid == LV8C["id"] else LV8_SPECS.get(sid)
    if sc is not None:
        again = run_graph_scenario(sc, llm, weather_fn=weather, rag_fn=rag, now=now)
        replayed = graph_indicators(sc, again)
    elif sid.startswith("LV1-"):
        again = run_representative(sid.split("-", 1)[1], llm, child_mode="skip", now=now, weather_fn=weather, rag_fn=rag)
        replayed, again = again, again["obs"]
    elif entry is not None:
        again = run_understanding(entry, llm, now, weather_fn=weather, rag_fn=rag)
        replayed, again = again, again["obs"]
    elif spec is not None:
        selection = obs.get("selection")
        index = SimpleNamespace(device=(obs.get("turns") or [{}])[0].get("embedding_device") or {"requested": FAKE_DEVICE},
                                chunks=0)
        again = run_data_flow(spec, llm, weather_fn=weather, rag_fn=rag, now=now,
                              device=(selection or {}).get("selected") or FAKE_DEVICE, index=index, real_weather=False,
                              real_rag=False, real_llm=False, selection=selection)
        replayed, again = again, again["obs"]
    else:
        return {"reproduced": False, "reason": f"재현할 수 없는 사례 {sid}", "divergence": []}

    def comparable(turns: list[dict[str, Any]]) -> list[Any]:
        """턴 비교 값 (action_id는 실행마다 새로 만드는 값이라 뺀다. 별도 프로세스 턴은 재현 범위 밖)"""
        out = []
        for turn in turns:
            if str(turn.get("text", "")).startswith("[별도 프로세스]"):
                continue
            card_seen = {k: v for k, v in (turn.get("card") or {}).items() if k != "action_id"} or None
            out.append((turn.get("text"), card_seen, turn.get("answer"), turn.get("status"), turn.get("missing")))
        return out
    divergence = list(llm.divergence + weather.divergence + rag.divergence)
    if normalized_doc(again.get("initial_doc")) != normalized_doc(obs.get("initial_doc")):
        divergence.append("초기 저장소가 기록과 다름")
    for label, left in (("LLM", llm.remaining()), ("날씨", weather.remaining()), ("RAG", rag.remaining())):
        if left:
            divergence.append(f"{label} 기록 {left}건이 쓰이지 않음(기록보다 적게 호출)")
    if comparable(again["turns"]) != comparable(obs["turns"]):
        divergence.append("카드·답변·상태·질문 항목이 기록과 다름")
    if again.get("changed") != obs.get("changed") or normalized_doc(again.get("final_doc")) != normalized_doc(obs.get("final_doc")):
        divergence.append("저장 결과가 기록과 다름")
    return {"reproduced": not divergence, "divergence": divergence, "indicators": replayed.get("indicators"),
            "problems": replayed.get("problems"), "not_covered": "별도 프로세스 단계" if sid.startswith("LV1-") else None}


def classify_cause(row: dict[str, Any], replay: dict[str, Any], precheck_passed: bool) -> str:
    """원인 판정 (네 가지 중 하나). 근거가 부족하면 '원인 미확정'이다."""
    if row.get("env_error"):
        return "환경 오류"
    if not replay.get("reproduced"):
        return "원인 미확정"
    ind = row.get("indicators") or {}
    model_failed = FAIL in (ind.get("intent_ok"), ind.get("judgment_ok"))
    result_failed = any(ind.get(k) == FAIL for k in ("proposal_ok", "decision_ok", "saved_ok", "requery_ok", "persist_ok"))
    if result_failed and not model_failed:
        return "프로그램 결함"
    if model_failed and precheck_passed:
        return "모델 판단 한계"
    return "원인 미확정"


# %% M층. 판정·통계·기록 도구 단위 테스트
class M_LV1_Matchers(OfflineCase):

    def test_LV01_matcher_rules(self):
        """LV01 이해 칸 비교: 목록은 포함, '*'는 비어 있지 않음, '~'는 포함, _any는 하나라도, 그 밖은 같음"""
        actual = {"operation": "cancel", "request_mode": "request", "preferences": ["avoid_rain", "avoid_wind"],
                  "city_text": "경기도 광주", "shift_text": ""}
        self.assertEqual(matches(actual, {"operation": "cancel", "preferences": ["avoid_rain"], "city_text": "~경기"}), [])
        self.assertTrue(matches(actual, {"shift_text": "*"}))
        self.assertEqual(matches(actual, {"_any": [{"request_mode": "conditional"}, {"preferences": ["avoid_rain"]}]}), [])
        self.assertTrue(matches(actual, {"_any": [{"request_mode": "conditional"}, {"preferences": ["avoid_heat"]}]}))
        self.assertEqual(matches(None, {"operation": "x"}), ["이해 결과 없음"])

    def test_LV02_card_and_turn_checks(self):
        """LV02 카드·안내·질문·초안 비교: 카드 없음 기대에 카드가 있으면 실패, 지역은 허용 목록, 실행되지 않은 턴은 실패"""
        seen = {"type": "create_schedule", "schedule_id": None, "start": iso(27, 15), "city": "광주광역시", "title": "친구 만나기"}
        self.assertTrue(check_card(seen, None))
        self.assertTrue(check_card(None, card("create_schedule")))
        self.assertTrue(check_card(seen, card("create_schedule", city=GG_CITY)))
        self.assertEqual(check_card(seen, card("create_schedule", start=iso(27, 15), city=["광주광역시"], title="친구")), [])
        turns = [{"answer": "어느 지역인지 알려 주세요", "missing": ["city"], "card": None, "draft": {"target_id": "a"}},
                 {"answer": "등록할까요?", "missing": ["city"], "card": seen, "draft": {"target_id": "a"}}]
        problems = check_turns(turns, [[0, {"card": None, "missing": ["city"]}], [1, {"no_city_question": True, "draft_kept": True}],
                                       [5, {"card": None}]])
        self.assertEqual(problems, ["턴2 지역 질문 반복", "턴 5 미실행"])


class M_LV2_Indicators(OfflineCase):

    def obs(self, **over: Any) -> dict[str, Any]:
        base = {"scenario": "LV2-x", "graph_ok": True, "error": None, "changed": False, "ops_changed": False,
                "unused_created": False,
                "turns": [{"a_intent": "schedule_change", "s_understanding": {"operation": "update"}, "card": None,
                           "answer": "안내", "missing": [], "llm_error_statuses": [], "status": "needs_input"}]}
        base.update(over)
        return base

    def sc(self, **over: Any) -> dict[str, Any]:
        return scenario("LV2-x", "LV2", ["문장"], basis="시험", intents={0: SCHED}, understanding={0: {"operation": "update"}},
                        checks=over.pop("checks", [(0, {"card": None})]), **over)

    def test_LV03_model_failure_is_not_offset_by_safety(self):
        """LV03 모델이 틀렸지만 Python이 막으면 judgment FAIL + safe_fallback PASS. 성공으로 상쇄하지 않는다."""
        result = graph_indicators(self.sc(), self.obs(turns=[{"a_intent": "schedule_change", "s_understanding":
                                                              {"operation": "create"}, "card": None, "answer": "",
                                                              "missing": [], "llm_error_statuses": []}]))
        self.assertEqual((result["indicators"]["judgment_ok"], result["indicators"]["safe_fallback"]), (FAIL, PASS))
        self.assertEqual(result["cause_hint"], "모델 판단 한계 후보 (재현 필요)")
        self.assertEqual(result["s1"], [])

    def test_LV04_wrong_card_and_unapproved_save(self):
        """LV04 기대와 다른 카드가 남으면 safe_fallback FAIL, 승인 없이 파일이 바뀌면 S1"""
        wrong = {"type": "update_schedule", "schedule_id": "schedule_002", "start": iso(26, 16), "city": None, "title": None}
        result = graph_indicators(self.sc(), self.obs(turns=[{"a_intent": "schedule_change", "s_understanding":
                                                              {"operation": "update"}, "card": wrong, "answer": "",
                                                              "missing": [], "llm_error_statuses": []}]))
        self.assertEqual((result["indicators"]["proposal_ok"], result["indicators"]["safe_fallback"]), (FAIL, FAIL))
        self.assertEqual(result["cause_hint"], "프로그램 결함 후보 (재현 필요)")
        self.assertIn("승인 없는 저장(임시 저장소 파일이 바뀜)", graph_indicators(self.sc(), self.obs(changed=True))["s1"])
        self.assertTrue(graph_indicators(self.sc(), self.obs(graph_ok=False, error="Boom"))["s1"])
        self.assertTrue(graph_indicators(self.sc(), self.obs(ops_changed=True))["s1"])

    def test_LV05_env_error_marks_indicators(self):
        """LV05 Ollama 오류는 ENV_ERROR로 표시하고 모델 실패로 세지 않는다."""
        turn = {"a_intent": None, "s_understanding": None, "card": None, "answer": "", "missing": [],
                "llm_error_statuses": ["llm_unavailable"], "status": "llm_unavailable"}
        result = graph_indicators(self.sc(), self.obs(turns=[turn]))
        self.assertTrue(result["env_error"])
        self.assertEqual(result["indicators"]["judgment_ok"], ENV_ERROR)
        self.assertEqual(result["cause_hint"], "환경 오류 후보")

    def test_LV06_summary_counts_and_na(self):
        """LV06 요약: N/A는 성공에 넣지 않고, 성공률은 PASS/(PASS+FAIL). 미실행·환경 오류 회차를 따로 센다."""
        rows = [{"indicators": dict({k: NA for k in INDICATORS}, graph_ok=PASS, judgment_ok=PASS)},
                {"indicators": dict({k: NA for k in INDICATORS}, graph_ok=PASS, judgment_ok=FAIL)},
                {"indicators": {k: ENV_ERROR for k in INDICATORS}, "env_error": True},
                {"indicators": {k: NOT_RUN for k in INDICATORS}, "not_run": True}]
        out = summarize(rows, planned=4)
        judgment = out["indicators"]["judgment_ok"]
        self.assertEqual((out["planned"], out["executed"], out["not_run_runs"], out["env_error_runs"]), (4, 3, 1, 1))
        self.assertEqual((judgment[PASS], judgment[FAIL], judgment[ENV_ERROR], judgment[NOT_RUN], judgment["evaluable"],
                          judgment["rate"]), (1, 1, 1, 1, 2, 0.5))
        self.assertEqual(out["indicators"]["saved_ok"][NA], 2)
        self.assertIsNone(out["indicators"]["saved_ok"]["rate"])

    def test_LV07_stats_and_latency(self):
        """LV07 응답 시간 통계: 표본 수·최솟값·중앙값·평균·최댓값, 첫 호출 cold 분리"""
        self.assertEqual(stats([]), {"n": 0})
        self.assertEqual(stats([1.0, 3.0, 2.0]), {"n": 3, "min": 1.0, "median": 2.0, "mean": 2.0, "max": 3.0})
        rows = [{"obs": {"turns": [{"text": "a", "seconds": 5.0, "llm_calls": [{"stage": "understanding", "seconds": 4.0},
                                                                              {"stage": "answer", "seconds": 1.0}]},
                                   {"text": "[approve]", "seconds": 0, "llm_calls": []}]}}]
        out = latency(rows, cold_first=True)
        self.assertEqual(out["cold_first_call_seconds"]["n"], 1)
        self.assertEqual(out["llm_stage_seconds"]["answer"]["n"], 1)
        self.assertEqual(out["turn_seconds"]["n"], 1)

    def test_LV08_components_require_one_flow(self):
        """LV08 4.8 종합 판정: 필수 구성 요소가 모두 PASS일 때만 PASS. RAG 미실행(NOT_RUN)·필수 요소의 N/A는 성공으로 바꾸지 않고
        NOT_RUN, 본문 관련성 수동 검토(REVIEW)는 REVIEW, 실패·환경 오류가 있으면 그 상태다. 종합 흐름이 아닌 사례는 N/A"""
        required = required_components(LV8_SPECS["LV8-delegate"], real_weather=True)
        self.assertIn("weather_real", required)
        self.assertNotIn("weather_real", required_components(LV8C, real_weather=False))
        self.assertEqual(required_components(LV8_SPECS["LV8-weather"], real_weather=True), [])
        all_pass = {k: PASS for k in COMPONENTS}
        self.assertEqual(integrated_success(all_pass, required), PASS)
        self.assertEqual(integrated_success(dict(all_pass, rag_call=NOT_RUN), required), NOT_RUN)
        self.assertEqual(integrated_success(dict(all_pass, device=NA), required), NOT_RUN)
        self.assertEqual(integrated_success(dict(all_pass, rag_content=REVIEW), required), REVIEW)
        self.assertEqual(integrated_success(dict(all_pass, proposal=FAIL, weather_real=ENV_ERROR), required), ENV_ERROR)
        self.assertEqual(integrated_success(dict(all_pass, proposal=FAIL), required), FAIL)
        self.assertEqual(integrated_success(all_pass, []), NA)

    def test_LV09_rag_probe_judge_has_no_new_threshold(self):
        """LV09 RAG 판정: 출처 파일이 주제와 맞는지만 본다. 점수는 기록만 하고 새 임계값으로 쓰지 않는다."""
        probe = {"topic": "한파"}
        hit = {"filename": sr.WINTER, "pdf_page": 1, "printed_page": None, "similarity": 0.31, "content": "방한"}
        self.assertEqual(judge_rag_probe(probe, {"status": "ok", "results": [hit]})["status"], PASS)
        self.assertEqual(judge_rag_probe(probe, {"status": "ok", "results": [dict(hit, filename=sr.HEAT)]})["status"], FAIL)
        # (5-6 최종 보완) search_error는 환경 문제라는 근거가 없어 원인 미확정 FAIL (환경 오류로 합치지 않음)
        searched = judge_rag_probe(probe, {"status": "search_error", "results": []})
        self.assertEqual((searched["status"], searched["reason"]), (FAIL, "search_error (원인 미확정)"))
        self.assertEqual(sr.cosine_score(0.2), 0.8)          # 점수 = 1 − 거리 (거리 아님)
        self.assertEqual(sr.MIN_SIMILARITY, 0.30)


class M_LV3_SuiteMachinery(OfflineCase):

    def setUp(self):
        super().setUp()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)

    def suite(self, **over: Any) -> LiveSuite:
        over.setdefault("log_dir", self.dir)
        over.setdefault("live", False)
        return LiveSuite("LVT", over.pop("planned", 3), kinds={"llm": "fake"}, **over)

    def ok_unit(self, env: bool = False, s1: bool = False) -> Callable[[], dict[str, Any]]:
        return lambda: {"indicators": dict({k: NA for k in INDICATORS}, graph_ok=PASS), "env_error": env,
                        "s1": ["승인 없는 저장"] if s1 else [], "problems": []}

    def test_LV10_logs_are_new_files_and_lock_is_exclusive(self):
        """LV10 로그는 x 모드로 새로 만들고, 실행 중에는 같은 폴더의 다른 묶음이 시작하지 못하며, 끝나면 내 잠금만 지운다."""
        first = self.suite()
        first.start()
        second = self.suite(rerun_reason="동시 실행 시험")      # 사유가 있어도 잠금이 있으면 시작하지 않는다
        with self.assertRaises(unittest.SkipTest) as caught:
            second.start()
        self.assertIn("다른 5-6 측정이 실행 중", str(caught.exception))
        first.run([("u1", self.ok_unit())])
        summary = first.finish()
        self.assertFalse((self.dir / LOCK_NAME).exists())
        self.assertEqual(summary["summary"]["executed"], 1)
        logs = list(self.dir.glob("5-6_LVT_*.jsonl"))
        self.assertEqual(len(logs), 1)
        records = [json.loads(line) for line in logs[0].read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["type"] for r in records], ["header", "run", "summary"])
        self.assertIn("S1 안전 불변식 통과만", records[0]["note"])
        (self.dir / LOCK_NAME).write_text("다른 실행", encoding="utf-8")
        third = LiveSuite("LVT", 1, kinds={}, log_dir=self.dir, live=False, rerun_reason="잠금 시험")
        with self.assertRaises(unittest.SkipTest):
            third.start()
        self.assertEqual((self.dir / LOCK_NAME).read_text(encoding="utf-8"), "다른 실행")   # 남의 잠금은 지우지 않음

    def test_LV11_rerun_needs_reason_and_keeps_previous(self):
        """LV11 같은 묶음의 이전 기록이 있으면 사유 없이 다시 실행하지 않는다. 사유가 있으면 기록과 함께 새 파일로 실행한다."""
        (self.dir / "5-6_LVT_old.jsonl").write_text("{}\n", encoding="utf-8")
        with self.assertRaises(unittest.SkipTest):
            self.suite().start()
        again = self.suite(rerun_reason="환경 오류로 중단된 묶음 재실행")
        again.start()
        self.assertEqual(again.header["previous_runs"], ["5-6_LVT_old.jsonl"])
        again.finish()
        self.assertEqual((self.dir / "5-6_LVT_old.jsonl").read_text(encoding="utf-8"), "{}\n")

    def test_LV12_stop_conditions_record_not_run(self):
        """LV12 S1·환경 오류 연속 2회·시간 상한에서 멈추고 남은 회차를 NOT_RUN으로 남긴다. 시작 조건 실패는 전부 ENV_ERROR."""
        for label, units, expected in (
                ("S1", [self.ok_unit(s1=True), self.ok_unit(), self.ok_unit()], "S1 발생"),
                ("환경", [self.ok_unit(env=True), self.ok_unit(env=True), self.ok_unit()], "환경 오류 연속 2회")):
            with self.subTest(label):
                suite = self.suite(rerun_reason=label)
                suite.start()
                rows = suite.run([(f"u{i}", unit) for i, unit in enumerate(units)])
                suite.finish()
                self.assertEqual(suite.stop_reason, expected)
                self.assertTrue(rows[-1]["not_run"])
                self.assertEqual(rows[-1]["indicators"]["graph_ok"], NOT_RUN)
        ticks = iter([0.0, 0.0, 10_000.0, 10_000.0, 10_000.0])
        suite = self.suite(rerun_reason="시간", clock=lambda: next(ticks), per_run_seconds=1.0)
        suite.start()
        rows = suite.run([("a", self.ok_unit()), ("b", self.ok_unit())])
        suite.finish()
        self.assertTrue(rows[1]["not_run"])
        self.assertIn("시간 상한", suite.stop_reason or "")
        suite = self.suite(rerun_reason="시작 조건")
        suite.start()
        rows = suite.run([("a", self.ok_unit())], precondition=lambda: "Ollama 서버 꺼짐")
        summary = suite.finish()
        self.assertEqual(rows[0]["indicators"]["judgment_ok"], ENV_ERROR)
        self.assertEqual(summary["summary"]["env_error_runs"], 1)

    def test_LV13_sanitize_and_network_guard(self):
        """LV13 공개용 사본은 프로젝트 경로·사용자 폴더·지정 문자열을 가린다. 기본 테스트에서는 네트워크 연결이 막혀 있다."""
        text = f"{ROOT}\\x.py {Path.home()} server.example 10.0.0.5"
        clean = sanitize(text, extra=("server.example", "10.0.0.5"))
        self.assertNotIn(str(ROOT), clean)
        self.assertNotIn(str(Path.home()), clean)
        self.assertNotIn("10.0.0.5", clean)
        with self.assertRaises(NetworkBlocked):
            socket.create_connection(("127.0.0.1", 11434), timeout=1)
        with self.assertRaises(od.DecisionError) as caught:
            od.call_ollama([{"role": "user", "content": "x"}], {}, od.ollama_settings(), stage="understanding")
        self.assertEqual(caught.exception.status, "llm_unavailable")
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")

    def test_LV14_imports_did_not_parse_cli_or_touch_store(self):
        """LV14 기존 테스트 모듈을 불러와도 CLI 옵션이 켜지거나 운영 저장소가 생기지 않는다. (main은 __main__에서만)"""
        self.assertFalse(ts.OPTIONS.live_ollama)
        self.assertIsNone(ts.OPTIONS.device)
        self.assertFalse(tod.OPTIONS.live_full)
        self.assertFalse(tsg.OPTIONS.live_ollama)
        self.assertEqual({str(p): ts.file_state(p) for p in ts.OPERATIONAL_FILES}, ts.OPERATIONAL_BEFORE)

    def test_LV15_fixed_counts_match_plan(self):
        """LV15 실측 사례와 반복 횟수는 계획과 같이 고정돼 있다. (실측 뒤에 바꾸지 않음)"""
        counts = {name: sum(s["repeats"] for s in group) for name, group in SUITES.items()}
        self.assertEqual(counts, {"LV2": 24, "LV3": 12, "LV4": 27, "LV5": 9, "LV6": 24, "LV7": 12})
        self.assertEqual((len(LV1_RUNS), len(LV9), sum(e["repeats"] for e in LV9)), (6, 21, 63))
        self.assertEqual((sum(s["repeats"] for s in LV8), LV8C["repeats"], len(LV8R)), (12, 3, 6))
        ids = [s["id"] for s in GRAPH_SCENARIOS] + [e["id"] for e in LV9]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(s["repeats"] == REPEATS for s in GRAPH_SCENARIOS))
        json.dumps(GRAPH_SCENARIOS + LV9 + LV8 + [LV8C] + LV8R, ensure_ascii=False)


class M_LV4_Replay(OfflineCase):

    def test_LV16_replay_tools_return_recorded_values_and_report_divergence(self):
        """LV16 재현기: 기록된 LLM 원문·날씨·RAG 응답을 같은 순서로 돌려주고, 단계·인자가 다르면 어긋남을 기록한다."""
        llm = ReplayLLM([{"stage": "understanding", "raw": "{\"a\": 1}", "model": "m"},
                         {"stage": "answer", "raw": None, "error": {"status": "llm_unavailable", "code": "x", "message": "m"}}])
        self.assertEqual(llm([], {}, {"model": "m"}, stage="understanding")[0], "{\"a\": 1}")
        with self.assertRaises(od.DecisionError):
            llm([], {}, {"model": "m"}, stage="schedule_choice")
        self.assertIn("2번째 호출 단계 answer → schedule_choice", llm.divergence)
        weather = ReplayCalls([{"kwargs": {"outing_date": "2026-11-26"}, "result": {"status": "ok"}}], "weather")
        self.assertEqual(weather(outing_date="2026-11-27")["status"], "ok")
        self.assertTrue(weather.divergence)
        self.assertEqual(weather(outing_date="x")["status"], "api_error")


# %% I층. 사전 검증 (실측과 같은 입력·예보·저장소 + 올바른 이해 → 기대 결과를 실측 전에 고정)
class I_LV1_RepresentativePrecheck(OfflineCase):

    def test_LV20_representative_flow_judges_all_indicators(self):
        """LV20 4.1 흐름: 올바른 이해(FLOW_A·FLOW_S)로 승인·거절이 모두 기대대로 판정된다.
        승인은 저장·두 번 누름·재조회·새 그래프·별도 프로세스(가상 LLM, 네트워크 차단)까지, 거절은 파일·해시 불변·원래 시각 재조회"""
        for kind in ("approve", "reject"):
            with self.subTest(kind):
                llm = RecordingLLM(ts.ScheduleLLM(understanding=ts.FLOW_A, schedule=ts.FLOW_S))
                result = run_representative(kind, llm, child_mode="fake")
                ind = result["indicators"]
                self.assertEqual(result["s1"], [], result["problems"])
                expected = {"graph_ok", "intent_ok", "judgment_ok", "proposal_ok", "decision_ok", "requery_ok", "persist_ok"}
                if kind == "approve":
                    expected.add("saved_ok")
                self.assertEqual({k for k, v in ind.items() if v == PASS}, expected, (ind, result["problems"]))
                obs = result["obs"]
                self.assertEqual(obs["changed"], kind == "approve")
                if kind == "approve":
                    self.assertTrue(obs["child"]["ok"], obs["child"])
                    self.assertNotEqual(obs["child"]["pid"], os.getpid())
                else:
                    self.assertEqual(obs["before_sha"], obs["after_sha"])

    def test_LV21_wrong_proposal_is_not_approved(self):
        """LV21 모델이 엉뚱한 시각을 내 제안이 기대와 다르면 승인하지 않고, 후속 지표는 NOT_RUN, 파일은 그대로다."""
        # 모델이 '오후 네 시쯤'에서 '오후'만 옮겨 적은 경우: 근거 확인은 통과하지만 시간대 탐색이 되어 17:00이 아닌 시각을 낸다.
        wrong = dict(ts.FLOW_S, **{"오후 네 시쯤": {"operation": "update", "answers_previous_question": True,
                                                  "new_time_text": "오후"}})
        result = run_representative("approve", RecordingLLM(ts.ScheduleLLM(understanding=ts.FLOW_A, schedule=wrong)),
                                    child_mode="fake")
        self.assertEqual(result["indicators"]["proposal_ok"], FAIL)
        self.assertEqual([result["indicators"][k] for k in ("decision_ok", "saved_ok", "requery_ok", "persist_ok")], [NOT_RUN] * 4)
        self.assertFalse(result["obs"]["changed"])
        self.assertEqual(result["s1"], [])


class I_LV2_ScenarioPrecheck(OfflineCase):

    def test_LV22_every_graph_scenario_expectation_is_reachable(self):
        """LV22 4.2~4.7 모든 시나리오: 올바른 이해를 넣으면 기대 결과(카드·안내·질문·의도·이해)가 모두 PASS이고 S1이 없다.
        (실측 기대값이 실제 코드로 도달 가능한 값임을 실측 전에 확인한다)"""
        for sc in GRAPH_SCENARIOS:
            with self.subTest(sc["id"]):
                obs = run_graph_scenario(sc, scripted_llm(sc))
                result = graph_indicators(sc, obs)
                self.assertEqual(result["s1"], [])
                self.assertEqual(result["problems"], [], [(t["text"], t["answer"][:160], t["card"], t["missing"], t["kind"])
                                                          for t in obs["turns"]])
                for key in ("intent_ok", "judgment_ok"):
                    self.assertIn(result["indicators"][key], (PASS, NA))
                self.assertEqual(result["indicators"]["proposal_ok"], PASS)
                self.assertFalse(obs["changed"])

    def test_LV23_conditional_vs_preference_create(self):
        """LV23 (외부 검토 3) AG41 문장을 모델이 conditional로 분류하면 미지원 안내(카드 없음·예보 조회 없음)가 되어
        AG41 기대(즉시 예보 확인 경로)와 다르다 → judgment FAIL·proposal FAIL·safe_fallback PASS. 조건을 지운 등록은 조건 무시 카드다."""
        sc = next(s for s in LV2 if s["id"] == "LV2-direct-clear")
        as_conditional = dict(sc, s={AG41_MSG: dict(AG41_U, request_mode="conditional"), "인천": dict(CITY_U, operation="create")})
        result = graph_indicators(sc, run_graph_scenario(as_conditional, scripted_llm(as_conditional)))
        self.assertEqual([result["indicators"][k] for k in ("judgment_ok", "proposal_ok", "safe_fallback")], [FAIL, FAIL, PASS])
        dropped = dict(sc, s={AG41_MSG: {k: v for k, v in AG41_U.items() if k != "preferences"},
                              "인천": dict(CITY_U, operation="create")})
        obs = run_graph_scenario(dropped, scripted_llm(dropped))
        result = graph_indicators(sc, obs)
        self.assertEqual(result["indicators"]["judgment_ok"], FAIL)
        self.assertIsNotNone(obs["turns"][0]["card"])                # 조건을 지우면 예보 확인 없이 카드가 생긴다
        self.assertEqual(result["indicators"]["proposal_ok"], FAIL)

    def test_LV24_replay_reproduces_recorded_run(self):
        """LV24 기록된 회차(LLM 원문·예보·RAG 응답)를 다시 재생하면 같은 결과가 나온다. (재현 기록은 성공률과 분리)"""
        sc = next(s for s in LV2 if s["id"] == "LV2-direct-clear")
        obs = run_graph_scenario(sc, scripted_llm(sc))
        replay = replay_row({"obs": obs})
        self.assertTrue(replay["reproduced"], replay)
        self.assertEqual(replay["problems"], [])

    def test_LV28_D1_city_answer_keeps_date_and_time(self):
        """LV28 결함 D1 회귀 (그래프 경로): '모레 오후 세 시에 광주에서 친구 만나기로 했어.' → 어느 광주인지 질문
        → '경기도 광주'·'광주광역시'·(변경) '경기도 광주' → 날짜·시각·대상이 그대로 이어진 카드.
        (수정 전에는 날짜·대상을 다시 물었다. schedule_agent의 hold_said·restore_held로 보완, 에이전트 단위 회귀는 test_schedule AG48~53)"""
        for sid in ("LV6-city-gg", "LV6-city-metro", "LV6-update-city"):
            with self.subTest(sid):
                sc = next(s for s in GRAPH_SCENARIOS if s["id"] == sid)
                obs = run_graph_scenario(sc, scripted_llm(sc))
                self.assertEqual(graph_indicators(sc, obs)["problems"], [])


class I_LV3_UnderstandingPrecheck(OfflineCase):

    def test_LV25_understanding_items(self):
        """LV25 4.9-B 21문장: 올바른 이해를 넣으면 필수 칸 판정이 PASS이고, 결과 종류(유지·보호·미지원)가 기대와 같다. 파일은 그대로다."""
        for entry in LV9:
            with self.subTest(entry["id"]):
                llm = RecordingLLM(ts.ScheduleLLM(schedule={entry["text"]: entry["s"]}))
                result = run_understanding(entry, llm)
                self.assertEqual(result["s1"], [])
                self.assertEqual(result["indicators"]["judgment_ok"], PASS, result["problems"])
                self.assertEqual(outcome_problems(entry, result["obs"]["turns"][0]), [])


class I_LV4_DataFlowPrecheck(OfflineCase):

    def test_LV26_controlled_flow_components(self):
        """LV26 4.8-C 흐름: 가상 LLM·가상 RAG로 판정 경로를 확인한다. 고른 17시에 강풍 신호가 있어 RAG가 실행되고(풍수해 자료),
        17:00 카드를 승인해 저장한다. 가상 구성 요소는 real_*=False로 N/A가 되어 종합 성공(PASS)으로 세지 않는다."""
        rag = tod.FakeRAG()
        result = run_data_flow(LV8C, scripted_llm(LV8C), weather_fn=make_weather(LV8C["forecast"]), rag_fn=rag, now=FIXED_NOW,
                               device=FAKE_DEVICE, index=tod.FAKE_INDEX, real_weather=False, real_rag=False, real_llm=False)
        parts = result["components"]
        self.assertEqual(result["s1"], [], result["problems"])
        self.assertTrue(rag.calls)
        self.assertEqual((parts["llm_judgment"], parts["proposal"], parts["graph_decision"], parts["json_change"]),
                         (PASS, PASS, PASS, PASS), result["component_reasons"])
        self.assertEqual((parts["rag_call"], parts["rag_source"], parts["rag_content"]), (PASS, PASS, REVIEW))
        self.assertEqual((parts["weather_real"], parts["qwen_call"], parts["device"]), (NA, NA, NA))
        self.assertNotEqual(result["integrated"], PASS)
        self.assertTrue(result["obs"]["changed"])

    def test_LV27_real_weather_flow_without_signal(self):
        """LV27 4.8 흐름(가상 대체): 비 신호가 없는 예보면 위임 카드가 없거나 RAG가 실행되지 않아 해당 구성 요소는 NOT_RUN,
        종합 성공이 아니다. 데이터는 바꾸지 않는다."""
        spec = LV8[2]
        llm = RecordingLLM(ts.ScheduleLLM(understanding={spec["turns"][0]: tod.u(intent="schedule_change")},
                                          schedule={spec["turns"][0]: {"operation": "update", "target_text": "시장",
                                                                      "delegate_choice": True, "preferences": ["avoid_rain"]}},
                                          choices=[{"choice": "C1", "reasons": ["no_conflict"]}]))
        rag = tod.FakeRAG()
        result = run_data_flow(spec, llm, weather_fn=make_weather({}), rag_fn=rag, now=FIXED_NOW, device=FAKE_DEVICE,
                               index=tod.FAKE_INDEX, real_weather=True, real_rag=True, real_llm=True)
        self.assertEqual(result["s1"], [], result["problems"])
        self.assertEqual(result["components"]["rag_call"], NOT_RUN)
        self.assertNotEqual(result["integrated"], PASS)
        self.assertFalse(result["obs"]["changed"] and result["components"]["json_change"] != PASS)


# %% M층·I층. 2차 외부 검토(D2~D11) 회귀: 판정·재현·안전 불변식·잠금이 '있는지'가 아니라 '맞는지'를 본다
class M_LV5_ComponentJudge(OfflineCase):
    TOMORROW_15 = iso(26, 15)

    def delegate_obs(self, **over: Any) -> dict[str, Any]:
        """LV8-delegate 모양의 합성 기록 (내일 15:00 시장 카드, 인천 예보)"""
        card_seen = {"type": "update_schedule", "schedule_id": SID["시장"], "start": self.TOMORROW_15, "end": iso(26, 16),
                     "city": "인천", "title": "시장 장보기"}
        base = {"turns": [{"a_intent": "schedule_change", "s_understanding": {"operation": "update", "delegate_choice": True,
                                                                             "preferences": ["avoid_rain"]},
                           "card": card_seen, "status": "ok", "llm_error_statuses": [],
                           "s_weather": [{"at": self.TOMORROW_15, "status": "no_signal"}]}],
                # (D25 보완) 응답 계약 검사를 위해 1단계 메타데이터(지역·시각·구간·출처)와 missing_fields를 채운다. 수치는 그대로
                "weather_calls": [{"args": [], "kwargs": {"outing_date": "2026-11-26", "outing_time": "15:00", "city": "인천"},
                                   "result": {"status": "ok", "weather": dict(contract_meta("인천", "2026-11-26", "15:00"),
                                                                              **tod.BASE_READINGS, missing_fields=[])}}],
                "rag_calls": [], "llm_calls": [{"stage": "schedule_understanding", "seconds": 1.0, "raw": "{\"operation\": \"update\"}"}],
                "initial_doc": ts.seed_doc(), "now": FIXED_NOW.isoformat(), "decision_status": NOT_RUN, "json_status": NOT_RUN}
        base.update(over)
        return base

    def test_LV30_reviewer_synthetic_case_is_not_success(self):
        """LV30 (D2 재현) 외부 검토의 합성 데이터: 날씨 성공+오류, 무관한 PDF, 깨진 LLM 원문, operation=other, 잘못된 카드,
        승인·저장 PASS 표시 → 해당 구성 요소가 PASS가 아니고 종합 성공도 아니다."""
        obs = {"turns": [{"s_understanding": {"operation": "other"}, "card": {"type": "wrong"}, "status": "ok",
                          "llm_error_statuses": [], "s_weather": []}],
               "weather_calls": [{"args": [], "kwargs": {}, "result": {"status": "ok"}},
                                 {"args": [], "kwargs": {}, "result": {"status": "network_error"}}],
               "rag_calls": [{"args": ["q"], "kwargs": {}, "result": {"status": "ok", "results": [{"filename": "unrelated.pdf"}]}}],
               "llm_calls": [{"stage": "schedule_understanding", "seconds": 1, "raw": "invalid"}],
               "decision_status": PASS, "json_status": PASS, "initial_doc": ts.seed_doc(), "now": FIXED_NOW.isoformat()}
        judged = judge_components(obs, LV8_SPECS["LV8-delegate"], real_weather=True, real_rag=True, real_llm=True)
        status = judged["status"]
        for key in ("qwen_format", "llm_judgment", "proposal", "rag_source", "weather_real"):
            self.assertNotEqual(status[key], PASS, (key, judged["reasons"].get(key)))
        self.assertNotEqual(integrated_success(status, required_components(LV8_SPECS["LV8-delegate"], real_weather=True)), PASS)

    def test_LV31_weather_required_calls_and_concealment(self):
        """LV31 필수 호출(카드 시각·요청 지역)이 ok여야 PASS, 필수 호출 실패는 ENV_ERROR, 카드 시각 호출이 없으면 FAIL.
        보조 호출 실패가 unknown으로 처리되면 경고만, no_signal로 보고되면 오류 은폐 FAIL"""
        spec = LV8_SPECS["LV8-delegate"]
        ok = judge_components(self.delegate_obs(), spec, real_weather=True, real_rag=True, real_llm=True)
        self.assertEqual(ok["status"]["weather_real"], PASS, ok["reasons"])
        failed_side = {"args": [], "kwargs": {"outing_date": "2026-11-26", "outing_time": "14:30", "city": "인천"},
                       "result": {"status": "network_error", "weather": None}}
        warned = self.delegate_obs(weather_calls=self.delegate_obs()["weather_calls"] + [failed_side])
        warned["turns"][0]["s_weather"].append({"at": iso(26, 14, 30), "status": "unknown"})
        judged = judge_components(warned, spec, real_weather=True, real_rag=True, real_llm=True)
        self.assertEqual(judged["status"]["weather_real"], PASS)
        self.assertTrue(judged["warnings"])
        concealed = copy.deepcopy(warned)
        concealed["turns"][0]["s_weather"][-1]["status"] = "no_signal"
        self.assertEqual(judge_components(concealed, spec, real_weather=True, real_rag=True, real_llm=True)["status"]["weather_real"], FAIL)
        broken = self.delegate_obs()
        # (5-6 최종 보완) 외부 서비스 문제라는 근거(HTTP 5xx)가 있는 api_error만 ENV_ERROR, 근거가 없으면 원인 미확정 FAIL (LV54)
        broken["weather_calls"][0]["result"] = {"status": "api_error", "message": "기상 API가 요청을 처리하지 못했습니다(HTTP 503).",
                                                "weather": None}
        self.assertEqual(judge_components(broken, spec, real_weather=True, real_rag=True, real_llm=True)["status"]["weather_real"], ENV_ERROR)
        broken["weather_calls"][0]["result"] = {"status": "api_error", "weather": None}
        self.assertEqual(judge_components(broken, spec, real_weather=True, real_rag=True, real_llm=True)["status"]["weather_real"], FAIL)
        missing = self.delegate_obs(weather_calls=[])
        self.assertEqual(judge_components(missing, spec, real_weather=True, real_rag=True, real_llm=True)["status"]["weather_real"], FAIL)

    def test_LV32_rag_call_source_and_content_are_separate(self):
        """LV32 RAG 호출 성공·출처 일치·본문 관련성을 나눈다. 본문 관련성은 자동 확정하지 않고 REVIEW(수동 검토)이며 PASS가 아니다."""
        spec = LV8C
        hit = {"filename": sr.STORM, "pdf_page": 1, "similarity": 0.5, "content": "강풍이 불 때는 간판 근처를 피합니다."}
        obs = self.delegate_obs(rag_calls=[{"args": ["강풍 질문"], "kwargs": {}, "result": {"status": "ok", "results": [hit]}}])
        status = judge_components(obs, spec, real_weather=False, real_rag=True, real_llm=True)["status"]
        self.assertEqual((status["rag_call"], status["rag_source"], status["rag_content"]), (PASS, PASS, REVIEW))
        wrong = self.delegate_obs(rag_calls=[{"args": ["q"], "kwargs": {}, "result": {"status": "ok", "results": [dict(hit, filename=sr.HEAT)]}}])
        self.assertEqual(judge_components(wrong, spec, real_weather=False, real_rag=True, real_llm=True)["status"]["rag_source"], FAIL)
        error = self.delegate_obs(rag_calls=[{"args": ["q"], "kwargs": {}, "result": {"status": "search_error", "results": []}}])
        # (5-6 최종 보완) search_error는 환경 문제라는 근거가 없어 원인 미확정 FAIL (연결·자원 예외만 ENV_ERROR, LV54)
        self.assertEqual(judge_components(error, spec, real_weather=False, real_rag=True, real_llm=True)["status"]["rag_call"], FAIL)
        none = judge_components(self.delegate_obs(), spec, real_weather=False, real_rag=True, real_llm=True)["status"]
        self.assertEqual((none["rag_call"], none["rag_source"], none["rag_content"]), (NOT_RUN, NOT_RUN, NOT_RUN))
        self.assertNotEqual(integrated_success(dict({k: PASS for k in COMPONENTS}, rag_content=REVIEW),
                                               list(COMPONENTS)), PASS)
        self.assertEqual(integrated_success(dict({k: PASS for k in COMPONENTS}, rag_content=REVIEW), list(COMPONENTS)), REVIEW)

    def test_LV33_card_content_is_checked(self):
        """LV33 카드 판정: 종류·대상·날짜·소요 시간·지역·시각 외 항목이 저장 일정·기대와 맞아야 한다."""
        doc = ts.seed_doc()
        good = {"type": "update_schedule", "schedule_id": SID["시장"], "start": iso(26, 15), "end": iso(26, 16), "city": "인천",
                "title": "시장 장보기"}
        want = LV8_SPECS["LV8-delegate"]["card"]
        self.assertEqual(judge_card(good, want, doc, FIXED_NOW), [])
        for change, label in (({"end": iso(26, 17)}, "소요 시간"), ({"title": "다른 이름"}, "시각 외"), ({"city": "서울"}, "지역"),
                              ({"start": iso(27, 15), "end": iso(27, 16)}, "날짜"), ({"schedule_id": SID["복지관"]}, "대상")):
            with self.subTest(label):
                self.assertTrue(judge_card(dict(good, **change), want, doc, FIXED_NOW))

    def test_LV34_device_compares_kind_and_index(self):
        """LV34 (D3 재현) 실제 장치 판정: requested는 장치 종류, weights·token·embedding은 장치 번호까지 선택 결과와 같아야 하고,
        차원 384. 불일치는 FAIL(S1 장치 오보고), RAG가 실행되지 않으면 NOT_RUN. 특정 번호를 고정하지 않는다."""
        for kind, number in (("xpu", "xpu:0"), ("cuda", "cuda:1"), ("cpu", "cpu")):
            with self.subTest(kind):
                selection = {"selected": kind, "device": number}
                good = {"requested": kind, "weights": number, "token_input": number, "token_output": number, "embedding": number,
                        "dimensions": 384}
                self.assertEqual(judge_device(selection, [good])[0], PASS)
                self.assertEqual(judge_device(selection, [])[0], NOT_RUN)
                self.assertEqual(judge_device(selection, [dict(good, dimensions=768)])[0], FAIL)
                self.assertEqual(judge_device(selection, [dict(good, requested="other")])[0], FAIL)
        self.assertEqual(judge_device({"selected": "cuda", "device": "cuda:0"},
                                      [{"requested": "cuda", "weights": "cuda:1", "token_input": "cuda:1", "token_output": "cuda:1",
                                        "embedding": "cuda:1", "dimensions": 384}])[0], FAIL)
        self.assertEqual(judge_device(None, [{"requested": "xpu"}])[0], FAIL)

    def test_LV35_ollama_settings_follow_environment(self):
        """LV35 (D4 재현) 가상 측정은 OLLAMA_*를 지우고, 실측 환경은 사용자 설정을 그대로 둔다. 실측의 준비 확인과 호출 설정이 같다."""
        with mock.patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://127.0.0.1:12345", "OLLAMA_MODEL": "other:1b"}):
            outer = ollama_key()
            with tempfile.TemporaryDirectory() as name:
                live_env, fake_env = safe_env(Path(name), live=True), safe_env(Path(name))
            with mock.patch.dict(os.environ, live_env, clear=True):
                self.assertEqual(ollama_key(), outer)
            with mock.patch.dict(os.environ, fake_env, clear=True):
                self.assertEqual(ollama_key()["model"], "qwen2.5:3b")
            self.assertIn("qwen2.5:3b", approved_model_problem(outer) or "")
        self.assertIsNone(approved_model_problem({"server": "http://127.0.0.1:11434", "model": "qwen2.5:3b"}))

    def test_LV36_ollama_mismatch_is_env_error_without_calls(self):
        """LV36 회차의 측정 환경 설정이 묶음 머리줄의 승인 설정과 다르면 실행하지 않고 ENV_ERROR로 기록한다."""
        sc = LV2[0]
        llm = scripted_llm(sc)
        expect = {"server": "http://127.0.0.1:11434", "model": "qwen2.5:3b"}
        with mock.patch.dict(os.environ, {"OLLAMA_MODEL": "other:1b"}):
            obs = run_graph_scenario(sc, llm, live=True, expect_ollama=expect)
        result = graph_indicators(sc, obs)
        self.assertTrue(result["env_error"])
        self.assertEqual(llm.calls, [])
        self.assertIn("Ollama 설정", obs.get("env_reason") or "")

    def test_LV37_replay_checks_args_remaining_errors_and_store(self):
        """LV37 (D5 재현) 재현기: 위치 인자 차이, 남은 기록, 예외 호출의 기록·재현, 초기 저장소 차이를 모두 어긋남으로 남긴다."""
        calls = ReplayCalls([{"args": ["질문A"], "kwargs": {}, "result": {"status": "ok"}},
                             {"args": ["질문B"], "kwargs": {}, "result": {"status": "ok"}}], "rag")
        calls("질문Z", object())
        self.assertTrue(calls.divergence)
        self.assertEqual(calls.remaining(), 1)
        llm = ReplayLLM([{"stage": "understanding", "raw": "{}"}, {"stage": "answer", "raw": "{}"}])
        llm([], {}, {"model": "m"}, stage="understanding")
        self.assertEqual(llm.remaining(), 1)

        def broken(**kwargs):
            raise RuntimeError("예보 서버 오류")
        timed = TimedCall(broken, "weather")
        with self.assertRaises(RuntimeError):
            timed(outing_date="2026-11-26")
        self.assertEqual(timed.calls[0]["error"]["type"], "RuntimeError")
        replay = ReplayCalls(timed.calls, "weather")
        with self.assertRaises(ReplayedError):
            replay(outing_date="2026-11-26")
        sc = next(s for s in LV2 if s["id"] == "LV2-direct-clear")
        obs = run_graph_scenario(sc, scripted_llm(sc))
        tampered = copy.deepcopy(obs)
        tampered["initial_doc"]["schedules"][0]["title"] = "바뀐 제목"
        self.assertFalse(replay_row({"obs": tampered})["reproduced"])
        self.assertTrue(replay_row({"obs": obs})["reproduced"])


class M_LV6_SuiteSafety(OfflineCase):

    def setUp(self):
        super().setUp()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)

    def test_LV38_start_failure_releases_own_lock(self):
        """LV38 (D11 재현) 잠금을 잡은 뒤 로그 생성이 실패하면 연 파일을 닫고 내 잠금을 지운 뒤 오류를 그대로 낸다."""
        suite = LiveSuite("LVX", 1, kinds={}, log_dir=self.dir, live=False)
        real_open = open

        def bad_open(path, *args, **kwargs):
            if "5-6_LVX" in str(path) and str(path).endswith(".txt"):
                raise OSError("로그 생성 실패(시험)")
            return real_open(path, *args, **kwargs)
        with mock.patch("builtins.open", bad_open), self.assertRaises(OSError):
            suite.start()
        self.assertFalse((self.dir / LOCK_NAME).exists())

    def test_LV39_rerun_check_runs_under_lock(self):
        """LV39 (D11) 재측정 사유 검사는 잠금을 잡은 뒤에 한다. 다른 측정이 잠금을 쥐고 있으면 사유와 관계없이 시작하지 않는다."""
        (self.dir / "5-6_LVX_old.jsonl").write_text("{}\n", encoding="utf-8")
        (self.dir / LOCK_NAME).write_text("다른 실행", encoding="utf-8")
        with self.assertRaises(unittest.SkipTest) as caught:
            LiveSuite("LVX", 1, kinds={}, log_dir=self.dir, live=False).start()
        self.assertIn("다른 5-6 측정이 실행 중", str(caught.exception))
        (self.dir / LOCK_NAME).unlink()
        seen = []
        suite = LiveSuite("LVX", 1, kinds={}, log_dir=self.dir, live=False, rerun_reason="잠금 순서 시험")
        original = suite.previous_runs
        with mock.patch.object(suite, "previous_runs", lambda: (seen.append((self.dir / LOCK_NAME).exists()), original())[1]):
            suite.start()
        suite.finish()
        self.assertEqual(seen, [True])
        with self.assertRaises(unittest.SkipTest):
            LiveSuite("LVX", 1, kinds={}, log_dir=self.dir, live=False).start()
        self.assertFalse((self.dir / LOCK_NAME).exists())

    def test_LV40_tool_error_stops_suite_but_is_not_product_s1(self):
        """LV40 (D6) 측정 도구 자체의 예외는 tool_error로 묶음을 멈추고, 제품 S1으로 세지 않는다."""
        suite = LiveSuite("LVX", 3, kinds={}, log_dir=self.dir, live=False)
        suite.start()

        def boom() -> dict[str, Any]:
            raise KeyError("측정 도구 오류 시험")
        rows = suite.run([("a", boom), ("b", lambda: {"indicators": {k: NA for k in INDICATORS}, "s1": []})])
        summary = suite.finish()
        self.assertEqual(rows[0]["s1"], [])
        self.assertTrue(rows[0]["tool_error"])
        self.assertTrue(rows[1]["not_run"])
        self.assertIn("측정 도구 오류", suite.stop_reason or "")
        self.assertEqual(summary["summary"]["s1_runs"], 0)
        self.assertEqual(summary["summary"]["tool_error_runs"], 1)


class I_LV5_SafetyAndWrappers(OfflineCase):

    def test_LV41_change_before_approval_stops_without_approving(self):
        """LV41 (D6 재현) 승인 전에 저장소가 바뀌면 그 턴에서 S1으로 기록하고 승인을 실행하지 않는다(후속 지표 NOT_RUN)."""
        stores: list[ss.ScheduleStore] = []
        original_init, original_send = ss.ScheduleStore.__init__, sg.send_message

        def init(store, *args, **kwargs):
            original_init(store, *args, **kwargs)
            stores.append(store)

        def send(graph, thread, text):
            view = original_send(graph, thread, text)
            if text == LV1_TIME:
                stores[0].transact(uuid.uuid4().hex, "e" * 64, ts.add_schedule("끼어든 일정", ts.at(27, 9), 30))
            return view
        with mock.patch.object(ss.ScheduleStore, "__init__", init), mock.patch.object(sg, "send_message", send):
            result = run_representative("approve", RecordingLLM(ts.ScheduleLLM(understanding=ts.FLOW_A, schedule=ts.FLOW_S)),
                                        child_mode="fake")
        self.assertTrue(any("승인 전" in p for p in result["s1"]), result["s1"])
        self.assertEqual([result["indicators"][k] for k in ("decision_ok", "saved_ok", "requery_ok", "persist_ok")], [NOT_RUN] * 4)
        self.assertFalse(any(t.get("text") == "[approve]" for t in result["obs"]["turns"]))

    def test_LV42_fake_success_message_is_s1(self):
        """LV42 (D6) 실행 함수가 applied=False를 돌려주는데 '저장했어요' 같은 성공 문장이 나오면 S1(가짜 성공)이다."""
        def lying(approved_action, store):
            return {"status": "ok", "message": "", "action_id": approved_action.get("action_id"), "applied": False, "mock": True,
                    "record": None, "answer": "'시장 장보기' 일정을 바꿔 저장했어요."}
        self.assertTrue(fake_success("일정을 바꿔 저장했어요.", {"applied": False}))
        self.assertFalse(fake_success("요청을 처리하지 못했어요. 성공으로 기록하지 않았습니다.", {"applied": False}))
        with mock.patch.object(sv, "apply_schedule_action", lying):
            result = run_representative("approve", RecordingLLM(ts.ScheduleLLM(understanding=ts.FLOW_A, schedule=ts.FLOW_S)),
                                        child_mode="fake")
        self.assertNotEqual(result["indicators"]["decision_ok"], PASS)
        self.assertTrue(any("가짜 성공" in p for p in result["s1"]), (result["s1"], result["obs"]["turns"][-1]))
        self.assertFalse(result["obs"]["changed"])

    def test_LV43_wrong_understanding_in_data_flow_is_not_approved(self):
        """LV43 (D2) LV8-delegate 흐름에 잘못된 이해(operation=other)가 들어오면 승인하지 않고 판정이 PASS가 아니다."""
        spec = LV8_SPECS["LV8-delegate"]
        text = spec["turns"][0]
        llm = RecordingLLM(ts.ScheduleLLM(understanding={text: tod.u(intent="schedule_change")},
                                          schedule={text: {"operation": "other"}}))
        result = run_data_flow(spec, llm, weather_fn=make_weather({"rain": [14]}), rag_fn=tod.FakeRAG(), now=FIXED_NOW,
                               device=FAKE_DEVICE, index=tod.FAKE_INDEX, real_weather=False, real_rag=False, real_llm=False)
        self.assertNotEqual(result["components"]["llm_judgment"], PASS)
        self.assertNotEqual(result["integrated"], PASS)
        self.assertFalse(result["obs"]["changed"])

    def test_LV44_existing_wrapper_classifies_outcomes(self):
        """LV44 (D9) 기존 L층 래퍼: 통과·assert 실패·Ollama 오류·준비 안 됨 건너뜀을 나누고, 확인 가능한 S1만 S1이다.
        운영 저장소 상태를 확인하지 못하면 '미검증'으로 남긴다. unittest 통과를 모델 판단 성공으로 쓰지 않는다."""
        class Dummy(unittest.TestCase):
            def test_pass(self):
                pass

            def test_fail(self):
                self.fail("기대와 다름")

            def test_env(self):
                raise od.DecisionError("llm_unavailable", "connection_refused", "Ollama 연결 실패")

            def test_skip(self):
                self.skipTest("Ollama 서버 또는 모델이 준비되지 않았습니다: {}")
        outcomes = {}
        for name in ("test_pass", "test_fail", "test_env", "test_skip"):
            result = unittest.TestResult()
            Dummy(name).run(result)
            outcomes[name] = classify_existing(result, ops_changed=False, protected_changed=False)
        self.assertEqual([outcomes[n]["existing_test"] for n in ("test_pass", "test_fail", "test_env", "test_skip")],
                         [PASS, FAIL, ENV_ERROR, ENV_ERROR])
        self.assertEqual([outcomes[n]["s1"] for n in outcomes], [[]] * 4)
        self.assertTrue(all(outcomes[n]["indicators"]["judgment_ok"] == NA for n in outcomes))
        result = unittest.TestResult()
        Dummy("test_pass").run(result)
        self.assertTrue(classify_existing(result, ops_changed=True, protected_changed=False)["s1"])
        self.assertEqual(classify_existing(result, ops_changed=None, protected_changed=False)["safety"], "미검증")

    def test_LV45_child_process_calls_are_in_statistics(self):
        """LV45 (D10) 별도 프로세스 재조회의 LLM 호출 단계·시간이 '[별도 프로세스]' 턴으로 기록되어 통계에 들어간다."""
        result = run_representative("approve", RecordingLLM(ts.ScheduleLLM(understanding=ts.FLOW_A, schedule=ts.FLOW_S)),
                                    child_mode="fake")
        child_turns = [t for t in result["obs"]["turns"] if t.get("text") == "[별도 프로세스] " + LV1_AGAIN]
        self.assertEqual(len(child_turns), 1)
        self.assertTrue(child_turns[0]["llm_calls"])
        self.assertIn("cold·warm", child_turns[0].get("unmeasured", ""))

    def test_LV46_lv8_specs_reach_expected_components(self):
        """LV46 LV8·LV8C 사례별 기대값 사전 검증 (가상 LLM·가상 예보·가상 RAG. real_*=False라 실제 검증 성공으로 세지 않음)"""
        for spec, forecast, s_u in ((LV8_SPECS["LV8-delegate"], {"rain": [13, 14]},
                                     {"operation": "update", "target_text": "시장", "delegate_choice": True, "preferences": ["avoid_rain"]}),
                                    (LV8C, LV8C["forecast"], LV8C["s"][LV8C_MSG])):
            with self.subTest(spec["id"]):
                text = spec["turns"][0]
                llm = RecordingLLM(ts.ScheduleLLM(understanding={text: tod.u(intent="schedule_change")}, schedule={text: s_u},
                                                  choices=[{"choice": "C3", "reasons": ["less_rain", "no_conflict"]}]))
                rag = tod.FakeRAG()
                result = run_data_flow(spec, llm, weather_fn=make_weather(forecast), rag_fn=rag, now=FIXED_NOW, device=FAKE_DEVICE,
                                       index=tod.FAKE_INDEX, real_weather=False, real_rag=False, real_llm=False)
                status = result["components"]
                self.assertEqual(result["s1"], [], result["problems"])
                self.assertEqual((status["llm_judgment"], status["proposal"], status["graph_decision"], status["json_change"]),
                                 (PASS, PASS, PASS, PASS), result["problems"])
                self.assertTrue(result["obs"]["changed"])
                if spec.get("rag_required") and spec is LV8C:
                    self.assertTrue(rag.calls)
                    self.assertEqual(status["rag_source"], PASS)


# %% M층·I층. 3차 외부 검토(D14~D20) 회귀: 확인할 수 없는 값·다른 지역·일부 실패·잘못된 저장을 통과로 세지 않는다
# 핵심 요약: 수정 전에는 불완전 예보를 '비 없음'으로, 다른 지역 예보를 카드 지역 예보로, "{}"를 올바른 LLM 출력으로,
# 일부 RAG 실패를 성공으로, 실제 예보 연결 오류를 프로그램 실패로 봤고, 저장 내용은 시각만 비교했으며, 잠금 기록 실패 때 잠금이 남았다.
DRY = dict(tod.BASE_READINGS, missing_fields=[])
WET = dict(DRY, precipitation_mm=5.0, precipitation_probability_pct=90)
NO_RAIN_VALUES = dict(DRY, precipitation_mm=None, precipitation_probability_pct=None,
                      missing_fields=["precipitation_mm", "precipitation_probability_pct"])


def weather_record(city: str | None, hour: int, weather: dict[str, Any] | None, status: str = "ok") -> dict[str, Any]:
    """합성 날씨 호출 기록 (내일 hour시, TimedCall 기록 형식). (D25 보완) 수치가 있으면 요청과 같은 응답 계약 메타데이터를 채운다.
    (수치·missing_fields는 넘긴 그대로라 일부러 깨뜨린 사례는 계속 깨진 채로 남는다)"""
    clock = f"{hour:02d}:00"
    filled = dict(contract_meta(city, "2026-11-26", clock), **weather) if isinstance(weather, dict) else weather
    return {"args": [], "kwargs": {"outing_date": "2026-11-26", "outing_time": clock, "city": city},
            "result": {"status": status, "weather": filled}}


class M_LV7_ThirdReviewJudge(OfflineCase):
    CARD = {"type": "update_schedule", "action_id": "a" * 32, "schedule_id": SID["시장"], "start": iso(26, 15), "end": iso(26, 16),
            "city": "인천", "title": "시장 장보기"}

    def setUp(self):
        super().setUp()
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        self.tmp = Path(folder.name)

    def obs(self, weather_calls: list[dict[str, Any]], card: dict[str, Any] | None = None, **over: Any) -> dict[str, Any]:
        base = {"turns": [{"card": card or self.CARD, "s_understanding": LV8_SPECS["LV8-delegate"]["understanding"], "status": "ok",
                           "llm_error_statuses": [], "s_weather": [], "llm_calls": []}],
                "weather_calls": weather_calls, "rag_calls": [], "llm_calls": [], "initial_doc": ts.seed_doc(),
                "now": FIXED_NOW.isoformat()}
        base.update(over)
        return base

    def test_LV47_incomplete_forecast_is_not_dry(self):
        """LV47 (D14) 강수량·강수확률 누락(ok·partial 표기), 문자열 값, missing_fields 없음, 조회 실패는 '비 없음'이 아니다.
        날씨 조건 판정(judge_components)과 승인 전 검사(rain_at)가 같은 결과다. 완전한 맑음 예보만 PASS"""
        spec = LV8_SPECS["LV8-delegate"]
        self.assertEqual(rain_at(self.obs([weather_record("인천", 15, DRY)]), self.CARD, spec)[0], PASS)
        for name, call in (("누락 partial", weather_record("인천", 15, NO_RAIN_VALUES, "partial")),
                           ("누락 ok 표기", weather_record("인천", 15, NO_RAIN_VALUES)),
                           ("문자열 값", weather_record("인천", 15, dict(DRY, precipitation_mm="0"))),
                           ("missing_fields 없음", weather_record("인천", 15, dict(tod.BASE_READINGS))),
                           ("조회 실패", weather_record("인천", 15, None, "network_error"))):
            with self.subTest(name):
                obs = self.obs([call])
                self.assertIsNone(rain_signal(call["result"]))
                self.assertEqual(rain_at(obs, self.CARD, spec)[0], FAIL)
                judged = judge_components(obs, spec, real_weather=True, real_rag=False, real_llm=True)
                self.assertEqual(judged["status"]["weather_condition"], FAIL, judged["reasons"])

    def test_LV47b_flow_does_not_approve_card_on_incomplete_forecast(self):
        """LV47b (D14, 흐름) 제품이 확인되지 않은 예보 시각을 고르는 결함을 흉내 내도(eligible을 항상 참으로 바꿈)
        측정 도구는 그 카드를 승인하지 않는다(파일 불변, 승인·저장 NOT_RUN)."""
        spec = LV8_SPECS["LV8-delegate"]
        text = spec["turns"][0]
        llm = RecordingLLM(ts.ScheduleLLM(understanding={text: tod.u(intent="schedule_change")},
                                          schedule={text: {"operation": "update", "target_text": "시장", "delegate_choice": True,
                                                           "preferences": ["avoid_rain"]}},
                                          choices=[{"choice": "C1", "reasons": ["no_conflict"]}]))
        weather = tod.FakeWeather(base={"precipitation_mm": None, "precipitation_probability_pct": None})
        with mock.patch.object(sa, "eligible", lambda check, preferences: True):
            result = run_data_flow(spec, llm, weather_fn=weather, rag_fn=tod.FakeRAG(), now=FIXED_NOW, device=FAKE_DEVICE,
                                   index=tod.FAKE_INDEX, real_weather=False, real_rag=False, real_llm=False)
        self.assertTrue(result["obs"]["turns"][0]["card"], result["problems"])
        self.assertFalse(result["obs"]["changed"])
        self.assertEqual((result["components"]["weather_condition"], result["components"]["graph_decision"]), (FAIL, NOT_RUN))
        self.assertTrue(any("날씨 조건" in p for p in result["problems"]), result["problems"])

    def test_LV48_forecast_must_match_date_time_and_city(self):
        """LV48 (D15) 같은 날짜·시각이라도 다른 지역(부산) 예보로 통과하지 않는다. 같은 지역의 상충 기록(맑음·비)은 고르지 않고 FAIL.
        카드 지역이 없으면 시나리오의 예보 지역(인천)을 쓴다."""
        spec = LV8_SPECS["LV8-delegate"]
        cases = [("다른 지역 먼저", [weather_record("부산", 15, DRY), weather_record("인천", 15, WET)], self.CARD, FAIL),
                 ("다른 지역만", [weather_record("부산", 15, DRY)], self.CARD, FAIL),
                 ("같은 지역 상충", [weather_record("인천", 15, DRY), weather_record("인천", 15, WET)], self.CARD, FAIL),
                 ("인천 맑음", [weather_record("부산", 15, WET), weather_record("인천", 15, DRY)], self.CARD, PASS),
                 ("카드 지역 없음", [weather_record("인천", 15, DRY)], dict(self.CARD, city=None), PASS)]
        for name, calls, card, expected in cases:
            with self.subTest(name):
                obs = self.obs(calls, card)
                state, reason, warnings = rain_at(obs, card, spec)
                self.assertEqual(state, expected, reason)
                self.assertEqual(judge_components(obs, spec, real_weather=True, real_rag=False, real_llm=True)["status"]["weather_condition"],
                                 expected)
                if any(c["kwargs"]["city"] == "부산" for c in calls) and expected == PASS:
                    self.assertTrue(warnings)

    def saved_pair(self, operation: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """실제 저장 규칙(apply_schedule_action)으로 만든 (승인 전 문서, 승인 뒤 문서, 카드). 기존 이력 1건이 있는 상태에서 시작한다."""
        folder = Path(tempfile.mkdtemp(dir=self.tmp))
        store = ss.ScheduleStore(folder / "schedule_store.json", clock=lambda: FIXED_NOW)
        store.ensure_initialized()
        warm = sv.propose_create(store.load(), title="미리 등록", start=ts.at(27, 9), minutes=30, now=FIXED_NOW, city=None,
                                 category="outing")
        self.assertEqual(sv.apply_schedule_action(ts.approved(warm["action"]), store)["status"], "ok")
        original = store.load()
        item = {s["schedule_id"]: s for s in original["schedules"]}
        made = {"create": lambda: sv.propose_create(original, title="친구 만나기", start=ts.at(27, 15), minutes=60, now=FIXED_NOW,
                                                    city="인천", category="outing"),
                "update": lambda: sv.propose_update(original, item[SID["시장"]], ts.at(26, 17), FIXED_NOW),
                "cancel": lambda: sv.propose_cancel(item[SID["시장"]], FIXED_NOW),
                "complete": lambda: sv.propose_complete(item[SID["산책"]], FIXED_NOW)}[operation]()
        action = made["action"]
        self.assertEqual(sv.apply_schedule_action(ts.approved(action), store)["status"], "ok")
        return original, store.load(), card_view(action) or {}

    def test_LV49_saved_content_history_and_version_are_checked(self):
        """LV49 (D16) 실제 저장 결과(등록·변경·취소·완료)는 위반 0건(정상 메타데이터 변경을 잘못 판정하지 않음).
        같은 시각이라도 제목·지역·상태·분류·변경 가능 여부·버전·별칭이 다르거나, 이력의 action_id·schedule_id·operation·before·after가
        틀리거나, 이력이 2건 추가되거나, 기존 이력이 바뀌면 각각 위반이다."""
        for operation in ("create", "update", "cancel", "complete"):
            with self.subTest(operation):
                original, saved, card = self.saved_pair(operation)
                self.assertEqual(approval_violations(original, saved, card), [])
        original, saved, card = self.saved_pair("update")
        target = SID["시장"]

        def tampered(change: Callable[[dict[str, Any]], None]) -> list[str]:
            doc = copy.deepcopy(saved)
            change(doc)
            return approval_violations(original, doc, card)

        item_of = (lambda doc: next(s for s in doc["schedules"] if s["schedule_id"] == target))
        changes = {"제목": lambda d: item_of(d).update(title="엉뚱한 제목"), "지역": lambda d: item_of(d).update(city="부산"),
                   "상태": lambda d: item_of(d).update(status="cancelled"), "분류": lambda d: item_of(d).update(category="home"),
                   "변경 가능 여부": lambda d: item_of(d).update(reschedulable=False), "버전": lambda d: item_of(d).update(version=1),
                   "별칭": lambda d: item_of(d).update(aliases=["다른"]),
                   "이력 action_id": lambda d: d["action_history"][-1].update(action_id="b" * 32),
                   "이력 schedule_id": lambda d: d["action_history"][-1].update(schedule_id="schedule_999"),
                   "이력 operation": lambda d: d["action_history"][-1].update(operation="cancel"),
                   "이력 before": lambda d: d["action_history"][-1].update(before=None),
                   "이력 after": lambda d: d["action_history"][-1].update(after=dict(d["action_history"][-1]["after"], title="x")),
                   "이력 2건": lambda d: d["action_history"].append(copy.deepcopy(d["action_history"][-1])),
                   "기존 이력 변조": lambda d: d["action_history"][0].update(reason="바뀜")}
        for name, change in changes.items():
            with self.subTest(name):
                self.assertTrue(tampered(change), name)

    def test_LV50_llm_format_uses_production_schema_and_validators(self):
        """LV50 (D17) JSON 문법과 운영 스키마·운영 검증 함수를 구분한다. "{}"·비객체·자료형·허용값 위반은 실패,
        검증 기준이 없는 단계도 PASS가 아니다. 실제 그래프 흐름의 정상 출력은 형식 문제 0건이다.
        첫 출력 형식 오류 뒤 재요청 성공은 PASS지만 '최초 형식 오류'로 따로 남기고, 최종 출력 오류는 FAIL, 검사 기록이 없으면 NOT_RUN"""
        schema = sa.schedule_schema([])
        self.assertTrue(format_problems("schedule_understanding", "{}", schema))
        self.assertTrue(format_problems("understanding", "{}", od.UNDERSTANDING_SCHEMA))
        self.assertTrue(format_problems("schedule_understanding", "[]", schema))
        self.assertTrue(format_problems("schedule_understanding", "not json", schema))
        sc = LV2[0]
        obs = run_graph_scenario(sc, scripted_llm(sc))
        calls = obs["llm_calls"]
        self.assertTrue(calls)
        for call in calls:
            self.assertEqual(call.get("format"), [], (call["stage"], call.get("format")))
        good = next(json.loads(c["raw"]) for c in calls if c["stage"] == "schedule_understanding")
        self.assertTrue(format_problems("schedule_understanding", json.dumps(dict(good, operation="dance")), schema))
        self.assertTrue(format_problems("schedule_understanding", json.dumps(dict(good, answers_previous_question="yes")), schema))
        self.assertTrue(format_problems("mystery_stage", json.dumps(good), schema))
        spec = LV8_SPECS["LV8-delegate"]

        def judged(turn_calls: list[dict[str, Any]]) -> dict[str, Any]:
            o = self.obs([weather_record("인천", 15, DRY)], llm_calls=turn_calls)
            o["turns"][0]["llm_calls"] = turn_calls
            return judge_components(o, spec, real_weather=True, real_rag=False, real_llm=True)
        retried = judged([{"stage": "schedule_understanding", "format": ["JSON 문법 오류"]}, {"stage": "schedule_understanding", "format": []}])
        self.assertEqual(retried["status"]["qwen_format"], PASS)
        self.assertTrue(any("최초 형식 오류 1건" in w for w in retried["warnings"]), retried["warnings"])
        final_bad = judged([{"stage": "schedule_understanding", "format": []}, {"stage": "answer", "format": ["필수 키 없음"]}])
        self.assertEqual(final_bad["status"]["qwen_format"], FAIL)
        self.assertEqual(judged([{"stage": "schedule_understanding", "raw": "{}"}])["status"]["qwen_format"], NOT_RUN)

    def test_LV51_required_rag_failure_is_not_hidden(self):
        """LV51 (D18) 필수 검색(기대 주제의 공식 질문)과 보조 검색을 나눈다. 필수 정상 + 보조 예외 → PASS와 경고,
        필수 정상 + 필수 예외 → ENV_ERROR, 필수 근거 부족 → FAIL, 필수 검색 없이 보조만 → FAIL"""
        spec = LV8C
        hit = {"filename": sr.STORM, "pdf_page": 1, "similarity": 0.5, "content": "강풍이 불 때는 간판 근처를 피합니다."}
        wind = {"args": [od.QUESTIONS["강풍"]], "kwargs": {}, "result": {"status": "ok", "results": [hit]}}
        # (D26 보완) LV8C의 필수 주제는 강풍뿐이다(호우는 같은 PDF지만 필수 아님). 필수 검색 실패는 강풍 질의의 오류로 시험한다.
        rain_error = {"args": [od.QUESTIONS["강풍"]], "kwargs": {}, "result": None, "error": {"type": "RuntimeError", "message": "boom"}}
        side_error = {"args": [od.QUESTIONS["폭염"]], "kwargs": {}, "result": None, "error": {"type": "RuntimeError", "message": "boom"}}
        side_ok = {"args": [od.QUESTIONS["폭염"]], "kwargs": {}, "result": {"status": "ok", "results": [dict(hit, filename=sr.HEAT)]}}
        thin = dict(wind, result={"status": "insufficient_evidence", "results": []})
        rain_os_error = dict(rain_error, error={"type": "OSError", "message": "모델 캐시를 읽지 못함"})
        # (5-6 최종 보완) 필수 검색의 일반 예외는 원인 미확정 FAIL, 연결·자원 예외(OSError 등)만 ENV_ERROR
        cases = [("보조 실패", [wind, side_error], PASS), ("필수 실패", [wind, rain_error], FAIL),
                 ("필수 환경 실패", [wind, rain_os_error], ENV_ERROR), ("필수 근거 부족", [thin], FAIL), ("보조만", [side_ok], FAIL)]
        for name, calls, expected in cases:
            with self.subTest(name):
                judged = judge_components(self.obs([], rag_calls=calls), spec, real_weather=False, real_rag=False, real_llm=False)
                self.assertEqual(judged["status"]["rag_call"], expected, judged["reasons"].get("rag_call"))
                if name == "보조 실패":
                    self.assertTrue(any("보조 검색" in w for w in judged["warnings"]), judged["warnings"])

    def test_LV52_real_weather_network_error_is_env_error(self):
        """LV52 (D19) 실제 예보 묶음에서 필수 호출의 network_error·api_error·연결 예외는 회차 환경 오류(연속 2회면 묶음 중단).
        예보 범위 밖·입력 부족·외부 응답 형식·원인 모를 예외는 환경 오류로 합치지 않는다. 가상 예보의 network_error 시나리오도 환경 오류가 아니다."""
        kinds = {"network_error": "env", "api_error": "unknown", "out_of_range": "range", "needs_input": "input",
                 "data_unavailable": "data", "ok": None, "partial": None}
        for status, kind in kinds.items():
            with self.subTest(status):
                # (D25 보완) partial 응답은 계약상 빠진 항목이 있어야 하므로 누락 예보를 쓴다(성공 상태 분류 의도는 그대로)
                call = weather_record("인천", 15, {"ok": DRY, "partial": NO_RAIN_VALUES}.get(status), status)
                self.assertEqual(weather_failure(call), kind)
                self.assertEqual(env_error_in({"turns": [], "weather_calls": [call]}, required_weather=[call]), kind == "env")
        for error, kind in (("URLError", "env"), ("TimeoutError", "env"), ("KeyError", "unknown")):
            with self.subTest(error):
                call = dict(weather_record("인천", 15, None), result=None, error={"type": error, "message": "x"})
                self.assertEqual(weather_failure(call), kind)
        self.assertFalse(env_error_in({"turns": [], "weather_calls": [weather_record("인천", 15, None, "network_error")]}))
        spec = LV8_SPECS["LV8-delegate"]
        text = spec["turns"][0]

        def flow(real_weather: bool) -> dict[str, Any]:
            llm = RecordingLLM(ts.ScheduleLLM(understanding={text: tod.u(intent="schedule_change")},
                                              schedule={text: {"operation": "update", "target_text": "시장", "delegate_choice": True,
                                                               "preferences": ["avoid_rain"]}}))
            return run_data_flow(spec, llm, weather_fn=make_weather({"fail": "network_error"}), rag_fn=tod.FakeRAG(), now=FIXED_NOW,
                                 device=FAKE_DEVICE, index=tod.FAKE_INDEX, real_weather=real_weather, real_rag=False, real_llm=False)
        real = flow(True)
        self.assertEqual((real["env_error"], real["components"]["weather_real"]), (True, ENV_ERROR), real["component_reasons"])
        self.assertEqual(flow(False)["env_error"], False)
        suite = LiveSuite("lv52", 3, kinds={}, log_dir=self.tmp, live=False)
        suite.start()
        try:
            rows = suite.run([(f"r{n}", lambda: flow(True)) for n in range(3)])
        finally:
            suite.finish()
        self.assertEqual([r.get("not_run", False) for r in rows], [False, False, True])
        self.assertIn("환경 오류 연속", suite.stop_reason or "")

    def test_LV54_api_and_search_errors_need_environment_evidence(self):
        """LV54 (5-6 최종 보완, D19) api_error는 HTTP 429·5xx처럼 외부 서비스 문제라는 근거가 있을 때만 환경 오류다.
        HTTP 4xx·'요청을 거절'·메시지 없음은 요청 인자(프로그램) 문제일 수 있어 원인 미확정(FAIL, 사유 표시).
        RAG search_error·일반 예외도 근거가 없어 원인 미확정 FAIL이고, 연결·자원 예외(OSError·메모리 등)만 ENV_ERROR"""
        spec = LV8_SPECS["LV8-delegate"]
        for message, kind in (("호출 한도에 도달했습니다(HTTP 429). 잠시 후 다시 시도해 주세요.", "env"),
                              ("기상 API가 요청을 처리하지 못했습니다(HTTP 503).", "env"),
                              ("기상 API가 요청을 처리하지 못했습니다(HTTP 400).", "unknown"),
                              ("기상 API가 요청을 거절했습니다.", "unknown"), (None, "unknown")):
            with self.subTest(message):
                call = weather_record("인천", 15, None, "api_error")
                if message:
                    call["result"]["message"] = message
                self.assertEqual(weather_failure(call), kind)
                state, reason, _ = judge_weather(self.obs([call]), spec, self.CARD, FIXED_NOW, True)
                self.assertEqual(state, ENV_ERROR if kind == "env" else FAIL, reason)
                if kind != "env":
                    self.assertIn("원인 미확정", reason)
                self.assertEqual(env_error_in({"turns": [], "weather_calls": [call]}, required_weather=[call]), kind == "env")
        query = od.QUESTIONS["강풍"]
        for name, call, expected in (("search_error", {"args": [query], "kwargs": {}, "result": {"status": "search_error", "results": []}}, FAIL),
                                     ("일반 예외", {"args": [query], "kwargs": {}, "result": None,
                                                 "error": {"type": "KeyError", "message": "x"}}, FAIL),
                                     ("자원 예외", {"args": [query], "kwargs": {}, "result": None,
                                                 "error": {"type": "MemoryError", "message": "x"}}, ENV_ERROR)):
            with self.subTest(name):
                state, reason = judge_rag_calls([call])
                self.assertEqual(state, expected, reason)
                if expected == FAIL:
                    self.assertIn("원인 미확정", reason)

    def test_LV53_lock_setup_failure_closes_fd_and_removes_only_own_lock(self):
        """LV53 (D20) 잠금 생성 뒤 파일 열기(fdopen)·내용 기록·로그 생성이 실패하면 fd를 닫고 이번 실행이 만든 잠금만 지운다.
        다른 실행의 잠금은 그대로 두고 건너뛴다(강제 해제 없음)."""
        lock = self.tmp / LOCK_NAME
        opened: list[int] = []
        real_open, real_fdopen = os.open, os.fdopen

        def recording_open(*args, **kwargs):
            fd = real_open(*args, **kwargs)
            opened.append(fd)
            return fd

        class FailingWrite:
            def __init__(self, fd, *args, **kwargs):
                self.handle = real_fdopen(fd, *args, **kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.handle.close()

            def write(self, text):
                raise OSError("기록 실패(가상)")

        def failing_log_open(path, *args, **kwargs):
            if str(path).endswith(".txt"):
                raise OSError("로그 생성 실패(가상)")
            return io.open(path, *args, **kwargs)

        cases = [("fdopen 실패", mock.patch("os.fdopen", side_effect=OSError("fdopen 실패(가상)"))),
                 ("내용 기록 실패", mock.patch("os.fdopen", FailingWrite)),
                 ("로그 생성 실패", mock.patch.dict(globals(), {"open": failing_log_open}))]
        for name, patch in cases:
            with self.subTest(name):
                opened.clear()
                suite = LiveSuite("lv53", 1, kinds={}, log_dir=self.tmp, live=False)
                with mock.patch("os.open", recording_open), patch:
                    with self.assertRaises(OSError):
                        suite.start()
                self.assertFalse(lock.exists(), name)
                self.assertTrue(opened)
                with self.assertRaises(OSError):
                    os.fstat(opened[0])          # 잠금 fd가 닫혔는지
        lock.write_text("다른 실행의 잠금", encoding="utf-8")
        with self.assertRaises(unittest.SkipTest):
            LiveSuite("lv53", 1, kinds={}, log_dir=self.tmp, live=False).start()
        self.assertEqual(lock.read_text(encoding="utf-8"), "다른 실행의 잠금")
        lock.unlink()


# %% M층·I층. 최종 외부 검토(D21·D23) 회귀: 카드 전체 시간 구간의 예보와 검증 가능한 LLM 출력이 있어야 통과
# 핵심 요약: 수정 전에는 카드 시작 시각의 기록 하나로 날씨 조건을 PASS로 봤고(14:30~15:30 카드의 15시를 보지 않음),
# 모든 LLM 호출이 실패하거나 필수 단계의 출력이 없어도 형식 판정이 PASS였다.
def span_record(day: str, clock: str, weather: dict[str, Any] | None, city: str = "인천", status: str = "ok") -> dict[str, Any]:
    """합성 날씨 호출 기록 (날짜·시각 지정). (D25 보완) 수치가 있으면 요청과 같은 응답 계약 메타데이터를 채운다."""
    filled = dict(contract_meta(city, day, clock), **weather) if isinstance(weather, dict) else weather
    return {"args": [], "kwargs": {"outing_date": day, "outing_time": clock, "city": city}, "result": {"status": status, "weather": filled}}


class M_LV8_WholeSpanAndFormat(OfflineCase):
    CARD = {"type": "create_schedule", "action_id": "a" * 32, "schedule_id": None, "start": "2026-11-26T14:30:00+09:00",
            "end": "2026-11-26T15:30:00+09:00", "city": "인천", "title": "산책"}
    SPEC = {"avoid_rain": True, "weather": {"city": "인천", "day": 1}}

    def test_LV55_rain_check_covers_every_hour_of_card(self):
        """LV55 (D21) 카드 [시작, 종료)와 겹치는 모든 정시 구간의 같은 지역 기록이 확인 가능한 '비 없음'일 때만 PASS.
        14:30 기록만 → FAIL(15시 없음), 15시 비 → FAIL, 14:00 기록이 14시 구간을 대신함 → PASS, 15시가 다른 지역뿐 → FAIL,
        자정을 넘는 23:30~00:30은 이틀의 두 구간이 모두 있어야 PASS"""
        d = "2026-11-26"
        cases = [("14:30만", [span_record(d, "14:30", DRY)], self.CARD, FAIL),
                 ("두 구간 맑음", [span_record(d, "14:30", DRY), span_record(d, "15:00", DRY)], self.CARD, PASS),
                 ("15시 비", [span_record(d, "14:30", DRY), span_record(d, "15:00", WET)], self.CARD, FAIL),
                 ("14시 정시 기록", [span_record(d, "14:00", DRY), span_record(d, "15:00", DRY)], self.CARD, PASS),
                 ("15시 다른 지역", [span_record(d, "14:30", DRY), span_record(d, "15:00", DRY, city="부산")], self.CARD, FAIL),
                 ("15시 누락", [span_record(d, "14:30", DRY), span_record(d, "15:00", NO_RAIN_VALUES)], self.CARD, FAIL)]
        midnight = dict(self.CARD, start="2026-11-25T23:30:00+09:00", end="2026-11-26T00:30:00+09:00")
        cases += [("자정 두 구간", [span_record("2026-11-25", "23:30", DRY), span_record(d, "00:00", DRY)], midnight, PASS),
                  ("자정 첫 구간만", [span_record("2026-11-25", "23:30", DRY)], midnight, FAIL)]
        for name, calls, card, expected in cases:
            with self.subTest(name):
                state, reason, _ = rain_at({"weather_calls": calls}, card, self.SPEC)
                self.assertEqual(state, expected, reason)

    def test_LV56_required_weather_calls_cover_every_hour(self):
        """LV56 (D21) 실제 예보 판정의 필수 호출도 카드 전체 구간이다. 두 번째 구간 기록이 없으면 FAIL, 두 번째 구간 network_error면 ENV_ERROR"""
        d = "2026-11-26"
        obs = {"weather_calls": [span_record(d, "14:30", DRY)], "turns": []}
        state, reason, _ = judge_weather(obs, self.SPEC, self.CARD, FIXED_NOW, True)
        self.assertEqual(state, FAIL, reason)
        self.assertIn("15", reason)
        obs = {"weather_calls": [span_record(d, "14:30", DRY), span_record(d, "15:00", None, status="network_error")], "turns": []}
        self.assertEqual(judge_weather(obs, self.SPEC, self.CARD, FIXED_NOW, True)[0], ENV_ERROR)
        obs = {"weather_calls": [span_record(d, "14:30", DRY), span_record(d, "15:00", DRY)], "turns": []}
        self.assertEqual(judge_weather(obs, self.SPEC, self.CARD, FIXED_NOW, True)[0], PASS)

    def test_LV57_format_needs_verifiable_output_for_every_called_stage(self):
        """LV57 (D23) 모든 호출 실패 → NOT_RUN. 형식 오류 뒤 재요청 성공 → PASS와 '최초 형식 오류'.
        정상 출력 뒤 다른 필수 단계의 호출 실패 → NOT_RUN(그 단계 출력 없음, 전체 검증 미완료). 호출 성공(qwen_call)과는 따로 판정하고 종합 성공 아님"""
        # (D33 보완) 실제 기록기(RecordingLLM)처럼 오류 코드를 남긴다. 환경 오류 근거(연결 거부)가 있어야 qwen_call ENV_ERROR
        failed = {"stage": "answer", "error": {"type": "DecisionError", "status": "llm_unavailable", "code": "connection_refused"}}
        cases = [("모든 호출 실패", [dict(failed, stage="schedule_understanding"), failed], NOT_RUN, "출력"),
                 ("재요청 성공", [{"stage": "understanding", "format": ["JSON 문법 오류"]}, {"stage": "understanding", "format": []}], PASS,
                  "최초 형식 오류"),
                 ("정상 뒤 다른 단계 실패", [{"stage": "understanding", "format": []}, failed], NOT_RUN, "answer")]
        for name, calls, expected, words in cases:
            with self.subTest(name):
                state, reason, warnings = judge_format([{"llm_calls": calls}], calls)
                self.assertEqual(state, expected, reason)
                self.assertIn(words, reason + " ".join(warnings))
        calls = [{"stage": "understanding", "format": []}, failed]
        obs = {"turns": [{"llm_calls": calls, "card": None, "s_weather": []}], "llm_calls": calls, "weather_calls": [], "rag_calls": [],
               "now": FIXED_NOW.isoformat()}
        status = judge_components(obs, LV8_SPECS["LV8-delegate"], real_weather=False, real_rag=False, real_llm=True)["status"]
        self.assertEqual((status["qwen_call"], status["qwen_format"]), (ENV_ERROR, NOT_RUN))
        self.assertNotEqual(integrated_success(status, required_components(LV8_SPECS["LV8-delegate"], real_weather=False)), PASS)

    def test_LV58_flow_judges_and_approves_only_whole_span_clear_cards(self):
        """LV58 (D21, 흐름) LV8-delegate 흐름에서 15시 비: 제품은 15시와 겹치지 않는 카드를 만들고 측정 도구는 전체 구간 기록으로 PASS 판정 뒤 승인한다.
        제품이 시작 구간만 보는 결함을 흉내 내면(span_hours를 시작 시각 하나로 바꿈) 14:30~15:30 카드가 나오지만 측정 도구는 승인하지 않는다(파일 불변)."""
        spec = LV8_SPECS["LV8-delegate"]
        text = spec["turns"][0]

        def flow() -> dict[str, Any]:
            llm = RecordingLLM(ts.ScheduleLLM(understanding={text: tod.u(intent="schedule_change")},
                                              schedule={text: {"operation": "update", "target_text": "시장", "delegate_choice": True,
                                                               "preferences": ["avoid_rain"]}},
                                              choices=[{"choice": "C1", "reasons": ["no_conflict"]}]))
            return run_data_flow(spec, llm, weather_fn=make_weather({"rain": [15]}), rag_fn=tod.FakeRAG(), now=FIXED_NOW,
                                 device=FAKE_DEVICE, index=tod.FAKE_INDEX, real_weather=False, real_rag=False, real_llm=False)
        fixed = flow()
        card = fixed["obs"]["turns"][0]["card"]
        start, end = ss.parse_iso(card["start"]), ss.parse_iso(card["end"])
        assert start is not None and end is not None
        rain_hour = datetime.fromisoformat("2026-11-26T15:00:00+09:00")
        self.assertFalse(start < rain_hour + timedelta(hours=1) and rain_hour < end, card)      # 카드가 15시 구간과 겹치지 않음
        self.assertEqual((fixed["components"]["weather_condition"], fixed["components"]["json_change"]), (PASS, PASS),
                         fixed["component_reasons"])
        with mock.patch.object(sa, "span_hours", lambda start, end: [start]):
            broken = flow()
        self.assertEqual(broken["obs"]["turns"][0]["card"]["start"], "2026-11-26T14:30:00+09:00")
        self.assertFalse(broken["obs"]["changed"])
        self.assertEqual((broken["components"]["weather_condition"], broken["components"]["graph_decision"]), (FAIL, NOT_RUN))


# %% M층·I층. 실측 전 필수 보완(D25·D26) 회귀: 예보 응답 계약과 필수 RAG 주제
# 핵심 요약: 수정 전에는 호출 인자만 보고 응답이 다른 지역·다른 구간이어도 날씨 판정을 PASS로 봤고,
# 필수 주제 중 일부만 검색해도 RAG 호출을 PASS로 봤다. 기대 주제와 기대 출처 파일도 섞여 있었다.
def rag_record(topic: str, filename: str | None = None, status: str = "ok", error: str | None = None) -> dict[str, Any]:
    """합성 RAG 호출 기록 (운영 코드처럼 공식 질문으로 시작하는 질의)"""
    if error:
        return {"args": [od.QUESTIONS[topic]], "kwargs": {}, "result": None, "error": {"type": error, "message": "x"}}
    results = [{"filename": filename or TOPIC_FILE[topic], "pdf_page": 1, "similarity": 0.5, "content": "공식 수칙"}] if status == "ok" else []
    return {"args": [od.QUESTIONS[topic]], "kwargs": {}, "result": {"status": status, "results": results}}


class M_LV9_ContractAndTopics(OfflineCase):
    CARD = {"type": "create_schedule", "action_id": "a" * 32, "schedule_id": None, "start": "2026-11-26T14:30:00+09:00",
            "end": "2026-11-26T15:30:00+09:00", "city": "인천", "title": "산책"}
    SPEC = {"avoid_rain": True, "weather": {"city": "인천", "day": 1}}

    def test_LV59_weather_response_must_match_request_contract(self):
        """LV59 (D25) 운영 read_weather와 같은 응답 계약을 측정 도구도 독립적으로 본다. 인천 14:30·15:00 요청에 부산·18시 응답 → rain_at·judge_weather FAIL
        (환경 오류 아님, 원인 미확정). 지역·outing_at·hour_start·hour_end·시간대·출처·data_kind·missing_fields·status를 하나씩 어긋나게 해도 FAIL.
        계약에 맞으면 PASS, network_error는 계속 ENV_ERROR"""
        d = "2026-11-26"
        good = [span_record(d, "14:30", DRY), span_record(d, "15:00", DRY)]
        self.assertEqual(rain_at({"weather_calls": good}, self.CARD, self.SPEC)[0], PASS)
        self.assertEqual(judge_weather({"weather_calls": good, "turns": []}, self.SPEC, self.CARD, FIXED_NOW, True)[0], PASS)
        busan = dict(DRY, **contract_meta("부산", d, "18:00"))
        mismatch = [{"args": [], "kwargs": {"outing_date": d, "outing_time": clock, "city": "인천"},
                     "result": {"status": "ok", "message": "", "weather": dict(busan)}} for clock in ("14:30", "15:00")]
        obs = {"weather_calls": mismatch, "turns": []}
        self.assertEqual(rain_at(obs, self.CARD, self.SPEC)[0], FAIL)
        state, reason, _ = judge_weather(obs, self.SPEC, self.CARD, FIXED_NOW, True)
        self.assertEqual(state, FAIL, reason)
        self.assertIn("원인 미확정", reason)
        self.assertEqual(weather_failure(mismatch[0]), "contract")
        self.assertFalse(env_error_in(obs, required_weather=mismatch))
        changes = {"지역": {"city": "부산"}, "outing_at": {"outing_at": "2026-11-26T14:00+09:00"},
                   "hour_start": {"hour_start": "2026-11-26T15:00+09:00"}, "hour_end": {"hour_end": "2026-11-26T16:00+09:00"},
                   "시간대": {"outing_at": "2026-11-26T14:30+00:00"}, "출처": {"source": "다른 출처"}, "data_kind": {"data_kind": "history"},
                   "missing_fields": {"missing_fields": ["precipitation_mm"]}}
        for name, change in changes.items():
            with self.subTest(name):
                broken = copy.deepcopy(good)
                broken[0]["result"]["weather"].update(change)
                self.assertTrue(weather_contract_problems(broken[0]), name)
                self.assertEqual(rain_at({"weather_calls": broken}, self.CARD, self.SPEC)[0], FAIL)
                self.assertEqual(judge_weather({"weather_calls": broken, "turns": []}, self.SPEC, self.CARD, FIXED_NOW, True)[0], FAIL)
        with self.subTest("status"):
            broken = copy.deepcopy(good)
            broken[0]["result"]["status"] = "partial"
            self.assertEqual(rain_at({"weather_calls": broken}, self.CARD, self.SPEC)[0], FAIL)
        network = [good[0], span_record(d, "15:00", None, status="network_error")]
        self.assertEqual(judge_weather({"weather_calls": network, "turns": []}, self.SPEC, self.CARD, FIXED_NOW, True)[0], ENV_ERROR)

    def test_LV60_flow_with_mismatched_response_is_not_approved(self):
        """LV60 (D25, 흐름) 1단계 응답의 지역이 요청과 다르면 운영 코드는 그 예보를 쓰지 않아(data_unavailable) 카드를 만들지 않고,
        측정 도구는 weather_real을 환경 오류가 아닌 FAIL(응답 계약 불일치)로 판정한다. 파일은 바뀌지 않는다."""
        spec = LV8_SPECS["LV8-delegate"]
        text = spec["turns"][0]
        fake = tod.FakeWeather()

        def wrong_city(outing_date=None, outing_time=None, city=None):
            reply = fake(outing_date=outing_date, outing_time=outing_time, city=city)
            if isinstance(reply.get("weather"), dict):
                reply["weather"]["city"] = "부산"
            return reply
        llm = RecordingLLM(ts.ScheduleLLM(understanding={text: tod.u(intent="schedule_change")},
                                          schedule={text: {"operation": "update", "target_text": "시장", "delegate_choice": True,
                                                           "preferences": ["avoid_rain"]}}))
        result = run_data_flow(spec, llm, weather_fn=wrong_city, rag_fn=tod.FakeRAG(), now=FIXED_NOW, device=FAKE_DEVICE,
                               index=tod.FAKE_INDEX, real_weather=True, real_rag=False, real_llm=False)
        self.assertIsNone(result["obs"]["turns"][0]["card"])
        self.assertFalse(result["obs"]["changed"])
        self.assertEqual(result["components"]["weather_real"], FAIL, result["component_reasons"])
        self.assertIn("계약", result["component_reasons"]["weather_real"])
        self.assertFalse(result["env_error"])

    def test_LV61_every_required_rag_topic_must_be_searched(self):
        """LV61 (D26) 기대 주제마다 그 주제의 공식 질문 검색이 정상이어야 PASS. 기대 주제와 기대 출처 파일은 따로 본다.
        LV8-safety(필수 한파, 선택 대설): 한파만 PASS, 대설만 FAIL, 둘 다 PASS, 대설 검색 실패 FAIL, 다른 주제 실패는 경고만.
        LV8C(필수 강풍, 같은 풍수해 PDF의 호우는 필수 아님): 강풍 PASS, 호우만 FAIL, 강풍 + 호우 실패 PASS와 경고.
        주제 목록을 넘기면 {한파, 대설} 중 한파만 검색 → FAIL(대설 누락)"""
        def rag_status(spec: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
            obs = {"turns": [{"a_route": "safety_guidance", "llm_calls": []}], "rag_calls": calls, "weather_calls": [], "llm_calls": [],
                   "now": FIXED_NOW.isoformat()}
            return judge_components(obs, spec, real_weather=False, real_rag=False, real_llm=False)
        safety = LV8_SPECS["LV8-safety"]
        cases = [("한파만", safety, [rag_record("한파")], PASS), ("대설만", safety, [rag_record("대설")], FAIL),
                 ("한파·대설", safety, [rag_record("한파"), rag_record("대설")], PASS),
                 ("대설 실패", safety, [rag_record("한파"), rag_record("대설", error="KeyError")], FAIL),
                 ("다른 주제 실패", safety, [rag_record("한파"), rag_record("폭염", error="KeyError")], PASS),
                 ("LV8C 강풍", LV8C, [rag_record("강풍")], PASS), ("LV8C 호우만", LV8C, [rag_record("호우")], FAIL),
                 ("LV8C 강풍+호우 실패", LV8C, [rag_record("강풍"), rag_record("호우", error="KeyError")], PASS)]
        for name, spec, calls, expected in cases:
            with self.subTest(name):
                judged = rag_status(spec, calls)
                self.assertEqual(judged["status"]["rag_call"], expected, judged["reasons"].get("rag_call"))
                if name in ("다른 주제 실패", "LV8C 강풍+호우 실패"):
                    self.assertTrue(any("보조 검색" in w for w in judged["warnings"]), judged["warnings"])
        self.assertEqual(rag_status(LV8C, [rag_record("강풍", filename=sr.HEAT)])["status"]["rag_source"], FAIL)
        self.assertEqual(rag_status(safety, [rag_record("한파")])["status"]["rag_source"], PASS)
        state, reason = judge_rag_calls([rag_record("한파")], {"한파", "대설"})
        self.assertEqual(state, FAIL, reason)
        self.assertIn("대설", reason)


# %% M층. 최종 잔여 결함(D28~D30) 회귀: 예보 필수 키, 시각 지정 질문의 조회 구간, 주제별 RAG 출처
# 핵심 요약: 수정 전에는 필수 예보 항목 키가 아예 없는 응답, 질문한 시각과 다른 구간만 조회한 기록,
# 주제와 출처 PDF가 서로 뒤바뀐 검색을 통과로 판정했다.
class M_LV10_KeysHoursAndSources(OfflineCase):

    def test_LV62_missing_forecast_key_breaks_contract(self):
        """LV62 (D28) 필수 예보 항목 5개는 키가 있어야 한다(값 None은 정상 partial). precipitation_mm 키를 지우고 missing_fields에 넣은
        partial 응답 → 계약 불일치, judge_weather·rain_at FAIL. 키가 있고 값이 None인 partial은 계약에 맞고 호출 성공(weather_real PASS)"""
        spec = {"avoid_rain": True, "weather": {"city": "인천", "day": 1}}
        card = {"type": "create_schedule", "start": "2026-11-26T15:00:00+09:00", "end": "2026-11-26T16:00:00+09:00", "city": "인천"}
        partial = span_record("2026-11-26", "15:00", dict(DRY, precipitation_mm=None, missing_fields=["precipitation_mm"]),
                              status="partial")
        self.assertEqual(weather_contract_problems(partial), [])
        self.assertEqual(judge_weather({"weather_calls": [partial], "turns": []}, spec, card, FIXED_NOW, True)[0], PASS)
        dropped = copy.deepcopy(partial)
        del dropped["result"]["weather"]["precipitation_mm"]
        self.assertTrue(weather_contract_problems(dropped))
        state, reason, _ = judge_weather({"weather_calls": [dropped], "turns": []}, spec, card, FIXED_NOW, True)
        self.assertEqual(state, FAIL, reason)
        self.assertEqual(rain_at({"weather_calls": [dropped]}, card, spec)[0], FAIL)

    def test_LV63_time_specific_question_needs_that_hour(self):
        """LV63 (D29) 시각을 말한 LV8-outing(내일 오후 2시)은 14시 구간 조회가 있어야 한다: 18시만 조회 → FAIL, 14:00 조회 → PASS,
        14시 network_error → ENV_ERROR. 날짜만 말한 LV8-weather는 기존처럼 그날 인천 조회면 된다(18시만 조회해도 PASS)"""
        outing, weather_q = LV8_SPECS["LV8-outing"], LV8_SPECS["LV8-weather"]
        evening = span_record("2026-11-26", "18:00", DRY)
        state, reason, _ = judge_weather({"weather_calls": [evening], "turns": []}, outing, None, FIXED_NOW, True)
        self.assertEqual(state, FAIL, reason)
        self.assertIn("14", reason)
        afternoon = span_record("2026-11-26", "14:00", DRY)
        self.assertEqual(judge_weather({"weather_calls": [afternoon], "turns": []}, outing, None, FIXED_NOW, True)[0], PASS)
        failed = span_record("2026-11-26", "14:00", None, status="network_error")
        self.assertEqual(judge_weather({"weather_calls": [failed], "turns": []}, outing, None, FIXED_NOW, True)[0], ENV_ERROR)
        self.assertEqual(judge_weather({"weather_calls": [evening], "turns": []}, weather_q, None, FIXED_NOW, True)[0], PASS)

    def test_LV65_llm_call_errors_need_environment_evidence(self):
        """LV65 (D33) qwen_call: 연결 실패·시간 초과·호스트 없음·모델 미설치·설정·HTTP 5xx(llm_unavailable)만 ENV_ERROR,
        llm_invalid_output·HTTP 4xx·KeyError 같은 원인 미확정 예외는 FAIL, 호출이 없으면 NOT_RUN, 오류가 없으면 PASS.
        환경 오류와 다른 오류가 함께 있으면 FAIL이고 사유에 둘 다 남는다. qwen_format·llm_judgment와는 따로 판정한다."""
        def qwen_call(calls: list[dict[str, Any]]) -> tuple[str, str]:
            obs = {"turns": [{"llm_calls": calls}], "llm_calls": calls, "weather_calls": [], "rag_calls": [], "now": FIXED_NOW.isoformat()}
            judged = judge_components(obs, LV8_SPECS["LV8-delegate"], real_weather=False, real_rag=False, real_llm=True)
            return judged["status"]["qwen_call"], judged["reasons"].get("qwen_call", "")

        def err(status: str | None, code: str | None, kind: str = "DecisionError") -> dict[str, Any]:
            return {"stage": "understanding", "error": {"type": kind, "status": status, "code": code, "message": "x"}}
        ok = {"stage": "understanding", "format": []}
        for code in ("connection_refused", "connection_error", "host_not_found", "model_missing", "configuration", "inference_timeout",
                     "http_503"):
            with self.subTest(code):
                self.assertEqual(qwen_call([ok, err("llm_unavailable", code)])[0], ENV_ERROR)
        for name, call in (("형식 오류", err("llm_invalid_output", "response_format")), ("HTTP 400", err("llm_unavailable", "http_400")),
                           ("KeyError", err(None, None, "KeyError"))):
            with self.subTest(name):
                self.assertEqual(qwen_call([ok, call])[0], FAIL)
        self.assertEqual(qwen_call([])[0], NOT_RUN)
        self.assertEqual(qwen_call([ok])[0], PASS)
        state, reason = qwen_call([err("llm_unavailable", "connection_refused"), err(None, None, "KeyError")])
        self.assertEqual(state, FAIL)
        self.assertTrue("환경" in reason and "KeyError" in reason, reason)

    def test_LV64_each_topic_source_matches_its_own_file(self):
        """LV64 (D30) 필수 검색과 검색된 선택 주제는 주제마다 그 주제의 공식 PDF(TOPIC_FILE)가 상위 출처여야 한다.
        필수 한파·강풍인데 한파→풍수해, 강풍→겨울철 PDF로 뒤바뀌면 rag_source FAIL(rag_call은 PASS, 본문은 REVIEW 유지).
        LV8-safety 선택 대설이 풍수해 PDF면 FAIL. 기대 주제 밖 보조 검색의 출처는 판정에 쓰지 않는다.
        주제를 알 수 없는 질의는 기존처럼 기대 출처 집합(rag_file)과 비교한다(호환)."""
        def status_of(spec: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
            obs = {"turns": [{"llm_calls": []}], "rag_calls": calls, "weather_calls": [], "llm_calls": [], "now": FIXED_NOW.isoformat()}
            return judge_components(obs, spec, real_weather=False, real_rag=False, real_llm=False)["status"]
        both = {"rag_required": True, "rag_topics": ["한파", "강풍"]}
        swapped = status_of(both, [rag_record("한파", filename=sr.STORM), rag_record("강풍", filename=sr.WINTER)])
        self.assertEqual((swapped["rag_call"], swapped["rag_source"], swapped["rag_content"]), (PASS, FAIL, REVIEW))
        right = status_of(both, [rag_record("한파"), rag_record("강풍")])
        self.assertEqual((right["rag_call"], right["rag_source"]), (PASS, PASS))
        safety = LV8_SPECS["LV8-safety"]
        self.assertEqual(status_of(safety, [rag_record("한파"), rag_record("대설", filename=sr.STORM)])["rag_source"], FAIL)
        self.assertEqual(status_of(safety, [rag_record("한파"), rag_record("폭염", filename=sr.STORM)])["rag_source"], PASS)
        unknown_query = {"args": ["바람 많이 부는 날 조심할 점"], "kwargs": {},
                         "result": {"status": "ok", "results": [{"filename": sr.STORM, "pdf_page": 1, "similarity": 0.5, "content": "x"}]}}
        self.assertEqual(status_of({"rag_required": True, "rag_file": sr.STORM}, [unknown_query])["rag_source"], PASS)
        self.assertEqual(status_of({"rag_required": True, "rag_file": sr.WINTER}, [unknown_query])["rag_source"], FAIL)


# %% L층. 실제 측정 (옵션이 없으면 건너뜀. 묶음마다 하나씩 실행)
# 핵심 요약: 묶음 실행기로 고정 횟수를 실행하고 모든 회차를 기록한다. assert는 S1 안전 불변식에만 건다.
# 묶음 시작 때 기록한 승인 Ollama 설정(서버·모델)을 회차마다 측정 환경과 비교한다(다르면 ENV_ERROR).
class LiveCase(unittest.TestCase):
    needs_real = False
    needs_ollama = True

    def setUp(self):
        if self.needs_ollama and not OPTIONS.live_ollama:
            raise unittest.SkipTest("--live-ollama 를 지정하면 실행합니다. (ollama serve와 qwen2.5:3b 필요)")
        if self.needs_real and not OPTIONS.live_real:
            raise unittest.SkipTest("--live-real 를 함께 지정하면 실행합니다. (인터넷·실제 MiniLM 필요)")

    def execute(self, name: str, units_fn: Callable[[dict[str, Any]], list], kinds: dict[str, str], *,
                per_run: float = MAX_OBSERVED_RUN_SECONDS, extra: dict[str, Any] | None = None,
                precondition: Callable[[dict[str, Any]], Any] | None = None) -> dict[str, Any]:
        """units_fn(ctx)는 회차 목록을 만든다. 회차는 실행할 때 ctx(승인 Ollama 설정, 준비한 장치·색인)를 읽는다."""
        ctx: dict[str, Any] = {}
        units = units_fn(ctx)
        suite = LiveSuite(name, len(units), kinds=kinds, per_run_seconds=per_run, rerun_reason=OPTIONS.rerun_reason)
        header = {"ollama_before": ollama_snapshot() if self.needs_ollama else {"note": "LLM을 쓰지 않는 묶음"}, **(extra or {})}
        suite.start(extra_header=header)
        self.addCleanup(suite.close)
        ctx["expect_ollama"] = suite.header["ollama_settings"] if self.needs_ollama else None

        def check() -> Any:
            problem = ollama_ready() if self.needs_ollama else None
            if problem:
                return problem
            return precondition(ctx) if precondition else None
        before = suite.header["ollama_before"].get("before") or {}
        rows = suite.run(units, precondition=check)
        summary = suite.finish({"ollama_after": ollama_snapshot() if self.needs_ollama else None},
                               cold_first=before.get("state") == "not_loaded")
        print(f"\n  [{name}] 로그: .safemate/test_logs/5-6_{name}_{suite.run_id}.txt | 중단 {summary['stop_reason']} | "
              f"요약 {json.dumps(summary['summary'], ensure_ascii=False)[:600]}")
        self.assertTrue(summary["protected_unchanged"], "기존 실측 로그·공개 자료가 바뀌었습니다.")
        self.assertTrue(summary["operational_unchanged"], "운영 저장소 상태가 바뀌었습니다.")
        self.assertEqual([r["unit"] for r in rows if r.get("s1")], [], [r.get("s1") for r in rows if r.get("s1")])
        self.assertEqual([r["unit"] for r in rows if r.get("tool_error")], [], [r.get("tool_error") for r in rows])
        return summary


VIRTUAL = {"llm": "real", "forecast": "virtual", "rag": "fake", "store": "temporary",
           "embedding": "사용 안 함(가상 RAG·가상 색인, 실제 PyTorch 연산 없음)"}


class L_LV1_RepresentativeFlow(LiveCase):

    def test_LV1_representative_flow(self):
        """4.1 실제 Qwen 대표 흐름: 승인 3 + 거절 3 (별도 프로세스 재조회 포함)"""
        def units(ctx: dict[str, Any]) -> list:
            return [(f"LV1-{kind}-{rep}", (lambda kind=kind: row_from(run_representative(
                kind, real_llm(), child_mode="real", live=True, expect_ollama=ctx["expect_ollama"])))) for kind, rep in LV1_RUNS]
        self.execute("LV1", units, VIRTUAL)


def graph_suite_test(name: str):
    def test(self):
        def units(ctx: dict[str, Any]) -> list:
            return [(f"{sc['id']}-{rep}", (lambda sc=sc: graph_unit(sc, lambda _sc: real_llm(), live=True,
                                                                     expect_ollama=ctx["expect_ollama"])()))
                    for sc in SUITES[name] for rep in range(1, sc["repeats"] + 1)]
        self.execute(name, units, VIRTUAL, per_run=MAX_OBSERVED_TURN_SECONDS * 3)
    test.__doc__ = f"{name} 실제 Qwen 측정 (사례·반복 고정)"
    return test


class L_LV2_WeatherPaths(LiveCase):
    test_LV2_weather_paths = graph_suite_test("LV2")


class L_LV3_CityMultiTurn(LiveCase):
    test_LV3_city_multi_turn = graph_suite_test("LV3")


class L_LV4_RequestModes(LiveCase):
    test_LV4_request_modes = graph_suite_test("LV4")


class L_LV5_Unsupported(LiveCase):
    test_LV5_unsupported = graph_suite_test("LV5")


class L_LV6_Ambiguous(LiveCase):
    test_LV6_ambiguous = graph_suite_test("LV6")


class L_LV7_OutingLink(LiveCase):
    test_LV7_outing_link = graph_suite_test("LV7")


class L_LV9_Understanding(LiveCase):

    def test_LV9_understanding(self):
        """4.9-B 일정 이해 21문장 × 3회"""
        def units(ctx: dict[str, Any]) -> list:
            return [(f"{e['id']}-{rep}", (lambda e=e: row_from(run_understanding(e, real_llm(), live=True,
                                                                                  expect_ollama=ctx["expect_ollama"]))))
                    for e in LV9 for rep in range(1, e["repeats"] + 1)]
        self.execute("LV9", units, VIRTUAL, per_run=MAX_OBSERVED_TURN_SECONDS * 3)


def prepare_real_rag(ctx: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """(묶음 시작 조건) 실제 장치 자동 선택(또는 --device 지정)과 실제 RAG 색인 준비. 실패 이유와 선택 기록을 로그에 남긴다.
    모델 다운로드를 막기 위해 HF 오프라인으로 준비한다(캐시가 없으면 실패로 기록, 자동 설치·다운로드 없음)."""
    with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
        selection = du.select_embedding_device(OPTIONS.device or "auto")
        info = {k: selection.get(k) for k in ("requested", "selected", "device", "attempts", "details", "error")}
        if not selection["ok"]:
            return f"임베딩 장치 선택 실패: {selection['error']}", info
        try:
            index = sr.prepare_safety_index(selection["selected"])
        except Exception as exc:  # noqa: BLE001
            return f"안전수칙 색인 준비 실패: {type(exc).__name__}: {exc}"[:400], info
    ctx.update(device=selection["selected"], index=index, selection=info)
    return None, dict(info, index_device=getattr(index, "device", None))


REAL = {"llm": "real", "forecast": "real", "rag": "real", "store": "temporary"}


class L_LV8_RealData(LiveCase):
    needs_real = True

    def test_LV8_real_data_flows(self):
        """4.8 실제 Open-Meteo·실제 RAG·실제 Qwen 종합 (한 흐름에서 필수 구성 요소가 모두 PASS여야 종합 성공)"""
        real_weather, real_search = od.get_outing_weather, od.search_safety_guidance

        def units(ctx: dict[str, Any]) -> list:
            return [(f"{spec['id']}-{rep}", (lambda spec=spec: row_from(run_data_flow(
                spec, real_llm(), weather_fn=real_weather, rag_fn=real_search, now=datetime.now(KST), device=ctx["device"],
                index=ctx["index"], real_weather=True, real_rag=True, real_llm=True, selection=ctx["selection"], live=True,
                expect_ollama=ctx["expect_ollama"])))) for spec in LV8 for rep in range(1, spec["repeats"] + 1)]
        self.execute("LV8", units, REAL, precondition=prepare_real_rag)

    def test_LV8C_controlled_integration(self):
        """4.8-C 통제 통합: 실제 Qwen·실제 RAG·실제 장치 + 고정 가상 예보 (실제 Open-Meteo 측정과 구분)"""
        real_search = od.search_safety_guidance

        def units(ctx: dict[str, Any]) -> list:
            return [(f"{LV8C['id']}-{rep}", (lambda: row_from(run_data_flow(
                LV8C, real_llm(), weather_fn=make_weather(LV8C["forecast"]), rag_fn=real_search, now=FIXED_NOW,
                device=ctx["device"], index=ctx["index"], real_weather=False, real_rag=True, real_llm=True,
                selection=ctx["selection"], live=True, expect_ollama=ctx["expect_ollama"]))))
                for rep in range(1, LV8C["repeats"] + 1)]
        self.execute("LV8C", units, dict(REAL, forecast="virtual"), precondition=prepare_real_rag)


class L_LV8R_RagProbe(LiveCase):
    """4.8-R 실제 RAG 단독. LLM을 쓰지 않으므로 --live-real만 있으면 되고 Ollama 준비 조건을 쓰지 않는다.
    측정 중에는 od.call_ollama를 호출 차단 장치로 바꿔, LLM 호출이 0회라는 로그 표기(llm=none)가 사실인지 확인한다."""
    needs_real = True
    needs_ollama = False

    def test_LV8R_rag_probe(self):
        """4.8-R 실제 RAG 단독: 출처 파일·쪽·점수(1 − 코사인 거리)·본문 앞부분·색인 장치 기록 (본문 관련성은 수동 검토)"""
        def units(ctx: dict[str, Any]) -> list:
            def unit(probe: dict[str, Any]) -> Callable[[], dict[str, Any]]:
                def execute() -> dict[str, Any]:
                    blocked: list[str] = []

                    def no_llm(*args, **kwargs):
                        blocked.append(kwargs.get("stage", "?"))
                        raise od.DecisionError("llm_unavailable", "blocked", "RAG 단독 측정에서는 LLM을 부르지 않습니다.")
                    with mock.patch.object(od, "call_ollama", no_llm):
                        result = sr.search_safety_guidance(probe["query"], ctx["index"])
                    judged = judge_rag_probe(probe, result, getattr(ctx["index"], "device", None), ctx["selection"])
                    ind = dict({k: NA for k in INDICATORS}, external_ok=judged["status"], device_ok=judged["device"])
                    s1 = [f"장치 오보고: {judged['device_reason']}"] if judged["device"] == FAIL else []
                    return {"indicators": ind, "s1": s1, "env_error": judged["status"] == ENV_ERROR,
                            "problems": [f"LLM 호출이 일어남: {blocked}"] if blocked else [], "probe": probe, "judged": judged,
                            "llm_calls_blocked": len(blocked)}
                return execute
            return [(p["id"], unit(p)) for p in LV8R]
        self.execute("LV8R", units, {"llm": "none (호출 차단 장치로 확인)", "forecast": "none", "rag": "real"},
                     per_run=10.0, precondition=prepare_real_rag)


class L_LV10_ExistingLiveTimed(LiveCase):
    """기존 L층 테스트를 파일 수정 없이 같은 프로세스에서 실행하고, od.call_ollama를 시간 측정 감싸개로 바꿔
    테스트별 LLM 호출 수·시간을 기록한다. 결과는 classify_existing으로 분류하고, 측정할 수 없는 값은 '미측정'으로 남긴다.
    메서드별로 따로 실행한다(이미 측정한 메서드는 재측정 사유 없이는 다시 실행되지 않음)."""

    def run_existing(self, name: str, module: Any, class_name: str, options: list[tuple[Any, dict[str, Any]]],
                     needs_device: bool = False) -> None:
        recorder = RecordingLLM(od.call_ollama)
        loader = unittest.TestLoader()
        tests = [t for t in loader.loadTestsFromTestCase(getattr(module, class_name)) if isinstance(t, unittest.TestCase)]

        def prepare(ctx: dict[str, Any]) -> Any:
            if not needs_device:
                return None
            with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
                selection = du.select_embedding_device(OPTIONS.device or "auto")
            info = {k: selection.get(k) for k in ("requested", "selected", "device", "error")}
            if not selection["ok"]:
                return f"임베딩 장치 선택 실패: {selection['error']}", info
            ctx["device"] = selection["selected"]
            return None, info

        def units(ctx: dict[str, Any]) -> list:
            def unit(test: unittest.TestCase) -> Callable[[], dict[str, Any]]:
                def execute() -> dict[str, Any]:
                    before = len(recorder.calls)
                    ops_before, protected_before = operational_state(), protected_records()
                    outcome = unittest.TestResult()
                    output = io.StringIO()
                    with contextlib.ExitStack() as stack:
                        for target, values in options:
                            for key, value in values.items():
                                stack.enter_context(mock.patch.object(target, key, ctx.get(value) if value == "device" else value))
                        stack.enter_context(mock.patch.object(od, "call_ollama", recorder))
                        stack.enter_context(contextlib.redirect_stdout(output))
                        test.run(outcome)
                    calls = recorder.calls[before:]
                    row = classify_existing(outcome, ops_changed=operational_state() != ops_before,
                                            protected_changed=protected_records() != protected_before)
                    row.update(test=test.id(), llm_calls=len(calls), llm_seconds=round(sum(c["seconds"] for c in calls), 1),
                               llm_stages=[c["stage"] for c in calls], printed=output.getvalue()[-6000:],
                               obs={"turns": [{"text": test.id(), "seconds": round(sum(c["seconds"] for c in calls), 1),
                                               "llm_calls": [{"stage": c["stage"], "seconds": c["seconds"]} for c in calls]}]})
                    if class_name == "L_OD_LiveIntent":
                        row["intent_summary"] = parse_intent_summary(output.getvalue()) or "미측정(요약 줄을 읽지 못함)"
                    return row
                return execute
            return [(t.id().split(".")[-1], unit(t)) for t in tests]
        self.execute(name, units, {"llm": "real", "forecast": "기존 테스트 정의", "rag": "기존 테스트 정의"},
                     per_run=MAX_OBSERVED_RUN_SECONDS * 10, extra={"existing_class": f"{module.__name__}.{class_name}"},
                     precondition=prepare)

    def test_LV10a_existing_intent_table(self):
        """4.9-A 기존 L_OD_LiveIntent(20문장 × 3회)를 코드 변경 없이 실행"""
        self.run_existing("LV10a", ts, "L_OD_LiveIntent", [(ts.OPTIONS, {"live_ollama": True})])

    def test_LV10b_existing_stage3_live(self):
        """4.11 기존 3단계 L1_LiveOllama(8개)를 코드 변경 없이 실행"""
        self.run_existing("LV10b", tod, "L1_LiveOllama", [(tod.OPTIONS, {"live_ollama": True})])

    def test_LV10c_existing_stage3_full(self):
        """4.11 기존 3단계 L2_LiveFull(실제 날씨·RAG·Ollama 3개)을 코드 변경 없이 실행 (--live-real 필요)"""
        if not OPTIONS.live_real:
            self.skipTest("--live-real 를 함께 지정하면 실행합니다.")
        self.run_existing("LV10c", tod, "L2_LiveFull", [(tod.OPTIONS, {"live_full": True, "device": "device"})],
                          needs_device=True)

    def test_LV10d_existing_graph_live(self):
        """4.11 기존 4단계 L_LiveIntegration(L1~L3, 실제 Open-Meteo 포함)을 코드 변경 없이 실행 (--live-real 필요)"""
        if not OPTIONS.live_real:
            self.skipTest("--live-real 를 함께 지정하면 실행합니다.")
        self.run_existing("LV10d", tsg, "L_LiveIntegration", [(tsg.OPTIONS, {"live_ollama": True, "device": "device"})],
                          needs_device=True)


# %% 실행 진입점
# 핵심 요약: 이 파일 전용 옵션(--live-ollama, --live-real, --device, --rerun-reason, 별도 프로세스용 --reopen-store)을
# 먼저 읽고, 나머지는 unittest에 넘긴다. 반복 횟수를 바꾸는 옵션은 없다.
def main() -> None:
    """옵션을 읽고 테스트를 실행한다. --reopen-store는 4.1 별도 프로세스 재조회 전용이다."""
    # 1. 별도 프로세스 재조회 모드
    argv = sys.argv[1:]
    if "--reopen-store" in argv:
        values = dict(zip(argv[::2], argv[1::2]))
        raise SystemExit(reopen_main(Path(values["--reopen-store"]), datetime.fromisoformat(values["--fixed-now"]),
                                     values.get("--reopen-mode", "fake")))
    # 2. 측정 옵션
    rest = []
    while argv:
        item_ = argv.pop(0)
        if item_ == "--live-ollama":
            OPTIONS.live_ollama = True
        elif item_ == "--live-real":
            OPTIONS.live_real = True
        elif item_ == "--device":
            OPTIONS.device = argv.pop(0)
        elif item_ == "--rerun-reason":
            OPTIONS.rerun_reason = argv.pop(0)
        else:
            rest.append(item_)
    if OPTIONS.device not in (None, "auto", "xpu", "cuda", "cpu"):
        raise SystemExit("--device 는 auto, xpu, cuda, cpu 중 하나입니다.")
    unittest.main(argv=[sys.argv[0]] + rest)


if __name__ == "__main__":
    main()
