"""SafeMate 5단계: 가상 일정 JSON 저장소

핵심 요약
- 가상 사용자의 일정(schedules)과 승인받아 실제로 적용한 작업 기록(action_history)을 JSON 문서 하나에 함께 저장한다.
  둘을 한 번의 파일 교체로 저장하므로 "일정은 바뀌었는데 이력은 없음" 같은 불일치가 생기지 않는다.
- 프로그램을 종료해도 일정이 남는다. (InMemorySaver는 대화 상태만 보관하고, 일정은 이 파일이 보관한다)
- 초기 가상 일정은 저장소를 처음 만들 때 한 번만 넣는다. 이미 있는 저장소를 덮어쓰거나, 사라지거나 손상된 저장소를
  임의로 다시 만들지 않는다.
- 같은 action_id의 작업은 프로그램을 다시 실행해도 한 번만 적용한다. (승인 버튼 중복·재시도 방지)
- 일정의 의미 검사(대상 일정 존재, 버전, 충돌, 변경 제한)는 schedule_service.py가 한다.
  이 파일은 '안전하게 읽고, 잠그고, 한 번만 쓰는' 일만 맡는다.

외부 서비스·데이터베이스 없이 Python 표준 라이브러리만 사용한다.
"""

# %% 공통 라이브러리 불러오기
# 핵심 요약: 파일·시각·잠금 처리에 표준 라이브러리만 쓰고, 한국 시각(KST)은 1단계와 같은 값을 쓴다.
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from a_outing import KST


# %% 저장소 설정값
# 핵심 요약: 경로·허용 값·대기 시간처럼 저장소 전체가 함께 쓰는 값을 한곳에 모은다.
# 경로는 운영체제에 관계없이 이 파일이 있는 프로젝트 폴더를 기준으로 만든다. (Windows 절대 경로를 코드에 쓰지 않음)
SCHEMA_VERSION = 1
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_STORE_PATH = PROJECT_DIR / ".safemate" / "schedule_store.json"
DEFAULT_SEED_PATH = PROJECT_DIR / "data" / "schedule" / "seed_schedule.json"
STORE_PATH_ENV = "SAFEMATE_SCHEDULE_PATH"     # 학교 서버·Colab에서 저장 위치를 바꿀 때 쓰는 환경변수

CATEGORIES = ("outing", "appointment", "home", "other")   # outing만 날씨 판단 대상(실외 외출)
STATUSES = ("scheduled", "completed", "cancelled")
SOURCES = ("seed", "user")
OPERATIONS = ("create", "update", "cancel", "complete")
SCHEDULE_KEYS = ("schedule_id", "title", "aliases", "start_at", "end_at", "city", "category", "status",
                 "reschedulable", "version", "source", "created_at", "updated_at", "completed_at", "cancelled_at")
HISTORY_KEYS = ("action_id", "content_hash", "operation", "schedule_id", "before", "after", "reason",
                "approved_at", "executed_at", "result")
TITLE_LIMIT, ALIAS_LIMIT, CITY_LIMIT, REASON_LIMIT = 40, 8, 40, 300
MAX_SCHEDULES, MAX_HISTORY = 500, 5000
SCHEDULE_ID_RE = re.compile(r"schedule_[0-9]{3,}")
HEX32_RE, HEX64_RE = re.compile(r"[0-9a-f]{32}"), re.compile(r"[0-9a-f]{64}")

LOCK_WAIT_SECONDS = 5.0      # 다른 프로그램의 저장이 끝나기를 기다리는 최대 시간
LOCK_RETRY_SECONDS = 0.05
REPLACE_RETRIES = 3          # Windows에서 백신·편집기가 파일을 잠깐 잡고 있을 때 다시 시도하는 횟수
REPLACE_RETRY_SECONDS = 0.1


# %% 저장소 오류 형식
# 핵심 요약: 무엇이 잘못됐는지(code)와 사용자·개발자에게 보여 줄 설명(message)을 함께 담는다.
class ScheduleStoreError(Exception):
    """code: not_initialized, store_missing, corrupt, schema_error, read_denied, write_denied, write_failed,
    lock_timeout, lock_error, seed_error, mutation_error"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# %% 한국 시각 문자열 변환 함수
# 핵심 요약: 모든 시각을 '2026-11-26T14:00:00+09:00' 같은 KST ISO 문자열 하나의 형식으로 저장한다.
def iso(moment: datetime) -> str:
    """시간대가 있는 datetime을 KST ISO 문자열로 바꾼다."""
    return moment.astimezone(KST).isoformat(timespec="seconds")


def parse_iso(text: Any) -> datetime | None:
    """시간대가 있는 ISO 문자열이면 KST datetime, 아니면 None을 돌려준다. (시간대 없는 값은 받지 않음)"""
    if not isinstance(text, str):
        return None
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    return moment.astimezone(KST) if moment.tzinfo is not None else None


# %% 작업 내용 해시 함수
# 핵심 요약: 승인받은 작업의 type과 payload를 정해진 순서의 JSON으로 바꿔 SHA-256을 만든다.
# 같은 action_id로 다른 내용이 들어오면 해시가 달라지므로, 오래된 id를 재사용한 실행을 막을 수 있다.
def content_hash(action: dict[str, Any]) -> str:
    """{"type", "payload"}만으로 만든 64자리 해시. 승인 시각처럼 실행마다 달라지는 값은 넣지 않는다."""
    basis = {"type": action.get("type"), "payload": action.get("payload")}
    text = json.dumps(basis, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# %% 일정 문서 검사 함수
# 핵심 요약: 파일에서 읽은 문서와 저장하기 직전의 문서가 약속한 구조를 지키는지 확인한다.
# 문제가 있는 문서는 읽기도 저장도 하지 않는다. (잘못된 일정을 기준으로 충돌을 판단하지 않기 위함)
def _check_schedule(item: Any, index: int) -> list[str]:
    """일정 한 개의 문제 목록"""
    where = f"schedules[{index}]"
    if not isinstance(item, dict):
        return [f"{where}: 딕셔너리가 아님"]
    problems = []
    if set(item) != set(SCHEDULE_KEYS):
        problems.append(f"{where}: 키가 규격과 다름")
        return problems
    if not isinstance(item["schedule_id"], str) or not SCHEDULE_ID_RE.fullmatch(item["schedule_id"]):
        problems.append(f"{where}: schedule_id 형식")
    if not isinstance(item["title"], str) or not 1 <= len(item["title"].strip()) <= TITLE_LIMIT:
        problems.append(f"{where}: title")
    aliases = item["aliases"]
    if not isinstance(aliases, list) or len(aliases) > ALIAS_LIMIT or any(
            not isinstance(a, str) or not 1 <= len(a.strip()) <= TITLE_LIMIT for a in aliases):
        problems.append(f"{where}: aliases")
    start, end = parse_iso(item["start_at"]), parse_iso(item["end_at"])
    if start is None or end is None or end <= start:
        problems.append(f"{where}: start_at·end_at")
    if item["city"] is not None and (not isinstance(item["city"], str) or not 1 <= len(item["city"]) <= CITY_LIMIT):
        problems.append(f"{where}: city")
    if item["category"] not in CATEGORIES or item["status"] not in STATUSES or item["source"] not in SOURCES:
        problems.append(f"{where}: category·status·source 값")
    if type(item["reschedulable"]) is not bool or type(item["version"]) is not int or item["version"] < 1:
        problems.append(f"{where}: reschedulable·version")
    for key in ("created_at", "updated_at"):
        if parse_iso(item[key]) is None:
            problems.append(f"{where}: {key}")
    for key, status in (("completed_at", "completed"), ("cancelled_at", "cancelled")):
        value = item[key]
        if (value is None) != (item["status"] != status) or (value is not None and parse_iso(value) is None):
            problems.append(f"{where}: {key}와 status가 맞지 않음")
    return problems


def _check_history(entry: Any, index: int) -> list[str]:
    """작업 기록 한 줄의 문제 목록"""
    where = f"action_history[{index}]"
    if not isinstance(entry, dict) or set(entry) != set(HISTORY_KEYS):
        return [f"{where}: 키가 규격과 다름"]
    problems = []
    if not isinstance(entry["action_id"], str) or not HEX32_RE.fullmatch(entry["action_id"]):
        problems.append(f"{where}: action_id 형식")
    if not isinstance(entry["content_hash"], str) or not HEX64_RE.fullmatch(entry["content_hash"]):
        problems.append(f"{where}: content_hash 형식")
    if entry["operation"] not in OPERATIONS or entry["result"] != "applied":
        problems.append(f"{where}: operation·result 값")
    if not isinstance(entry["schedule_id"], str) or not SCHEDULE_ID_RE.fullmatch(entry["schedule_id"]):
        problems.append(f"{where}: schedule_id 형식")
    for key in ("before", "after"):
        if entry[key] is not None and not isinstance(entry[key], dict):
            problems.append(f"{where}: {key}")
    if not isinstance(entry["reason"], str) or len(entry["reason"]) > REASON_LIMIT:
        problems.append(f"{where}: reason")
    if parse_iso(entry["approved_at"]) is None or parse_iso(entry["executed_at"]) is None:
        problems.append(f"{where}: approved_at·executed_at")
    return problems


def validate_document(doc: Any) -> list[str]:
    """문서 전체의 문제 목록을 돌려준다. 빈 목록이면 정상이다."""
    # 1. 최상위 구조
    if not isinstance(doc, dict):
        return ["문서가 딕셔너리가 아님"]
    if set(doc) != {"schema_version", "store_id", "created_at", "updated_at", "schedules", "action_history"}:
        return ["최상위 키가 규격과 다름"]
    problems = []
    if doc["schema_version"] != SCHEMA_VERSION:
        problems.append("schema_version이 지원하지 않는 값")
    if not isinstance(doc["store_id"], str) or not HEX32_RE.fullmatch(doc["store_id"]):
        problems.append("store_id 형식")
    if parse_iso(doc["created_at"]) is None or parse_iso(doc["updated_at"]) is None:
        problems.append("created_at·updated_at")
    schedules, history = doc["schedules"], doc["action_history"]
    if not isinstance(schedules, list) or len(schedules) > MAX_SCHEDULES:
        return problems + ["schedules가 목록이 아니거나 너무 김"]
    if not isinstance(history, list) or len(history) > MAX_HISTORY:
        return problems + ["action_history가 목록이 아니거나 너무 김"]

    # 2. 일정·기록 하나하나
    for index, item in enumerate(schedules):
        problems += _check_schedule(item, index)
    for index, entry in enumerate(history):
        problems += _check_history(entry, index)

    # 3. 일정 id와 action_id는 겹치면 안 된다. (같은 작업이 두 번 기록된 상태를 허용하지 않음)
    # 리스트·딕셔너리처럼 set()에 넣을 수 없는 값은 문자열만 골라 비교한다. 형식 오류 자체는 위 2에서 이미 문제로 기록했다.
    ids = [s.get("schedule_id") for s in schedules if isinstance(s, dict) and isinstance(s.get("schedule_id"), str)]
    if len(ids) != len(set(ids)):
        problems.append("schedule_id 중복")
    actions = [h.get("action_id") for h in history if isinstance(h, dict) and isinstance(h.get("action_id"), str)]
    if len(actions) != len(set(actions)):
        problems.append("action_id 중복")
    return problems


# %% 초기 가상 일정 생성 함수
# 핵심 요약: 날짜를 고정하지 않은 초기 데이터(day_offset, start, minutes)를 주입한 현재 시각 기준의 실제 날짜로 바꾼다.
def build_seed_document(seed: Any, now: datetime) -> dict[str, Any]:
    """초기 데이터 파일 내용으로 저장소 문서를 만든다. 형식이 틀리면 seed_error를 낸다."""
    # 1. 초기 데이터 형식 확인
    if not isinstance(seed, dict) or seed.get("schema_version") != SCHEMA_VERSION or not isinstance(seed.get("schedules"), list):
        raise ScheduleStoreError("seed_error", "초기 일정 파일의 형식이 올바르지 않습니다.")
    stamp = iso(now)
    schedules = []

    # 2. 항목마다 '오늘 + day_offset일'의 시작 시각과 소요 시간으로 실제 시작·종료 시각을 만든다.
    for number, item in enumerate(seed["schedules"], start=1):
        try:
            offset, minutes = item["day_offset"], item["minutes"]
            hour, minute = (int(part) for part in item["start"].split(":"))
            if type(offset) is not int or type(minutes) is not int or not 0 <= offset <= 30 or not 10 <= minutes <= 720:
                raise ValueError
            day: date = now.astimezone(KST).date() + timedelta(days=offset)
            start = datetime(day.year, day.month, day.day, hour, minute, tzinfo=KST)
        except (KeyError, TypeError, ValueError, AttributeError):
            raise ScheduleStoreError("seed_error", f"초기 일정 {number}번째 항목의 날짜·시각·소요 시간이 올바르지 않습니다.") from None
        schedules.append({
            "schedule_id": f"schedule_{number:03}", "title": item.get("title"), "aliases": item.get("aliases", []),
            "start_at": iso(start), "end_at": iso(start + timedelta(minutes=minutes)), "city": item.get("city"),
            "category": item.get("category"), "status": "scheduled", "reschedulable": item.get("reschedulable"),
            "version": 1, "source": "seed", "created_at": stamp, "updated_at": stamp,
            "completed_at": None, "cancelled_at": None})

    # 3. 완성한 문서도 저장소 규격으로 검사한다.
    doc = {"schema_version": SCHEMA_VERSION, "store_id": uuid.uuid4().hex, "created_at": stamp, "updated_at": stamp,
           "schedules": schedules, "action_history": []}
    problems = validate_document(doc)
    if problems:
        raise ScheduleStoreError("seed_error", "초기 일정이 저장소 규격과 맞지 않습니다: " + "; ".join(problems[:3]))
    return doc


# %% 다음 일정 id 함수
def next_schedule_id(doc: dict[str, Any]) -> str:
    """기존 번호 중 가장 큰 값 + 1. 취소된 일정의 번호도 다시 쓰지 않는다."""
    numbers = [int(s["schedule_id"].split("_")[1]) for s in doc["schedules"]]
    return f"schedule_{max(numbers, default=0) + 1:03}"


# %% 같은 프로세스 안의 잠금
# 핵심 요약: Gradio처럼 여러 요청을 동시에 처리하는 프로그램 안에서, 같은 파일을 두 요청이 동시에 읽고 쓰지 않게 한다.
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.Lock:
    """파일 경로마다 하나의 Lock을 쓴다."""
    key = os.path.normcase(str(path))
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


# %% 다른 프로세스와의 잠금 파일
# 핵심 요약: 'store.json.lock' 파일을 '없을 때만 만들기'(O_EXCL)로 만들어, 다른 프로그램이 같은 파일을 동시에 쓰지 않게 한다.
# 오래된 잠금 파일이라도 다른 프로그램이 아직 쓰는 중인지 안전하게 알 수 없으므로 강제로 지우지 않는다.
# 기다려도 풀리지 않으면 저장을 중단하고, 사용자가 확인한 뒤 직접 지우도록 안내한다.
class FileLock:
    """with 문으로 쓴다. 내가 만든 잠금 파일만 지운다. (다른 프로그램의 잠금 파일은 지우지 않음)"""

    def __init__(self, path: Path, wait_seconds: float = LOCK_WAIT_SECONDS):
        self.path, self.wait_seconds = path, wait_seconds
        self.token = uuid.uuid4().hex

    def __enter__(self) -> "FileLock":
        # 1. 기다리는 시간 안에서 잠금 파일 만들기를 반복한다.
        deadline = time.monotonic() + self.wait_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise ScheduleStoreError("lock_timeout", self._timeout_message()) from None
                time.sleep(LOCK_RETRY_SECONDS)
                continue
            except PermissionError:
                # Windows에서 다른 프로그램이 잠금 파일을 지우는 중이면 잠깐 PermissionError가 날 수 있다.
                if time.monotonic() >= deadline:
                    raise ScheduleStoreError("lock_error", f"잠금 파일({self.path.name})을 만들 권한이 없습니다.") from None
                time.sleep(LOCK_RETRY_SECONDS)
                continue

            # 2. 잠금 파일에는 확인용 정보(내 토큰·만든 시각·프로세스 번호)만 적는다. 개인정보는 없다.
            # 기록에 실패하면 방금 O_EXCL로 만든 파일, 즉 내가 만든 것이 확실한 파일만 지우고 오류로 알린다.
            # (O_EXCL은 파일이 없을 때만 성공하므로 이 시점의 파일은 다른 프로그램의 잠금일 수 없다)
            try:
                handle = os.fdopen(fd, "w", encoding="utf-8")
            except OSError:
                os.close(fd)
                self._discard_created()
                raise ScheduleStoreError("lock_error", "잠금 파일을 준비하지 못해 저장하지 않았습니다. 일정 파일은 바뀌지 않았습니다.") from None
            try:
                with handle:
                    json.dump({"token": self.token, "pid": os.getpid(), "created_at": iso(datetime.now(KST))}, handle)
            except (OSError, ValueError):
                self._discard_created()
                raise ScheduleStoreError("lock_error", "잠금 정보를 기록하지 못해 저장하지 않았습니다. 일정 파일은 바뀌지 않았습니다.") from None
            return self

    def __exit__(self, *exc_info: Any) -> None:
        # 3. 잠금 파일의 토큰이 내 것일 때만 지운다.
        try:
            owner = json.loads(self.path.read_text(encoding="utf-8")).get("token")
        except (OSError, ValueError, AttributeError):
            return
        if owner == self.token:
            try:
                self.path.unlink()
            except OSError:
                pass

    def _discard_created(self) -> None:
        """__enter__에서 방금 만든 잠금 파일만 지운다. 지우지 못해도 다른 오류를 덮지 않는다."""
        try:
            self.path.unlink()
        except OSError:
            pass

    def _timeout_message(self) -> str:
        """오래 남은 잠금 파일의 나이와 복구 방법을 알려 준다."""
        try:
            age = int(time.time() - self.path.stat().st_mtime)
            age_text = f"{age}초 전에 만들어진 "
        except OSError:
            age_text = ""
        return (f"다른 프로그램이 일정 파일을 쓰고 있어 저장하지 않았습니다. {age_text}잠금 파일({self.path.name})이 남아 있습니다. "
                "SafeMate가 다른 창에서 실행 중이 아닌지 확인한 뒤, 실행 중인 프로그램이 없을 때만 이 잠금 파일을 직접 삭제하고 "
                "다시 시도해 주세요. 일정 파일은 바뀌지 않았습니다.")


# %% 원자적 파일 저장 함수
# 핵심 요약: 같은 폴더의 임시 파일에 다 쓴 뒤 os.replace로 한 번에 바꾼다.
# 저장 도중 오류가 나도 기존 파일은 그대로 남는다. (반쯤 쓰인 파일이 생기지 않음)
def write_json_atomic(path: Path, doc: dict[str, Any]) -> None:
    """실패하면 write_denied·write_failed 오류를 내고, 임시 파일은 지운다."""
    text = json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        # 1. 임시 파일에 UTF-8로 쓰고 디스크까지 내려보낸다.
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
        except PermissionError:
            raise ScheduleStoreError("write_denied", f"일정 파일 폴더({path.parent.name})에 쓸 권한이 없습니다. 일정 파일은 바뀌지 않았습니다.") from None
        except OSError as exc:
            raise ScheduleStoreError("write_failed", f"일정을 저장하는 중 오류가 났습니다({type(exc).__name__}). 일정 파일은 바뀌지 않았습니다.") from None

        # 2. 기존 파일과 한 번에 바꾼다. Windows에서 다른 프로그램이 잠깐 잡고 있으면 몇 번만 다시 시도한다.
        for attempt in range(1, REPLACE_RETRIES + 1):
            try:
                os.replace(temp, path)
                return
            except PermissionError:
                if attempt == REPLACE_RETRIES:
                    raise ScheduleStoreError("write_failed", "다른 프로그램이 일정 파일을 사용 중이라 저장하지 못했습니다. 일정 파일은 바뀌지 않았습니다.") from None
                time.sleep(REPLACE_RETRY_SECONDS)
            except OSError as exc:
                raise ScheduleStoreError("write_failed", f"일정 파일을 바꾸지 못했습니다({type(exc).__name__}). 일정 파일은 바뀌지 않았습니다.") from None
    finally:
        # 3. 성공했으면 임시 파일은 이미 없다. 실패했으면 남은 임시 파일을 지운다.
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass


# %% 저장소 경로 함수
def default_store_path() -> Path:
    """환경변수 SAFEMATE_SCHEDULE_PATH가 있으면 그 경로, 없으면 프로젝트의 .safemate/schedule_store.json"""
    configured = os.environ.get(STORE_PATH_ENV, "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_STORE_PATH


# %% 일정 저장소
# 핵심 요약: 처음 한 번만 초기 데이터로 만들고(ensure_initialized), 읽고(load), 한 작업을 잠금 안에서 한 번만 적용한다(transact).
class ScheduleStore:
    """path: 일정 JSON 경로(테스트는 임시 폴더), seed_path: 초기 일정, clock: 현재 시각 함수(테스트에서 고정)"""

    def __init__(self, path: str | Path | None = None, *, seed_path: str | Path | None = None,
                 clock: Callable[[], datetime] | None = None, lock_wait: float = LOCK_WAIT_SECONDS):
        self.path = Path(path).expanduser() if path is not None else default_store_path()
        self.seed_path = Path(seed_path) if seed_path is not None else DEFAULT_SEED_PATH
        self.marker_path = self.path.with_name(self.path.name + ".created")   # 처음 만들었다는 표시
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.clock = clock or (lambda: datetime.now(KST))
        self.lock_wait = lock_wait

    # %% 최초 1회 초기화
    # 핵심 요약: 저장소와 '처음 만들었다는 표시 파일'이 둘 다 없을 때만 초기 일정으로 만든다.
    # 표시 파일은 있는데 저장소가 없으면, 사라진 것이므로 다시 만들지 않고 복구를 안내한다.
    def ensure_initialized(self) -> dict[str, Any]:
        """{"status": "created" | "existing", "path", "schedules"}를 돌려준다."""
        with _thread_lock(self.path), FileLock(self.lock_path, self.lock_wait):
            # 1. 이미 있으면 읽어서 검사만 한다. 손상됐으면 오류를 내고 덮어쓰지 않는다.
            if self.path.exists():
                doc = self._read()
                if not self.marker_path.exists():
                    # 저장소를 쓴 뒤 표시 파일을 쓰기 전에 종료된 경우만 표시 파일을 보충한다. (일정은 건드리지 않음)
                    write_json_atomic(self.marker_path, {"store_id": doc["store_id"], "created_at": doc["created_at"]})
                return {"status": "existing", "path": str(self.path), "schedules": len(doc["schedules"])}

            # 2. 표시 파일만 남아 있으면 운영 저장소가 사라진 것이다. 임의로 새로 만들지 않는다.
            if self.marker_path.exists():
                raise ScheduleStoreError("store_missing", self._missing_message())

            # 3. 처음 실행: 초기 데이터로 문서를 만들고 저장한 뒤 표시 파일을 남긴다.
            try:
                seed = json.loads(self.seed_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise ScheduleStoreError("seed_error", f"초기 일정 파일({self.seed_path.name})을 읽지 못했습니다.") from None
            doc = build_seed_document(seed, self.clock())
            write_json_atomic(self.path, doc)
            write_json_atomic(self.marker_path, {"store_id": doc["store_id"], "created_at": doc["created_at"]})
            return {"status": "created", "path": str(self.path), "schedules": len(doc["schedules"])}

    # %% 읽기
    def load(self) -> dict[str, Any]:
        """검사를 통과한 문서의 사본을 돌려준다. 저장된 파일을 바꾸지 않는다."""
        with _thread_lock(self.path):
            return self._read()

    # %% 한 작업을 한 번만 적용하기
    # 핵심 요약: 잠금 안에서 '다시 읽기 → 같은 action_id 확인 → 변경 함수 실행 → 문서 검사 → 원자적 저장'을 한 번에 한다.
    # 변경 함수(mutate)는 schedule_service가 넘긴다. 저장 직전의 최신 문서를 받아 대상·버전·충돌을 다시 검사한다.
    def transact(self, action_id: str, digest: str, mutate: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        """{"status": "applied" | "already_executed" | "rejected", "code", "message", "record", "outcome"}

        mutate(doc)는 doc을 직접 고치고 {"ok": bool, "code": str, "message": str, "history": {...}}를 돌려준다.
        history에는 operation·schedule_id·before·after·reason·approved_at을 넣는다."""
        # 1. id·해시 형식이 틀리면 파일을 열기 전에 거절한다.
        if not isinstance(action_id, str) or not HEX32_RE.fullmatch(action_id) \
                or not isinstance(digest, str) or not HEX64_RE.fullmatch(digest):
            return {"status": "rejected", "code": "invalid_action", "message": "작업 id 형식이 올바르지 않습니다.",
                    "record": None, "outcome": None}
        with _thread_lock(self.path), FileLock(self.lock_path, self.lock_wait):
            doc = self._read()

            # 2. 멱등성: 이미 적용한 action_id면 다시 적용하지 않는다. 같은 id인데 내용이 다르면 거절한다.
            for entry in doc["action_history"]:
                if entry["action_id"] == action_id:
                    if entry["content_hash"] == digest:
                        return {"status": "already_executed", "code": "already_executed",
                                "message": "이미 적용한 작업입니다.", "record": copy.deepcopy(entry), "outcome": None}
                    return {"status": "rejected", "code": "action_id_reused",
                            "message": "같은 작업 id에 다른 내용이 들어왔습니다.", "record": None, "outcome": None}

            # 3. 최신 문서의 사본에서 변경 함수를 실행한다. 거절하면 파일을 쓰지 않는다.
            work = copy.deepcopy(doc)
            try:
                outcome = mutate(work)
            except Exception as exc:  # noqa: BLE001 - 변경 함수의 버그가 파일을 망가뜨리지 않게 한다
                raise ScheduleStoreError("mutation_error", f"일정 변경 처리 중 오류가 났습니다({type(exc).__name__}). 일정 파일은 바뀌지 않았습니다.") from exc
            if not isinstance(outcome, dict) or outcome.get("ok") is not True:
                code = outcome.get("code") if isinstance(outcome, dict) else None
                message = outcome.get("message") if isinstance(outcome, dict) else None
                return {"status": "rejected", "code": code if isinstance(code, str) else "rejected",
                        "message": message if isinstance(message, str) else "변경할 수 없습니다.",
                        "record": None, "outcome": outcome}

            # 4. 실제로 적용한 작업만 이력에 남긴다. 일정과 이력을 같은 문서에 함께 쓴다.
            now = iso(self.clock())
            raw_history = outcome.get("history")
            history: dict[str, Any] = raw_history if isinstance(raw_history, dict) else {}
            entry = {key: history.get(key) for key in ("operation", "schedule_id", "before", "after", "reason", "approved_at")}
            entry.update(action_id=action_id, content_hash=digest, executed_at=now, result="applied")
            entry = {key: entry[key] for key in HISTORY_KEYS}
            work["action_history"].append(entry)
            work["updated_at"] = now

            # 5. 저장 직전에 문서 전체를 다시 검사한다. 규격을 어기는 변경은 저장하지 않는다.
            problems = validate_document(work)
            if problems:
                return {"status": "rejected", "code": "invalid_change",
                        "message": "변경 결과가 저장소 규격과 맞지 않아 저장하지 않았습니다: " + "; ".join(problems[:3]),
                        "record": None, "outcome": outcome}
            write_json_atomic(self.path, work)
            return {"status": "applied", "code": "applied", "message": "적용했습니다.",
                    "record": copy.deepcopy(entry), "outcome": outcome}

    # %% 내부: 파일 읽기와 검사
    def _read(self) -> dict[str, Any]:
        """없는 파일·권한·손상·규격 오류를 구분해 알린다. 어떤 경우에도 파일을 고치지 않는다."""
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            if self.marker_path.exists():
                raise ScheduleStoreError("store_missing", self._missing_message()) from None
            raise ScheduleStoreError("not_initialized", "일정 저장소가 아직 만들어지지 않았습니다. 처음 실행할 때 초기화해 주세요.") from None
        except PermissionError:
            raise ScheduleStoreError("read_denied", f"일정 파일({self.path.name})을 읽을 권한이 없습니다.") from None
        except OSError as exc:
            raise ScheduleStoreError("read_denied", f"일정 파일을 읽지 못했습니다({type(exc).__name__}).") from None
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise ScheduleStoreError("corrupt", f"일정 파일({self.path.name})이 손상되어 읽을 수 없습니다. "
                                                "파일을 임의로 초기화하지 않았습니다. 백업본으로 복구하거나 파일을 확인해 주세요.") from None
        problems = validate_document(doc)
        if problems:
            raise ScheduleStoreError("schema_error", "일정 파일의 구조가 올바르지 않습니다: " + "; ".join(problems[:3]))
        return doc

    def _missing_message(self) -> str:
        """사라진 운영 저장소의 복구 안내"""
        return (f"일정 파일({self.path.name})이 사라졌습니다. 이전에 만든 기록이 있어 새 초기 일정으로 다시 만들지 않았습니다. "
                f"백업본을 같은 위치에 복원하거나, 새로 시작하려면 표시 파일({self.marker_path.name})을 직접 삭제한 뒤 다시 실행해 주세요.")


# %% 독립 실행: 저장소 상태 확인
# 핵심 요약: 'python schedule_store.py'로 운영 저장소를 (처음이면) 만들고 일정 개수를 확인한다. 테스트는 이 경로를 쓰지 않는다.
if __name__ == "__main__":
    store = ScheduleStore()
    try:
        print(store.ensure_initialized())
    except ScheduleStoreError as error:
        print(f"[{error.code}] {error}")
