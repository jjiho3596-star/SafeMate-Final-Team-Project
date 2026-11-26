"""SafeMate 5단계 일정 관리 테스트

실행 방법 (.venv-xpu)
- 기본 (M층 단위·I층 통합·R층 회귀, 네트워크·Ollama 불필요):
    & .\\.venv-xpu\\Scripts\\python.exe test_schedule.py -v
- 실제 연동 (L층, 실제 Ollama qwen2.5:3b 필요):
    & .\\.venv-xpu\\Scripts\\python.exe test_schedule.py --live-ollama -v L_OD_LiveIntent L_AG_LiveFlow
    (L_AG_LiveFlow 진행 로그: 기본 .safemate/test_logs/5-4_live_flow.txt, --live-log 경로로 바꿀 수 있음)

테스트 층
- M층: 일정 저장소·서비스·일정 이해 검사의 단위 테스트 (가상 시각, 임시 폴더)
- I층: 실제 코드 통합(일정 에이전트·4단계 그래프·3단계) + 가상 LLM·가상 예보·가상 RAG
- L층: 실제 qwen2.5:3b (+ 명시된 가상 예보, 임시 저장소)
- R층: 1·2단계 파일 보존, 운영 일정 파일을 테스트가 건드리지 않았는지 확인

모든 테스트는 TemporaryDirectory 안의 임시 저장소만 쓴다. 운영 저장소(.safemate/schedule_store.json)는 읽거나 쓰지 않는다.
"""

# %% 공통 라이브러리와 검사 대상 불러오기
# 핵심 요약: 테스트 대상 모듈과 표준 라이브러리만 쓴다. 테스트 시작 시점의 운영 저장소 상태를 기록해 두었다가 끝에 비교한다.
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import outing_decision as od
import safemate_graph as sg
import schedule_agent as sa
import schedule_service as sv
import schedule_store as ss
import test_outing_decision as tod
import test_safemate_graph as tsg
from a_outing import KST

ROOT = Path(__file__).resolve().parent
OPTIONS = SimpleNamespace(device=None, live_ollama=False, live_log=None)
BASE_COMMIT = "c4e2ca0"
PROTECTED_FILES = ("a_outing.py", "safety_rag.py", "test_outing_decision.py")
# 모든 Mock 테스트의 기준 시각: 2026-11-25(수) 10:00 한국 시각
FIXED_NOW = datetime(2026, 11, 25, 10, 0, tzinfo=KST)


def file_state(path: Path) -> str | None:
    """파일이 없으면 None, 있으면 SHA-256"""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


OPERATIONAL_FILES = (ss.DEFAULT_STORE_PATH, ss.DEFAULT_STORE_PATH.with_name(ss.DEFAULT_STORE_PATH.name + ".created"))
OPERATIONAL_BEFORE = {str(p): file_state(p) for p in OPERATIONAL_FILES}


# %% 테스트 도우미
# 핵심 요약: 임시 폴더 저장소, 고정 시각, 간단한 변경 함수(mutate)를 만든다.
def clean_env() -> dict[str, str]:
    """운영 저장소 경로 환경변수를 지운 환경. (테스트가 운영 경로를 가리키지 않게 함)"""
    return {k: v for k, v in os.environ.items() if k != ss.STORE_PATH_ENV}


class StoreCase(unittest.TestCase):
    """테스트마다 새 임시 폴더와 새 저장소를 만든다."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        env = mock.patch.dict(os.environ, clean_env(), clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.path = Path(self.temp.name) / "store" / "schedule_store.json"
        self.now = FIXED_NOW

    def store(self, **kwargs) -> ss.ScheduleStore:
        """고정 시각을 쓰는 임시 저장소"""
        kwargs.setdefault("clock", lambda: self.now)
        return ss.ScheduleStore(self.path, **kwargs)

    def ready(self, **kwargs) -> ss.ScheduleStore:
        """초기화까지 마친 임시 저장소"""
        store = self.store(**kwargs)
        store.ensure_initialized()
        return store

    def raw(self) -> bytes:
        return self.path.read_bytes()

    def assert_error(self, code: str, func, *args):
        """ScheduleStoreError의 code를 확인하고 오류를 돌려준다."""
        with self.assertRaises(ss.ScheduleStoreError) as caught:
            func(*args)
        self.assertEqual(caught.exception.code, code, str(caught.exception))
        return caught.exception


def new_action(payload: dict[str, Any] | None = None, kind: str = "create_schedule") -> tuple[str, str]:
    """(action_id, content_hash)"""
    action = {"type": kind, "payload": payload or {"title": "테스트", "n": uuid.uuid4().hex}}
    return uuid.uuid4().hex, ss.content_hash(action)


def add_schedule(title: str, start: datetime, minutes: int = 60, now: datetime = FIXED_NOW):
    """새 일정을 하나 추가하는 변경 함수 (저장소 테스트용. 실제 등록 검사는 schedule_service가 맡는다)"""
    def mutate(doc: dict[str, Any]) -> dict[str, Any]:
        item = {"schedule_id": ss.next_schedule_id(doc), "title": title, "aliases": [], "start_at": ss.iso(start),
                "end_at": ss.iso(start + timedelta(minutes=minutes)), "city": None, "category": "home",
                "status": "scheduled", "reschedulable": True, "version": 1, "source": "user",
                "created_at": ss.iso(now), "updated_at": ss.iso(now), "completed_at": None, "cancelled_at": None}
        doc["schedules"].append(item)
        return {"ok": True, "history": {"operation": "create", "schedule_id": item["schedule_id"], "before": None,
                                        "after": copy.deepcopy(item), "reason": "테스트 등록", "approved_at": ss.iso(now)}}
    return mutate


# %% M-ST1. 최초 초기화와 영구 저장 (요구 06·12)
class M_ST1_Initialization(StoreCase):

    def test_ST01_first_run_creates_seed_relative_to_now(self):
        """ST01 처음 실행하면 초기 일정 4개를 '오늘 기준 상대 날짜'로 만들고 표시 파일을 남긴다."""
        result = self.store().ensure_initialized()
        self.assertEqual((result["status"], result["schedules"]), ("created", 4))
        doc = json.loads(self.raw().decode("utf-8"))
        self.assertEqual(ss.validate_document(doc), [])
        starts = {s["title"]: s["start_at"] for s in doc["schedules"]}
        self.assertEqual(starts["시장 장보기"], "2026-11-26T14:00:00+09:00")
        self.assertEqual(starts["복지관 방문"], "2026-11-26T16:00:00+09:00")
        self.assertEqual(starts["병원 진료"], "2026-11-28T10:00:00+09:00")
        hospital = next(s for s in doc["schedules"] if s["title"] == "병원 진료")
        self.assertEqual((hospital["end_at"], hospital["reschedulable"]), ("2026-11-28T11:30:00+09:00", False))
        self.assertTrue(self.store().marker_path.exists())
        # 같은 초기 데이터라도 다른 날 처음 만들면 날짜가 달라진다. (날짜를 코드·데이터에 고정하지 않음)
        self.now = datetime(2027, 1, 31, 9, 0, tzinfo=KST)
        other = ss.ScheduleStore(Path(self.temp.name) / "other.json", clock=lambda: self.now)
        other.ensure_initialized()
        market = next(s for s in other.load()["schedules"] if s["title"] == "시장 장보기")
        self.assertEqual(market["start_at"], "2027-02-01T14:00:00+09:00")

    def test_ST02_existing_store_is_never_overwritten(self):
        """ST02 저장소가 있으면 초기화를 다시 불러도 덮어쓰지 않는다. (다른 날 실행해도 같음)"""
        store = self.ready()
        action_id, digest = new_action()
        store.transact(action_id, digest, add_schedule("복지관 체조", FIXED_NOW + timedelta(days=2)))
        before = self.raw()
        self.now = FIXED_NOW + timedelta(days=10)
        self.assertEqual(self.store().ensure_initialized()["status"], "existing")
        self.assertEqual(self.raw(), before)

    def test_ST03_persists_across_restart(self):
        """ST03 프로그램을 다시 실행한 것처럼 새 저장소 객체로 읽어도 변경이 남아 있다."""
        store = self.ready()
        action_id, digest = new_action()
        applied = store.transact(action_id, digest, add_schedule("친구 만나기", FIXED_NOW + timedelta(days=1, hours=1)))
        self.assertEqual(applied["status"], "applied")
        restarted = ss.ScheduleStore(self.path, clock=lambda: self.now)
        titles = [s["title"] for s in restarted.load()["schedules"]]
        self.assertIn("친구 만나기", titles)
        self.assertEqual(restarted.load()["action_history"][0]["action_id"], action_id)

    def test_ST04_missing_store_with_marker_is_not_recreated(self):
        """ST04 처음 만든 뒤 운영 저장소가 사라지면 다시 만들지 않고 복구를 안내한다."""
        store = self.ready()
        self.path.unlink()
        error = self.assert_error("store_missing", store.ensure_initialized)
        self.assertIn("다시 만들지 않았습니다", str(error))
        self.assert_error("store_missing", store.load)
        self.assertFalse(self.path.exists())

    def test_ST05_not_initialized_before_first_run(self):
        """ST05 초기화 전에는 읽지 않는다. (읽기가 파일을 만들지 않음)"""
        self.assert_error("not_initialized", self.store().load)
        self.assertFalse(self.path.exists())

    def test_ST06_marker_restored_only_when_store_is_valid(self):
        """ST06 저장소만 있고 표시 파일이 없으면(쓰기 중 종료) 일정은 그대로 두고 표시 파일만 보충한다."""
        store = self.ready()
        store.marker_path.unlink()
        before = self.raw()
        self.assertEqual(store.ensure_initialized()["status"], "existing")
        self.assertTrue(store.marker_path.exists())
        self.assertEqual(self.raw(), before)

    def test_ST07_bad_seed_creates_nothing(self):
        """ST07 초기 일정 파일이 틀리면 아무 파일도 만들지 않는다."""
        seed = Path(self.temp.name) / "seed.json"
        for content in ("{", json.dumps({"schema_version": 1, "schedules": [{"title": "x", "day_offset": -1,
                                                                             "start": "09:00", "minutes": 30}]}),
                        json.dumps({"schema_version": 1, "schedules": [{"title": "x", "day_offset": 0,
                                                                        "start": "25:00", "minutes": 30}]})):
            with self.subTest(content=content[:30]):
                seed.write_text(content, encoding="utf-8")
                self.assert_error("seed_error", self.store(seed_path=seed).ensure_initialized)
                self.assertFalse(self.path.exists())
                self.assertFalse(self.store().marker_path.exists())


# %% M-ST2. 손상·권한·쓰기 실패에서 기존 데이터 보존 (요구 24)
class M_ST2_FileErrors(StoreCase):

    def test_ST08_corrupt_json_is_preserved(self):
        """ST08 잘못된 JSON은 읽지 않고, 초기화도 덮어쓰지 않는다."""
        store = self.ready()
        self.path.write_bytes(b'{"schema_version": 1, "schedules": [')
        before = self.raw()
        self.assert_error("corrupt", store.load)
        self.assert_error("corrupt", store.ensure_initialized)
        action_id, digest = new_action()
        self.assert_error("corrupt", store.transact, action_id, digest, add_schedule("x", FIXED_NOW))
        self.assertEqual(self.raw(), before)

    def test_ST09_schema_errors_are_preserved(self):
        """ST09 구조가 틀린 문서(키 누락·상태 값·시각 순서·id 중복·이력 형식)는 거절하고 파일을 바꾸지 않는다."""
        store = self.ready()
        good = json.loads(self.raw().decode("utf-8"))

        def broken(change):
            doc = copy.deepcopy(good)
            change(doc)
            return doc

        cases = {
            "missing_key": broken(lambda d: d["schedules"][0].pop("status")),
            "bad_status": broken(lambda d: d["schedules"][0].update(status="done")),
            "end_before_start": broken(lambda d: d["schedules"][0].update(end_at=d["schedules"][0]["start_at"])),
            "no_timezone": broken(lambda d: d["schedules"][0].update(start_at="2026-11-26T14:00:00")),
            "duplicate_id": broken(lambda d: d["schedules"][1].update(schedule_id=d["schedules"][0]["schedule_id"])),
            "completed_without_time": broken(lambda d: d["schedules"][0].update(status="completed")),
            "bad_history": broken(lambda d: d["action_history"].append({"action_id": "x"})),
            "version_zero": broken(lambda d: d["schedules"][0].update(version=0)),
            "schema_version": broken(lambda d: d.update(schema_version=2)),
            "not_dict": [],
        }
        for name, doc in cases.items():
            with self.subTest(name):
                self.path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
                before = self.raw()
                self.assert_error("schema_error", store.load)
                self.assertEqual(self.raw(), before)

    def test_ST10_read_permission_denied(self):
        """ST10 읽기 권한이 없으면 read_denied로 알린다."""
        store = self.ready()
        with mock.patch.object(Path, "read_bytes", side_effect=PermissionError("denied")):
            self.assert_error("read_denied", store.load)

    def test_ST11_replace_failure_keeps_original_and_cleans_temp(self):
        """ST11 파일 교체가 계속 실패하면 write_failed. 기존 파일은 그대로, 임시 파일은 남지 않는다."""
        store = self.ready()
        before = self.raw()
        action_id, digest = new_action()
        with mock.patch.object(ss.os, "replace", side_effect=PermissionError("in use")), \
                mock.patch.object(ss, "REPLACE_RETRY_SECONDS", 0):
            self.assert_error("write_failed", store.transact, action_id, digest, add_schedule("x", FIXED_NOW))
        self.assertEqual(self.raw(), before)
        self.assertEqual([p.name for p in self.path.parent.glob("*.tmp")], [])
        # 실패한 작업은 적용되지 않았으므로 같은 id로 다시 시도하면 적용된다.
        self.assertEqual(store.transact(action_id, digest, add_schedule("x", FIXED_NOW))["status"], "applied")

    def test_ST12_write_permission_denied(self):
        """ST12 임시 파일을 만들 권한이 없으면 write_denied. 기존 파일은 그대로다."""
        store = self.ready()
        before = self.raw()
        action_id, digest = new_action()
        real_open = open

        def deny_temp(file, *args, **kwargs):
            if str(file).endswith(".tmp"):
                raise PermissionError("denied")
            return real_open(file, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=deny_temp):
            self.assert_error("write_denied", store.transact, action_id, digest, add_schedule("x", FIXED_NOW))
        self.assertEqual(self.raw(), before)

    def test_ST13_replace_retry_succeeds(self):
        """ST13 Windows에서 잠깐 잡혀 있던 파일은 다시 시도해 저장한다."""
        store = self.ready()
        real_replace = os.replace
        calls = []

        def flaky(src, dst):
            calls.append(1)
            if len(calls) == 1:
                raise PermissionError("busy")
            return real_replace(src, dst)

        action_id, digest = new_action()
        with mock.patch.object(ss.os, "replace", side_effect=flaky), mock.patch.object(ss, "REPLACE_RETRY_SECONDS", 0):
            self.assertEqual(store.transact(action_id, digest, add_schedule("x", FIXED_NOW))["status"], "applied")
        self.assertEqual(len(calls), 2)


# %% M-ST3. 한 작업은 한 번만, 거절된 작업은 기록하지 않음 (요구 20·22·25)
class M_ST3_Idempotency(StoreCase):

    def test_ST14_same_action_applied_once_even_after_restart(self):
        """ST14 같은 action_id·같은 내용은 한 번만 적용한다. 다시 실행한 프로그램에서도 같다."""
        store = self.ready()
        action_id, digest = new_action()
        mutate = add_schedule("산책 모임", FIXED_NOW + timedelta(days=1))
        self.assertEqual(store.transact(action_id, digest, mutate)["status"], "applied")
        after_first = self.raw()
        again = store.transact(action_id, digest, mutate)
        self.assertEqual((again["status"], again["record"]["action_id"]), ("already_executed", action_id))
        restarted = ss.ScheduleStore(self.path, clock=lambda: self.now)
        self.assertEqual(restarted.transact(action_id, digest, mutate)["status"], "already_executed")
        self.assertEqual(self.raw(), after_first)
        self.assertEqual(len(restarted.load()["action_history"]), 1)

    def test_ST15_reused_action_id_with_other_content_is_rejected(self):
        """ST15 같은 action_id에 다른 내용이 오면 거절하고 저장하지 않는다."""
        store = self.ready()
        action_id, digest = new_action({"title": "A"})
        store.transact(action_id, digest, add_schedule("A", FIXED_NOW + timedelta(days=1)))
        before = self.raw()
        other_digest = ss.content_hash({"type": "create_schedule", "payload": {"title": "B"}})
        result = store.transact(action_id, other_digest, add_schedule("B", FIXED_NOW + timedelta(days=2)))
        self.assertEqual((result["status"], result["code"]), ("rejected", "action_id_reused"))
        self.assertEqual(self.raw(), before)

    def test_ST16_rejected_or_invalid_change_writes_nothing(self):
        """ST16 변경 함수가 거절하거나, 규격을 어기는 변경을 만들거나, 예외를 내면 파일을 바꾸지 않는다."""
        store = self.ready()
        before = self.raw()

        def refuse(doc):
            return {"ok": False, "code": "conflict", "message": "다른 일정과 겹칩니다."}

        def corrupt(doc):
            doc["schedules"][0]["status"] = "done"
            return {"ok": True, "history": {"operation": "update", "schedule_id": "schedule_001", "before": None,
                                            "after": None, "reason": "", "approved_at": ss.iso(FIXED_NOW)}}

        def crash(doc):
            raise KeyError("bug")

        for name, mutate, expected in (("refuse", refuse, "conflict"), ("corrupt", corrupt, "invalid_change")):
            with self.subTest(name):
                action_id, digest = new_action()
                result = store.transact(action_id, digest, mutate)
                self.assertEqual((result["status"], result["code"]), ("rejected", expected))
        action_id, digest = new_action()
        self.assert_error("mutation_error", store.transact, action_id, digest, crash)
        self.assertEqual(self.raw(), before)
        self.assertEqual(store.load()["action_history"], [])

    def test_ST17_bad_action_id_or_hash(self):
        """ST17 작업 id·해시 형식이 틀리면 파일을 열기 전에 거절한다."""
        store = self.ready()
        cases: list[tuple[Any, Any]] = [("short", "0" * 64), ("a" * 32, "x"), (["a" * 32], "0" * 64)]
        for action_id, digest in cases:
            result = store.transact(action_id, digest, add_schedule("x", FIXED_NOW))
            self.assertEqual((result["status"], result["code"]), ("rejected", "invalid_action"))

    def test_ST18_history_keeps_before_after_and_times(self):
        """ST18 이력 한 줄에 작업 종류·일정 id·변경 전후·승인·실행 시각·결과가 남는다."""
        store = self.ready()
        action_id, digest = new_action()
        record = store.transact(action_id, digest, add_schedule("체조", FIXED_NOW + timedelta(days=1)))["record"]
        self.assertEqual(set(record), set(ss.HISTORY_KEYS))
        self.assertEqual((record["operation"], record["result"], record["before"]), ("create", "applied", None))
        self.assertEqual(record["after"]["title"], "체조")
        self.assertEqual(record["executed_at"], ss.iso(FIXED_NOW))


# %% M-ST4. 동시 요청과 잠금 (요구 24)
class M_ST4_Concurrency(StoreCase):

    def test_ST19_parallel_threads_lose_no_update(self):
        """ST19 서로 다른 작업 8개를 동시에 적용해도 하나도 빠지지 않는다."""
        store = self.ready()
        results, threads = [], []
        for number in range(8):
            action_id, digest = new_action()
            mutate = add_schedule(f"동시 {number}", FIXED_NOW + timedelta(days=1, hours=number))
            threads.append(threading.Thread(target=lambda a=action_id, d=digest, m=mutate: results.append(
                ss.ScheduleStore(self.path, clock=lambda: self.now).transact(a, d, m)["status"])))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        doc = store.load()
        self.assertEqual(results.count("applied"), 8)
        self.assertEqual((len(doc["schedules"]), len(doc["action_history"])), (12, 8))
        self.assertEqual(len({s["schedule_id"] for s in doc["schedules"]}), 12)

    def test_ST20_same_action_in_parallel_applies_once(self):
        """ST20 같은 작업이 동시에 5번 들어와도(버튼 연타) 한 번만 적용된다."""
        store = self.ready()
        action_id, digest = new_action()
        mutate = add_schedule("한 번만", FIXED_NOW + timedelta(days=1))
        results = []
        threads = [threading.Thread(target=lambda: results.append(store.transact(action_id, digest, mutate)["status"]))
                   for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(results), ["already_executed"] * 4 + ["applied"])
        self.assertEqual(len(store.load()["action_history"]), 1)

    def test_ST21_foreign_old_lock_is_not_removed(self):
        """ST21 다른 프로그램의 잠금 파일은 오래됐어도 지우지 않는다. 저장을 중단하고 복구 방법을 안내한다."""
        store = self.ready(lock_wait=0.2)
        store.lock_path.write_text(json.dumps({"token": "other", "pid": 1}), encoding="utf-8")
        old = datetime.now().timestamp() - 3600
        os.utime(store.lock_path, (old, old))
        before = self.raw()
        action_id, digest = new_action()
        error = self.assert_error("lock_timeout", store.transact, action_id, digest, add_schedule("x", FIXED_NOW))
        self.assertIn("직접 삭제", str(error))
        self.assertIn("바뀌지 않았습니다", str(error))
        self.assertTrue(store.lock_path.exists())
        self.assertEqual(json.loads(store.lock_path.read_text(encoding="utf-8"))["token"], "other")
        self.assertEqual(self.raw(), before)
        self.assert_error("lock_timeout", store.ensure_initialized)

    def test_ST22_own_lock_is_released_and_only_own(self):
        """ST22 작업이 끝나면 내 잠금 파일은 지운다. 토큰이 다르면(남의 잠금) 지우지 않는다."""
        store = self.ready()
        action_id, digest = new_action()
        store.transact(action_id, digest, add_schedule("x", FIXED_NOW + timedelta(days=1)))
        self.assertFalse(store.lock_path.exists())
        lock = ss.FileLock(store.lock_path, 0.2)
        with lock:
            store.lock_path.write_text(json.dumps({"token": "someone-else"}), encoding="utf-8")
        self.assertTrue(store.lock_path.exists())

    def test_ST23_separate_process_waits_for_lock(self):
        """ST23 다른 프로세스가 잠금을 쥐고 있으면 기다렸다가, 풀린 뒤 저장한다. (실제 별도 파이썬 프로세스)"""
        store = self.ready(lock_wait=10)
        script = ("import sys, time, schedule_store as ss\n"
                  "from pathlib import Path\n"
                  "with ss.FileLock(Path(sys.argv[1]), 5):\n"
                  "    print('locked', flush=True)\n"
                  "    time.sleep(1.0)\n")
        proc = subprocess.Popen([sys.executable, "-c", script, str(store.lock_path)], cwd=ROOT,
                                stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(proc.stdout.readline().strip() if proc.stdout else "", "locked")
            action_id, digest = new_action()
            result = store.transact(action_id, digest, add_schedule("기다림", FIXED_NOW + timedelta(days=1)))
            self.assertEqual(result["status"], "applied")
        finally:
            proc.wait(timeout=20)
            if proc.stdout:
                proc.stdout.close()
        self.assertFalse(store.lock_path.exists())


# %% M-ST5. 초기 데이터·경로·인코딩 (요구 06·12, 7.6 실행 환경)
class M_ST5_DataAndPaths(StoreCase):

    def test_ST24_repository_seed_is_public_and_relative(self):
        """ST24 저장소에 올라가는 초기 일정은 규격을 지키고, 날짜를 고정하지 않으며, 연락처 같은 개인정보가 없다."""
        text = ss.DEFAULT_SEED_PATH.read_text(encoding="utf-8")
        seed = json.loads(text)
        self.assertEqual(ss.validate_document(ss.build_seed_document(seed, FIXED_NOW)), [])
        self.assertIsNone(re.search(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}", text))
        self.assertIsNone(re.search(r"01[0-9]-?[0-9]{3,4}-?[0-9]{4}|@", text))
        self.assertIn("가상", seed["description"])

    def test_ST25_default_path_and_env_override(self):
        """ST25 기본 경로는 프로젝트 폴더 기준 .safemate 아래, 환경변수로 바꿀 수 있다. 코드에 Windows 절대 경로가 없다."""
        self.assertEqual(ss.default_store_path(), ROOT / ".safemate" / "schedule_store.json")
        custom = Path(self.temp.name) / "drive" / "store.json"
        with mock.patch.dict(os.environ, {ss.STORE_PATH_ENV: str(custom)}):
            self.assertEqual(ss.ScheduleStore().path, custom)
        source = (ROOT / "schedule_store.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"[A-Za-z]:\\\\|C:/", source))

    def test_ST26_utf8_korean_round_trip(self):
        """ST26 한글 제목이 UTF-8 JSON으로 그대로 저장되고 다시 읽힌다."""
        store = self.ready()
        action_id, digest = new_action()
        store.transact(action_id, digest, add_schedule("손주 생일 잔치", FIXED_NOW + timedelta(days=2)))
        self.assertIn("손주 생일 잔치", self.raw().decode("utf-8"))
        self.assertIn("손주 생일 잔치", [s["title"] for s in store.load()["schedules"]])

    def test_ST27_load_returns_copy(self):
        """ST27 load()가 돌려준 문서를 고쳐도 파일은 바뀌지 않는다."""
        store = self.ready()
        before = self.raw()
        doc = store.load()
        doc["schedules"].clear()
        self.assertEqual(self.raw(), before)
        self.assertEqual(len(store.load()["schedules"]), 4)


# %% M-ST6. 외부 코드 검토에서 재현된 예외 처리 회귀 테스트
class M_ST6_ReviewRegressions(StoreCase):

    def test_ST28_unhashable_ids_are_schema_errors(self):
        """ST28 schedule_id·action_id에 리스트·딕셔너리가 와도 TypeError 없이 schema_error로 처리하고 파일을 바꾸지 않는다."""
        store = self.ready()
        good = json.loads(self.raw().decode("utf-8"))
        history_entry = {"action_id": "a" * 32, "content_hash": "0" * 64, "operation": "create",
                         "schedule_id": "schedule_001", "before": None, "after": None, "reason": "",
                         "approved_at": ss.iso(FIXED_NOW), "executed_at": ss.iso(FIXED_NOW), "result": "applied"}
        for bad in ([], {}, ["schedule_001"], {"id": 1}):
            with self.subTest(bad=bad):
                doc = copy.deepcopy(good)
                doc["schedules"][0]["schedule_id"] = bad
                doc["action_history"] = [dict(history_entry, action_id=bad), dict(history_entry, action_id="b" * 32)]
                problems = ss.validate_document(doc)
                self.assertTrue(any("schedule_id 형식" in p for p in problems))
                self.assertTrue(any("action_id 형식" in p for p in problems))
                self.path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
                before = self.raw()
                self.assert_error("schema_error", store.load)
                self.assert_error("schema_error", store.ensure_initialized)
                self.assertEqual(self.raw(), before)

    def test_ST29_lock_write_failure_removes_only_own_lock(self):
        """ST29 잠금 파일을 만든 뒤 정보 기록(json.dump)이나 파일 열기(fdopen)에 실패하면,
        방금 만든 내 잠금 파일만 지우고 lock_error로 알린다. 일정 파일은 그대로이고 다음 저장은 정상이다."""
        store = self.ready()
        before = self.raw()
        for target, name in ((ss.json, "dump"), (ss.os, "fdopen")):
            with self.subTest(name):
                action_id, digest = new_action()
                with mock.patch.object(target, name, side_effect=OSError("disk error")):
                    self.assert_error("lock_error", store.transact, action_id, digest, add_schedule("x", FIXED_NOW))
                self.assertFalse(store.lock_path.exists())
                self.assertEqual(self.raw(), before)
        action_id, digest = new_action()
        self.assertEqual(store.transact(action_id, digest, add_schedule("정상", FIXED_NOW + timedelta(days=1)))["status"],
                         "applied")
        self.assertFalse(store.lock_path.exists())

    def test_ST30_lock_write_failure_never_touches_foreign_lock(self):
        """ST30 다른 프로그램의 잠금 파일이 있으면 기록 단계까지 가지 않으므로(FileExistsError) 그 파일은 그대로 남는다."""
        store = self.ready(lock_wait=0.2)
        store.lock_path.write_text(json.dumps({"token": "other"}), encoding="utf-8")
        action_id, digest = new_action()
        with mock.patch.object(ss.json, "dump", side_effect=OSError("disk error")):
            self.assert_error("lock_timeout", store.transact, action_id, digest, add_schedule("x", FIXED_NOW))
        self.assertEqual(json.loads(store.lock_path.read_text(encoding="utf-8"))["token"], "other")


# %% 일정 서비스 테스트 도우미
# 핵심 요약: 초기 가상 일정(오늘 08:00 산책, 내일 14:00 시장, 내일 16:00 복지관(제한), 11/28 10:00 병원(제한))으로 문서를 만든다.
def got(result: Any, key: str) -> Any:
    """결과 딕셔너리에서 값을 꺼낸다. 딕셔너리가 아니거나 키가 없으면 테스트 실패로 알린다."""
    if not isinstance(result, dict) or key not in result:
        raise AssertionError(f"{key} 없음: {result}")
    return result[key]


def seed_doc(now: datetime = FIXED_NOW) -> dict[str, Any]:
    return ss.build_seed_document(json.loads(ss.DEFAULT_SEED_PATH.read_text(encoding="utf-8")), now)


def at(day: int, hour: int, minute: int = 0) -> datetime:
    """2026-11-<day> hour:minute KST"""
    return datetime(2026, 11, day, hour, minute, tzinfo=KST)


def item_of(doc: dict[str, Any], title: str) -> dict[str, Any]:
    return next(s for s in doc["schedules"] if s["title"] == title)


def approved(action: dict[str, Any], now: datetime = FIXED_NOW) -> dict[str, Any]:
    """그래프가 사용자 승인을 확인한 뒤 붙이는 approval을 흉내 낸다. (원본 봉투는 고치지 않음)"""
    return dict(copy.deepcopy(action), approval={"status": "approved", "at": ss.iso(now)})


def with_item(doc: dict[str, Any], title: str, start: datetime, minutes: int = 60, status: str = "scheduled",
              reschedulable: bool = True) -> dict[str, Any]:
    """테스트용 일정을 문서에 직접 넣는다."""
    item = {"schedule_id": ss.next_schedule_id(doc), "title": title, "aliases": [], "start_at": ss.iso(start),
            "end_at": ss.iso(start + timedelta(minutes=minutes)), "city": "인천", "category": "other",
            "status": status, "reschedulable": reschedulable, "version": 1, "source": "user",
            "created_at": ss.iso(FIXED_NOW), "updated_at": ss.iso(FIXED_NOW),
            "completed_at": ss.iso(FIXED_NOW) if status == "completed" else None,
            "cancelled_at": ss.iso(FIXED_NOW) if status == "cancelled" else None}
    doc["schedules"].append(item)
    return item


# %% M-SV1. 자연어 날짜·시각·기간·상대 시간 (요구 01·02·03, 6절 날짜·시간 예외)
class M_SV1_NaturalTime(unittest.TestCase):

    def test_SV01_clock_expressions(self):
        """SV01 어르신이 쓰는 시각 표현을 정확한 HH:MM으로 바꾸고, 오전·오후가 없으면 추측하지 않는다."""
        expected = {"아침 여덟 시": "08:00", "아침 8시": "08:00", "오전 아홉 시 반": "09:30", "낮 열두 시": "12:00",
                    "오후 두 시": "14:00", "오후 네 시쯤": "16:00", "저녁 일곱 시": "19:00", "저녁 7시": "19:00",
                    "밤 아홉 시": "21:00", "밤 9시": "21:00", "새벽 한 시": "01:00", "오후 네 시 십오 분": "16:15",
                    "오후 네 시 반": "16:30", "오후 4시 사십오 분": "16:45", "오후 12시": "12:00", "오전 12시": "00:00",
                    "오후 네 시 조금 넘어서": "16:00"}
        for text, clock in expected.items():
            with self.subTest(text):
                self.assertEqual(got(sv.parse_clock(text), "time"), clock)
        for text in ("세 시 반", "네 시", "네 시 조금 넘어서", "여덟 시", "밤 12시"):
            with self.subTest(text):
                self.assertIn("ambiguous", sv.parse_clock(text) or {})
        self.assertTrue(got(sv.parse_clock("오후 네 시쯤"), "approximate"))
        self.assertTrue(got(sv.parse_clock("오후 네 시 조금 넘어서"), "approximate"))
        self.assertFalse(got(sv.parse_clock("오후 네 시"), "approximate"))
        self.assertIsNone(sv.parse_clock("시장에 갈 거야"))

    def test_SV02_correction_inherits_half_day(self):
        """SV02 '저녁 일곱 시 아니고 여덟 시': 직전 시각(19:00)의 오후를 이어받아 20:00, 이어받았다고 표시한다."""
        self.assertEqual(sv.parse_clock("여덟 시", inherit_from="19:00"),
                         {"time": "20:00", "approximate": False, "inherited": True})
        self.assertEqual(got(sv.parse_clock("열 시", inherit_from="09:00"), "time"), "10:00")
        self.assertIn("ambiguous", sv.parse_clock("여덟 시") or {})

    def test_SV03_dates_and_boundaries(self):
        """SV03 상대 날짜·요일·다음 달·연말연초·윤년·없는 날짜·주말·그날/다음 날"""
        now = FIXED_NOW
        cases = {"오늘": "2026-11-25", "내일": "2026-11-26", "모레": "2026-11-27", "글피": "2026-11-28",
                 "이번 주 금요일": "2026-11-27", "다음 주 월요일": "2026-11-30", "12월 5일": "2026-12-05",
                 "다음 달 3일": "2026-12-03", "일주일 뒤": "2026-12-02", "2주 뒤": "2026-12-09"}
        for text, day in cases.items():
            with self.subTest(text):
                self.assertEqual(got(sv.parse_day(text, now), "date"), day)
        # '다음 달 3일'은 이번 달 3일이 아직 안 지났어도 다음 달이다. (3단계 함수만으로는 이번 달로 읽음)
        early = datetime(2026, 11, 1, 9, 0, tzinfo=KST)
        self.assertEqual(got(sv.parse_day("다음 달 3일", early), "date"), "2026-12-03")
        self.assertEqual(got(sv.parse_day("다음 달 3일", datetime(2026, 12, 30, tzinfo=KST)), "date"), "2027-01-03")
        self.assertEqual(got(sv.parse_day("1월 3일", datetime(2026, 12, 30, tzinfo=KST)), "date"), "2027-01-03")
        self.assertEqual(got(sv.parse_day("2월 29일", datetime(2028, 1, 10, tzinfo=KST)), "date"), "2028-02-29")
        for text, moment in (("2월 29일", datetime(2027, 1, 10, tzinfo=KST)), ("2월 30일", now),
                             ("다음 달 31일", datetime(2027, 1, 10, tzinfo=KST)), ("이번 달 3일", now), ("이번 주말", now), ("그날", now), ("다음 날", now)):
            with self.subTest(text):
                self.assertIn("ambiguous", sv.parse_day(text, moment) or {})
        self.assertEqual(got(sv.parse_day("그날", now, "2026-11-28"), "date"), "2026-11-28")
        self.assertEqual(got(sv.parse_day("다음 날", now, "2026-11-30"), "date"), "2026-12-01")

    def test_SV04_durations(self):
        """SV04 소요 시간: 한글 수 보강('열두 시간'을 120분으로 잘못 읽지 않음), 범위 밖은 None"""
        cases = {"한 시간": 60, "한 시간 반": 90, "삼십 분": 30, "30분": 30, "두 시간": 120, "열두 시간": 720,
                 "1시간 15분": 75, "한 시간 십오 분": 75}
        for text, minutes in cases.items():
            with self.subTest(text):
                self.assertEqual(sv.parse_minutes(text), minutes)
        self.assertIsNone(sv.parse_minutes("5분"))
        self.assertIsNone(sv.parse_minutes("금방"))

    def test_SV05_relative_shift_and_midnight(self):
        """SV05 상대 시간: 일정 기준·지금 기준 계산, 자정 넘김, 방향을 모르면 되묻기"""
        base = at(26, 14)
        self.assertEqual(got(sv.resolve_shift("한 시간만 미뤄 줘", base), "at"), at(26, 15))
        self.assertEqual(got(sv.resolve_shift("원래 시간보다 30분 앞당겨 줘", base), "at"), at(26, 13, 30))
        self.assertEqual(got(sv.resolve_shift("다음 날 같은 시간", base), "at"), at(27, 14))
        self.assertEqual(got(sv.resolve_shift("일주일 뒤", base), "at"), datetime(2026, 12, 3, 14, tzinfo=KST))
        self.assertEqual(got(sv.resolve_shift("지금부터 9시간 뒤", FIXED_NOW), "at"), at(25, 19))
        self.assertEqual(got(sv.resolve_shift("두 시간 후", at(25, 23, 30)), "at"), at(26, 1, 30))
        self.assertEqual(got(sv.resolve_shift("한 시간 일찍", at(26, 0, 30)), "at"), at(25, 23, 30))
        self.assertIn("ambiguous", sv.resolve_shift("두 시간", base) or {})
        self.assertIsNone(sv.resolve_shift("시간 좀 바꿔 줘", base))

    def test_SV06_korean_time_words(self):
        """SV06 사용자 안내용 시각: 오전·오후, 자정, 정오"""
        self.assertEqual(sv.korean_time(at(26, 0)), "밤 12시")
        self.assertEqual(sv.korean_time(at(26, 12)), "낮 12시")
        self.assertEqual(sv.korean_time(at(26, 16, 30)), "오후 4시 30분")
        self.assertEqual(sv.korean_time(at(26, 9, 5)), "오전 9시 5분")


# %% M-SV2. 조회·상태 표시 (요구 07·14·26·27·41·42)
class M_SV2_Queries(unittest.TestCase):

    def setUp(self):
        self.doc = seed_doc()

    def test_SV07_day_week_and_display_state(self):
        """SV07 날짜·이번 주 조회, 시간이 지난 일정은 '완료 기록 없음'으로만 표시하고 상태를 바꾸지 않는다."""
        tomorrow = [s["title"] for s in sv.schedules_on(self.doc, at(26, 0).date())]
        self.assertEqual(tomorrow, ["시장 장보기", "복지관 방문"])
        first, last = sv.week_range(FIXED_NOW)
        self.assertEqual((first.isoformat(), last.isoformat()), ("2026-11-25", "2026-11-29"))
        self.assertEqual(len(sv.schedules_between(self.doc, first, last)), 4)
        walk = item_of(self.doc, "아침 산책")
        self.assertEqual((sv.display_state(walk, FIXED_NOW), walk["status"]), ("overdue", "scheduled"))
        self.assertIn("시간 지남(완료 기록 없음)", sv.describe(walk, FIXED_NOW))
        self.assertEqual(sv.display_state(walk, at(25, 8, 10)), "running")
        self.assertEqual(sv.schedules_on(self.doc, at(29, 0).date()), [])

    def test_SV08_next_running_remaining(self):
        """SV08 다음 일정은 지금 이후 시작하는 예정 일정만. 진행 중은 따로, 완료·취소는 제외"""
        self.assertEqual(got(sv.next_schedule(self.doc, FIXED_NOW), "title"), "시장 장보기")
        with_item(self.doc, "점심 약속", at(25, 12), status="cancelled")
        with_item(self.doc, "체조", at(25, 11), status="completed")
        lunch = with_item(self.doc, "복지관 점심", at(25, 12, 30))
        self.assertEqual(sv.next_schedule(self.doc, FIXED_NOW), lunch)
        self.assertEqual(sv.running_schedules(self.doc, at(25, 12, 40)), [lunch])
        self.assertEqual([s["title"] for s in sv.remaining_today(self.doc, FIXED_NOW)], ["복지관 점심"])
        self.assertEqual(sv.next_schedule(self.doc, at(28, 12)), None)

    def test_SV09_midnight_crossing_appears_on_both_days(self):
        """SV09 자정을 넘는 일정은 두 날짜 조회에 모두 나온다."""
        night = with_item(self.doc, "야간 기도회", at(26, 23), minutes=120)
        self.assertIn(night, sv.schedules_on(self.doc, at(26, 0).date()))
        self.assertIn(night, sv.schedules_on(self.doc, at(27, 0).date()))
        self.assertIn("모레", sv.describe(night, FIXED_NOW))


# %% M-SV3. 충돌·빈 시간·예보 상태 (요구 09·11, 보완 ③)
class M_SV3_ConflictAndSlots(unittest.TestCase):

    def setUp(self):
        self.doc = seed_doc()
        self.market = item_of(self.doc, "시장 장보기")

    def test_SV10_conflicts(self):
        """SV10 부분 겹침은 충돌, 끝과 시작이 맞닿으면 충돌 아님, 취소 일정 무시, 자정 넘김 비교"""
        self.assertEqual([c["title"] for c in sv.find_conflicts(self.doc, at(26, 16, 30), at(26, 17, 30))], ["복지관 방문"])
        self.assertEqual(sv.find_conflicts(self.doc, at(26, 15), at(26, 16)), [])
        self.assertEqual(sv.find_conflicts(self.doc, at(26, 14), at(26, 15), exclude_id=self.market["schedule_id"]), [])
        with_item(self.doc, "취소한 약속", at(26, 18), status="cancelled")
        self.assertEqual(sv.find_conflicts(self.doc, at(26, 18), at(26, 19)), [])
        night = with_item(self.doc, "야간 일정", at(26, 23), minutes=120)
        self.assertEqual(sv.find_conflicts(self.doc, at(27, 0, 30), at(27, 1)), [night])

    def test_SV11_free_slots_prefer_requested_time(self):
        """SV11 오후 4시를 원했지만 복지관과 겹침 → 가까운 순(같은 거리면 늦은 쪽 먼저) 최대 3개, 소요 시간 유지"""
        slots = sv.find_free_slots(self.doc, day=at(26, 0).date(), minutes=60, now=FIXED_NOW, prefer=at(26, 16),
                                   exclude_id=self.market["schedule_id"])
        self.assertEqual(slots, [at(26, 17), at(26, 15), at(26, 17, 30)])

    def test_SV12_free_slots_limits(self):
        """SV12 지난 시각 제외(지금+10분), 하루가 차면 빈 목록, 허락한 경우만 다음 날, 자동 범위(07~21시)"""
        today = sv.find_free_slots(self.doc, day=at(25, 0).date(), minutes=60, now=FIXED_NOW)
        self.assertEqual(today[0], at(25, 10, 30))
        full = seed_doc()
        with_item(full, "종일 행사", at(27, 6), minutes=16 * 60)
        self.assertEqual(sv.find_free_slots(full, day=at(27, 0).date(), minutes=60, now=FIXED_NOW), [])
        next_day = sv.find_free_slots(full, day=at(27, 0).date(), minutes=60, now=FIXED_NOW, extra_days=1)
        self.assertEqual(next_day[0], at(28, 7))
        late = sv.find_free_slots(self.doc, day=at(26, 0).date(), minutes=60, now=FIXED_NOW, prefer=at(26, 22))
        self.assertTrue(all(7 <= s.hour and s + timedelta(hours=1) <= at(26, 21) for s in late))

    def test_SV13_forecast_status(self):
        """SV13 예보 상태: 조회 성공·신호 없음=no_signal, 신호 있음=signal, 실패·누락·형식 오류=unknown
        (검토 반영: 'good·bad'는 신호가 없으면 안전한 것처럼 읽힐 수 있어 'no_signal·signal'로 이름만 바꿨다. 판정 기준은 같다)"""
        base = {"temperature_c": 12.0, "apparent_temperature_c": 10.0, "wind_speed_mps": 3.0,
                "precipitation_mm": 0.0, "precipitation_probability_pct": 10, "missing_fields": []}
        self.assertEqual(sv.forecast_status({"status": "ok", "weather": base})["status"], "no_signal")
        rain = dict(base, precipitation_mm=2.0, precipitation_probability_pct=80)
        self.assertEqual(sv.forecast_status({"status": "ok", "weather": rain})["status"], "signal")
        missing = dict(base, precipitation_mm=None, missing_fields=["precipitation_mm"])
        for reply in ({"status": "network_error", "weather": None}, {"status": "partial", "weather": missing},
                      None, ["ok"], {"status": ["ok"], "weather": base}):
            with self.subTest(reply=str(reply)[:30]):
                self.assertEqual(sv.forecast_status(reply)["status"], "unknown")


# %% M-SV4. 대상 일정 식별 (요구 08, 보완 ②: LLM id만으로 확정하지 않음)
class M_SV4_Targets(unittest.TestCase):

    def setUp(self):
        self.doc = seed_doc()

    def find(self, **kwargs) -> dict[str, Any]:
        kwargs.setdefault("operation", "update")
        return sv.find_targets(self.doc, now=FIXED_NOW, **kwargs)

    def titles(self, result) -> list[str]:
        return [s["title"] for s in result["matches"]]

    def test_SV14_names_and_aliases(self):
        """SV14 '시장 가는 거', '장보는 거', '병원 약속'처럼 이름을 정확히 말하지 않아도 별칭으로 찾는다."""
        for text, title in (("시장 가는 거", "시장 장보기"), ("장보는 거", "시장 장보기"), ("병원 약속", "병원 진료"),
                            ("복지관 가는 일정", "복지관 방문"), ("산책", "아침 산책")):
            with self.subTest(text):
                result = self.find(target_text=text)
                self.assertEqual((result["status"], self.titles(result)), ("single", [title]))

    def test_SV15_multiple_none_unspecified(self):
        """SV15 '내일 오후 일정'은 2개 → 되묻기, 없는 일정은 none, 아무것도 말하지 않으면 unspecified"""
        afternoon = self.find(day="2026-11-26", period="오후")
        self.assertEqual((afternoon["status"], self.titles(afternoon)), ("multiple", ["시장 장보기", "복지관 방문"]))
        self.assertEqual(self.find(target_text="등산 가는 거")["status"], "none")
        self.assertEqual(self.find(target_text="시장", day="2026-11-27")["status"], "none")
        self.assertEqual(self.find()["status"], "unspecified")
        clock = self.find(day="2026-11-26", clock="14:30")
        self.assertEqual(self.titles(clock), ["시장 장보기"])

    def test_SV16_llm_ids_need_user_evidence(self):
        """SV16 LLM이 고른 id만 있고 발화 근거가 없으면 확정하지 않고 확인(confirm).
        (검토 반영으로 기대값 수정) 발화로 후보가 여러 개 남으면 LLM이 하나를 골라도 확정(single)하지 않는다.
        이전에는 LLM 선택으로 좁혀 single을 기대했지만, '내일 오후 일정 바꿔 줘'처럼 사용자가 하나로 정하지 않은 경우
        자동 확정은 잘못이므로 multiple(후보 목록으로 묻기) + llm_suggested(추천 표시만)로 바로잡았다."""
        market = item_of(self.doc, "시장 장보기")["schedule_id"]
        welfare = item_of(self.doc, "복지관 방문")["schedule_id"]
        self.assertEqual(self.find(target_text="그거", llm_ids=[market])["status"], "confirm")
        narrowed = self.find(day="2026-11-26", period="오후", llm_ids=[welfare])
        self.assertEqual((narrowed["status"], self.titles(narrowed)), ("multiple", ["시장 장보기", "복지관 방문"]))
        self.assertEqual(narrowed["llm_suggested"], [welfare])
        # 사용자가 '시장'이라고 했는데 LLM이 복지관을 골라도 발화 근거(시장)가 이긴다.
        self.assertEqual(self.titles(self.find(target_text="시장", llm_ids=[welfare])), ["시장 장보기"])
        # 존재하지 않는 id·잘못된 자료형은 무시한다.
        junk_ids: list[Any] = ["schedule_999", ["x"], None]
        self.assertEqual(self.find(llm_ids=junk_ids)["status"], "unspecified")

    def test_SV17_status_filter_and_exclusion(self):
        """SV17 변경 대상은 예정 일정만(완료·취소 제외), 조회는 모두. '다른 일정'이면 직전 대상을 뺀다."""
        done = with_item(self.doc, "시장 다녀오기", at(25, 9), status="completed")
        self.assertNotIn(done, self.find(target_text="시장")["matches"])
        self.assertIn(done, self.find(target_text="시장", operation="query")["matches"])
        market = item_of(self.doc, "시장 장보기")["schedule_id"]
        other = self.find(day="2026-11-26", exclude_ids=[market])
        self.assertEqual(self.titles(other), ["복지관 방문"])

    def test_SV18_choose_from_candidates(self):
        """SV18 번호를 붙여 보여 준 후보에서 '두 번째', '2번', '모레 거', '시장'으로 고른다. 모르면 None"""
        candidates = [item_of(self.doc, "시장 장보기"), item_of(self.doc, "복지관 방문"), item_of(self.doc, "병원 진료")]
        self.assertEqual(sv.choose_from_candidates(candidates, "두 번째", FIXED_NOW), candidates[1])
        self.assertEqual(sv.choose_from_candidates(candidates, "1번이요", FIXED_NOW), candidates[0])
        self.assertEqual(sv.choose_from_candidates(candidates, "글피 거", FIXED_NOW), candidates[2])
        self.assertEqual(sv.choose_from_candidates(candidates, "시장 가는 거", FIXED_NOW), candidates[0])
        self.assertIsNone(sv.choose_from_candidates(candidates, "내일 거", FIXED_NOW))
        self.assertIsNone(sv.choose_from_candidates(candidates, "네 번째", FIXED_NOW))


# %% M-SV5. 제안 생성과 완료 조건 (요구 10·19·38·39·40, 보완 ③)
class M_SV5_Proposals(unittest.TestCase):

    def setUp(self):
        self.doc = seed_doc()
        self.market = item_of(self.doc, "시장 장보기")

    def test_SV19_update_proposal_envelope(self):
        """SV19 변경 제안: 소요 시간 유지, 대상 id·버전·변경 전 내용 포함, 공통 봉투 키, approval 없음"""
        result = sv.propose_update(self.doc, self.market, at(26, 17), FIXED_NOW, reason="사용자 사정: 늦잠")
        self.assertTrue(result["ok"])
        action = result["action"]
        self.assertEqual(set(action), {"type", "action_id", "summary", "requires_approval", "applied", "created_at",
                                       "source", "payload"})
        self.assertRegex(action["action_id"], r"^[0-9a-f]{32}$")
        self.assertEqual((action["type"], action["source"]), ("update_schedule", "schedule_service"))
        payload = action["payload"]
        self.assertEqual((payload["schedule_id"], payload["expected_version"]), (self.market["schedule_id"], 1))
        self.assertEqual(payload["before"]["start_at"], "2026-11-26T14:00:00+09:00")
        self.assertEqual((payload["after"]["start_at"], payload["after"]["end_at"]),
                         ("2026-11-26T17:00:00+09:00", "2026-11-26T18:00:00+09:00"))
        self.assertIn("오후 2시 → 내일(11월 26일 목요일) 오후 5시", action["summary"])

    def test_SV20_update_refusals(self):
        """SV20 변경 거절: 변경 제한(보호 일정을 대신 옮기지 않음)·같은 시각·지난 시각·충돌(대안 포함)·완료된 일정"""
        welfare = item_of(self.doc, "복지관 방문")
        self.assertEqual(sv.propose_update(self.doc, welfare, at(26, 18), FIXED_NOW)["code"], "protected")
        self.assertEqual(sv.propose_update(self.doc, self.market, at(26, 14), FIXED_NOW)["code"], "same_time")
        self.assertEqual(sv.propose_update(self.doc, self.market, at(25, 9), FIXED_NOW)["code"], "past_time")
        conflict = sv.propose_update(self.doc, self.market, at(26, 16), FIXED_NOW)
        self.assertEqual((conflict["code"], [c["title"] for c in conflict["conflicts"]]), ("conflict", ["복지관 방문"]))
        self.assertEqual(conflict["alternatives"][0], at(26, 17))
        self.assertEqual(welfare["start_at"], "2026-11-26T16:00:00+09:00")
        done = with_item(self.doc, "끝난 일", at(25, 8), status="completed")
        self.assertEqual(sv.propose_update(self.doc, done, at(26, 9), FIXED_NOW)["code"], "not_scheduled")

    def test_SV21_user_requested_late_time_is_allowed(self):
        """SV21 사용자가 직접 말한 밤 10시는 자동 추천 범위(07~21시) 밖이어도 막지 않고, 늦은 시각 안내만 붙인다."""
        result = sv.propose_update(self.doc, self.market, at(26, 22), FIXED_NOW, approximate=True)
        self.assertTrue(result["ok"])
        notes = result["action"]["payload"]["notes"]
        self.assertTrue(any("늦은 시각" in n for n in notes))
        self.assertTrue(any("대략" in n for n in notes))

    def test_SV22_create_cancel(self):
        """SV22 등록(기본 60분 표시, 충돌 시 대안, 이름 필요)·취소(보호·지난 일정 거절, 상태만 cancelled)"""
        created = sv.propose_create(self.doc, title="산책", start=at(26, 8), minutes=None, now=FIXED_NOW,
                                    category="outing")
        self.assertTrue(created["ok"])
        self.assertTrue(any("60분" in n for n in created["action"]["payload"]["notes"]))
        self.assertEqual(sv.propose_create(self.doc, title="", start=at(26, 8), minutes=30, now=FIXED_NOW)["code"],
                         "title_needed")
        clash = sv.propose_create(self.doc, title="친구", start=at(26, 14, 30), minutes=30, now=FIXED_NOW)
        self.assertEqual(clash["code"], "conflict")
        self.assertEqual(sv.propose_create(self.doc, title="친구", start=at(25, 9), minutes=30, now=FIXED_NOW)["code"],
                         "past_time")
        cancel = sv.propose_cancel(self.market, FIXED_NOW)
        self.assertEqual(cancel["action"]["payload"]["after"]["status"], "cancelled")
        self.assertEqual(sv.propose_cancel(item_of(self.doc, "병원 진료"), FIXED_NOW)["code"], "protected")
        self.assertEqual(sv.propose_cancel(item_of(self.doc, "아침 산책"), FIXED_NOW)["code"], "past_time")

    def test_SV23_completion_rules_by_state(self):
        """SV23 완료 기록: 완료됨·취소됨·내일 이후는 거절, 진행 중·시간 지남은 제안, 오늘 시작 전은 안내를 붙여 제안"""
        walk = item_of(self.doc, "아침 산책")
        self.assertTrue(sv.propose_complete(walk, FIXED_NOW)["ok"])
        self.assertTrue(sv.propose_complete(walk, at(25, 8, 10))["ok"])
        later = with_item(self.doc, "오후 체조", at(25, 15))
        early = sv.propose_complete(later, FIXED_NOW)
        self.assertTrue(early["ok"])
        self.assertIn("예정 시각(오후 3시) 전", early["action"]["payload"]["notes"][0])
        self.assertEqual(sv.propose_complete(self.market, FIXED_NOW)["code"], "future_day")
        done = with_item(self.doc, "완료한 일", at(25, 7), status="completed")
        gone = with_item(self.doc, "취소한 일", at(25, 7), status="cancelled")
        self.assertEqual(sv.propose_complete(done, FIXED_NOW)["code"], "already_completed")
        self.assertEqual(sv.propose_complete(gone, FIXED_NOW)["code"], "cancelled")
        self.assertEqual(sv.propose_complete(item_of(self.doc, "병원 진료"), FIXED_NOW)["code"], "future_day")


# %% M-SV6. 승인 후 실행: 저장 직전 재검사와 영구 저장 (요구 12·20·21·22·23·25·34)
class M_SV6_Execution(StoreCase):

    def setUp(self):
        super().setUp()
        self.st = self.ready()
        self.doc = self.st.load()
        self.market = item_of(self.doc, "시장 장보기")

    def update_to(self, start: datetime) -> dict[str, Any]:
        return sv.propose_update(self.doc, self.market, start, FIXED_NOW)["action"]

    def test_SV24_approved_update_changes_only_target(self):
        """SV24 승인한 변경만 저장: 대상 일정만 바뀌고 버전+1, 이력 1건, 다른 일정은 그대로"""
        before = self.st.load()
        result = sv.apply_schedule_action(approved(self.update_to(at(26, 17))), self.st)
        self.assertEqual((result["status"], result["applied"], result["mock"]), ("ok", True, True))
        self.assertIn("오후 5시로 바꿔 저장했어요", result["answer"])
        after = self.st.load()
        changed = item_of(after, "시장 장보기")
        self.assertEqual((changed["start_at"], changed["end_at"], changed["version"]),
                         ("2026-11-26T17:00:00+09:00", "2026-11-26T18:00:00+09:00", 2))
        others = [s for s in after["schedules"] if s["title"] != "시장 장보기"]
        self.assertEqual(others, [s for s in before["schedules"] if s["title"] != "시장 장보기"])
        record = after["action_history"][0]
        self.assertEqual((record["operation"], record["before"]["start_at"], record["after"]["start_at"]),
                         ("update", "2026-11-26T14:00:00+09:00", "2026-11-26T17:00:00+09:00"))
        # 다시 실행한 프로그램에서 읽어도 바뀐 값이 남아 있다.
        self.assertEqual(item_of(ss.ScheduleStore(self.path).load(), "시장 장보기")["start_at"], "2026-11-26T17:00:00+09:00")

    def test_SV25_without_approval_nothing_changes(self):
        """SV25 approval이 없거나 거절 상태면 저장하지 않는다."""
        before = self.raw()
        action = self.update_to(at(26, 17))
        for bad in (action, dict(action, approval={"status": "rejected", "at": ss.iso(FIXED_NOW)}),
                    dict(action, approval={"status": "approved"}), dict(action, approval="approved")):
            with self.subTest(approval=str(bad.get("approval"))[:30]):
                result = sv.apply_schedule_action(bad, self.st)
                self.assertEqual((result["status"], result["applied"]), ("rejected", False))
        self.assertEqual(self.raw(), before)

    def test_SV26_duplicate_approval_applies_once(self):
        """SV26 같은 승인을 두 번(재실행 포함) 보내도 한 번만 적용한다."""
        action = approved(self.update_to(at(26, 17)))
        self.assertEqual(sv.apply_schedule_action(action, self.st)["status"], "ok")
        snapshot_after = self.raw()
        again = sv.apply_schedule_action(action, ss.ScheduleStore(self.path, clock=lambda: self.now))
        self.assertEqual((again["status"], again["applied"]), ("already_executed", True))
        self.assertEqual(self.raw(), snapshot_after)

    def test_SV27_stale_and_new_conflict_block_saving(self):
        """SV27 승인 대기 중 대상이 바뀌거나(stale), 그 시간에 다른 일정이 생기면(conflict) 저장하지 않는다."""
        waiting = approved(self.update_to(at(26, 17)))
        other = approved(self.update_to(at(26, 18)))
        self.assertEqual(sv.apply_schedule_action(other, self.st)["status"], "ok")
        before = self.raw()
        stale = sv.apply_schedule_action(waiting, self.st)
        self.assertEqual((stale["status"], stale["applied"]), ("stale", False))
        self.assertIn("다시 확인", stale["answer"])
        self.assertEqual(self.raw(), before)

        # 다른 일정을 17시에 먼저 등록하면, 대기 중이던 '시장 17시' 변경은 충돌로 막힌다.
        doc = self.st.load()
        market = item_of(doc, "시장 장보기")
        move = approved(sv.propose_update(doc, market, at(26, 19), FIXED_NOW)["action"])
        friend = approved(sv.propose_create(doc, title="친구 방문", start=at(26, 19, 30), minutes=60, now=FIXED_NOW)["action"])
        self.assertEqual(sv.apply_schedule_action(friend, self.st)["status"], "ok")
        before = self.raw()
        blocked = sv.apply_schedule_action(move, self.st)
        self.assertEqual((blocked["status"], blocked["applied"]), ("conflict", False))
        self.assertEqual(self.raw(), before)

    def test_SV28_protected_missing_and_invalid_envelopes(self):
        """SV28 보호 일정 변경·없는 일정·형식이 틀린 봉투(잘못된 자료형 포함)는 예외 없이 거절하고 저장하지 않는다."""
        before = self.raw()
        action = self.update_to(at(26, 17))
        welfare = item_of(self.doc, "복지관 방문")
        protected = approved(dict(action, payload=dict(action["payload"], schedule_id=welfare["schedule_id"],
                                                       before=sv.snapshot(welfare))))
        self.assertEqual(sv.apply_schedule_action(protected, self.st)["status"], "protected")
        missing = approved(dict(action, payload=dict(action["payload"], schedule_id="schedule_099")))
        self.assertEqual(sv.apply_schedule_action(missing, self.st)["status"], "not_found")
        broken = [None, [], {"type": ["update_schedule"]}, dict(action, source="living_support"),
                  dict(action, type="request_caregiver_help"), dict(action, action_id=["x"]),
                  dict(action, payload=dict(action["payload"], operation="cancel")),
                  dict(action, payload=dict(action["payload"], expected_version="1")),
                  dict(action, applied=True), dict(action, payload=[])]
        for bad in broken:
            with self.subTest(bad=str(bad)[:40]):
                result = sv.apply_schedule_action(approved(bad) if isinstance(bad, dict) else bad, self.st)
                self.assertEqual((result["status"], result["applied"]), ("invalid_action", False))
        self.assertEqual(self.raw(), before)

    def test_SV29_create_cancel_complete_flow(self):
        """SV29 등록·취소·완료가 승인 후에만 저장되고, 취소·완료는 기록을 지우지 않고 상태만 바꾼다."""
        create = approved(sv.propose_create(self.doc, title="친구 만나기", start=at(27, 10), minutes=90,
                                            now=FIXED_NOW, category="appointment")["action"])
        made = sv.apply_schedule_action(create, self.st)
        self.assertEqual(made["status"], "ok")
        doc = self.st.load()
        friend = item_of(doc, "친구 만나기")
        self.assertEqual((friend["schedule_id"], friend["end_at"], friend["source"]),
                         ("schedule_005", "2026-11-27T11:30:00+09:00", "user"))
        cancel = approved(sv.propose_cancel(item_of(doc, "시장 장보기"), FIXED_NOW)["action"])
        self.assertEqual(sv.apply_schedule_action(cancel, self.st)["status"], "ok")
        complete = approved(sv.propose_complete(item_of(doc, "아침 산책"), FIXED_NOW)["action"])
        self.assertEqual(sv.apply_schedule_action(complete, self.st)["status"], "ok")
        doc = self.st.load()
        self.assertEqual(len(doc["schedules"]), 5)
        self.assertEqual((item_of(doc, "시장 장보기")["status"], item_of(doc, "아침 산책")["status"]),
                         ("cancelled", "completed"))
        self.assertEqual([h["operation"] for h in doc["action_history"]], ["create", "cancel", "complete"])
        self.assertEqual(sv.history_entries(doc, day=FIXED_NOW.date())[0]["operation"], "complete")
        # '아까 취소한 일정'은 실제 이력에서 찾는다.
        recent = sv.find_targets(doc, now=FIXED_NOW, operation="query", target_text="아까 취소한 일정")
        self.assertEqual([s["title"] for s in recent["matches"]], ["시장 장보기"])

    def test_SV30_store_error_is_reported_not_success(self):
        """SV30 일정 파일이 손상되면 error로 알리고 성공으로 표시하지 않는다. (파일을 초기화하지 않음)"""
        action = approved(self.update_to(at(26, 17)))
        self.path.write_text("{", encoding="utf-8")
        result = sv.apply_schedule_action(action, self.st)
        self.assertEqual((result["status"], result["applied"]), ("error", False))
        self.assertIn("저장하지 않았어요", result["answer"])
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{")


# %% M-SV7. 기존 3단계 외출 시간 변경 제안의 안전한 연결 (보완 ①)
class M_SV7_OutingLink(StoreCase):

    def a_result(self, original: str, proposed: str, *, city: str | None = "인천", purpose: str = "") -> dict[str, Any]:
        return {"proposed_action": {"type": "propose_outing_time_change", "original_outing_at": original,
                                    "proposed_outing_at": proposed, "reason": "두 예보의 차이입니다. 강수확률 80 → 10%",
                                    "requires_approval": True, "applied": False},
                "outing_request": {"city": city, "purpose": purpose}}

    def test_SV31_single_confirmed_target_becomes_update(self):
        """SV31 날짜·시각 구간·지역·목적이 맞는 일정이 정확히 1개면 변경 제안(대상 id·버전·변경 전 내용)으로 바꾼다."""
        doc = seed_doc()
        linked = sv.link_outing_proposal(doc, self.a_result("2026-11-26T14:00+09:00", "2026-11-26T17:00+09:00",
                                                            purpose="장보기"), FIXED_NOW)
        self.assertEqual(linked["kind"], "update")
        payload = linked["action"]["payload"]
        self.assertEqual((payload["schedule_id"], payload["expected_version"], payload["before"]["title"]),
                         ("schedule_002", 1, "시장 장보기"))
        self.assertIn("강수확률", payload["reason"])

    def test_SV32_time_only_is_not_enough(self):
        """SV32 시작 시각만 같고 목적이 다르거나 근거가 없으면 확정하지 않고 되묻는다."""
        doc = seed_doc()
        no_purpose = sv.link_outing_proposal(doc, self.a_result("2026-11-26T14:00+09:00", "2026-11-26T17:00+09:00",
                                                                city=None), FIXED_NOW)
        self.assertEqual((no_purpose["kind"], no_purpose["action"]), ("ambiguous", None))
        other = sv.link_outing_proposal(doc, self.a_result("2026-11-26T14:30+09:00", "2026-11-26T17:00+09:00",
                                                           purpose="친구 만나기"), FIXED_NOW)
        self.assertEqual(other["kind"], "ambiguous")
        with_item(doc, "시장 들르기", at(26, 14, 30), minutes=30)
        two = sv.link_outing_proposal(doc, self.a_result("2026-11-26T14:30+09:00", "2026-11-26T17:00+09:00",
                                                         purpose="시장"), FIXED_NOW)
        self.assertEqual((two["kind"], len(two["candidates"])), ("ambiguous", 2))

    def test_SV33_unsaved_outing_becomes_create_not_update(self):
        """SV33 저장된 일정에 없는 외출이면 기존 일정을 고치지 않고 신규 등록을 제안한다."""
        doc = seed_doc()
        linked = sv.link_outing_proposal(doc, self.a_result("2026-11-27T14:00+09:00", "2026-11-27T16:00+09:00",
                                                            purpose="병원"), FIXED_NOW)
        self.assertEqual(linked["kind"], "create")
        self.assertEqual((linked["action"]["type"], linked["action"]["payload"]["after"]["title"]),
                         ("create_schedule", "병원"))
        self.assertIsNone(linked["action"]["payload"]["schedule_id"])

    def test_SV34_link_refusals_and_bad_input(self):
        """SV34 보호 일정·충돌이면 제안을 만들지 않고, 형식이 틀린 결과는 invalid로 처리한다."""
        doc = seed_doc()
        protected = sv.link_outing_proposal(doc, self.a_result("2026-11-26T16:00+09:00", "2026-11-26T18:00+09:00",
                                                               purpose="복지관"), FIXED_NOW)
        self.assertEqual((protected["kind"], protected["refusal"]["code"]), ("refused", "protected"))
        clash = sv.link_outing_proposal(doc, self.a_result("2026-11-26T14:00+09:00", "2026-11-26T16:30+09:00",
                                                           purpose="시장"), FIXED_NOW)
        self.assertEqual((clash["kind"], clash["refusal"]["code"]), ("refused", "conflict"))
        for bad in ({}, {"proposed_action": []}, {"proposed_action": {"original_outing_at": "x"}}, [], None):
            with self.subTest(bad=str(bad)[:30]):
                junk: Any = bad
                self.assertEqual(sv.link_outing_proposal(doc, junk, FIXED_NOW)["kind"], "invalid")

    def test_SV35_apply_outing_change_requires_confirmed_target(self):
        """SV35 apply_outing_change: 대상 확인이 없는 옛 3단계 봉투는 저장하지 않고, 확인된 봉투만 일정 변경으로 저장한다."""
        store = self.ready()
        before = self.raw()
        legacy = approved({"type": "propose_outing_time_change", "action_id": uuid.uuid4().hex, "summary": "s",
                           "requires_approval": True, "applied": False, "created_at": ss.iso(FIXED_NOW),
                           "source": "outing_decision",
                           "payload": {"original_outing_at": "2026-11-26T14:00+09:00",
                                       "proposed_outing_at": "2026-11-26T17:00+09:00", "reason": "r"}})
        self.assertEqual(sv.apply_outing_change(legacy, store)["status"], "invalid_action")
        self.assertEqual(self.raw(), before)
        linked = sv.link_outing_proposal(store.load(), self.a_result("2026-11-26T14:00+09:00", "2026-11-26T17:00+09:00",
                                                                     purpose="시장"), FIXED_NOW)["action"]
        confirmed = approved(dict(linked, type="propose_outing_time_change", source="outing_decision"))
        self.assertEqual(sv.apply_outing_change(confirmed, store)["status"], "ok")


# %% M-SV8. 생활 기록과 B기능 기록 어댑터 (요구 30, 보완 ⑧: 가짜 B 기록을 만들지 않음)
class M_SV8_LifeRecord(StoreCase):

    def test_SV36_life_record_without_b(self):
        """SV36 B가 연결되지 않으면 보호자 요청 기록은 '연결 안 됨'으로 두고 빈 목록을 지어내지 않는다."""
        record = sv.life_record(seed_doc(), FIXED_NOW)
        self.assertEqual((record["b_status"], record["b_requests"]), ("not_connected", []))
        self.assertEqual([s["title"] for s in record["today"]], ["아침 산책"])
        self.assertEqual([s["title"] for s in record["tomorrow"]], ["시장 장보기", "복지관 방문"])

    def test_SV37_living_log_adapter_reads_only(self):
        """SV37 명세 2.3 형식의 JSON Lines만 읽고, 틀린 줄은 건너뛰며, 파일이 없으면 만들지 않는다."""
        log = Path(self.temp.name) / "mock_action_log.jsonl"
        self.assertEqual(sv.read_living_action_log(log)["status"], "missing")
        self.assertFalse(log.exists())
        good = {"action_id": "a" * 32, "type": "request_caregiver_help", "mock": True,
                "payload": {"message": "우유 1개 부탁드립니다.", "items": [{"name": "우유", "quantity": 1, "unit": "개"}]},
                "approved_at": ss.iso(FIXED_NOW), "executed_at": ss.iso(FIXED_NOW)}
        lines = [json.dumps(good, ensure_ascii=False), "{깨진 줄", json.dumps(dict(good, mock=False)),
                 json.dumps(dict(good, payload=[])), ""]
        log.write_text("\n".join(lines), encoding="utf-8")
        before = log.read_bytes()
        result = sv.read_living_action_log(log)
        self.assertEqual((result["status"], len(result["records"]), result["skipped"]), ("ok", 1, 3))
        self.assertEqual(log.read_bytes(), before)
        record = sv.life_record(seed_doc(), FIXED_NOW, result["records"])
        self.assertEqual((record["b_status"], record["b_requests"][0]["message"]), ("ok", "우유 1개 부탁드립니다."))


# %% M-SV9. 구현 단위 2 외부 코드 검토에서 재현된 문제의 회귀 테스트
class M_SV9_ReviewRegressions(StoreCase):

    def setUp(self):
        super().setUp()
        self.doc = seed_doc()

    def test_SV38_llm_choice_does_not_confirm_among_many(self):
        """SV38 (검토 1) '내일 오후 일정 바꿔 줘': 시장·복지관 2개가 남으면 LLM이 하나를 골라도 확정하지 않는다."""
        for title in ("시장 장보기", "복지관 방문"):
            with self.subTest(title):
                chosen = item_of(self.doc, title)["schedule_id"]
                result = sv.find_targets(self.doc, now=FIXED_NOW, operation="update", target_text="내일 오후 일정",
                                         day="2026-11-26", period="오후", llm_ids=[chosen])
                self.assertEqual((result["status"], len(result["matches"]), result["llm_suggested"]),
                                 ("multiple", 2, [chosen]))

    def test_SV39_correction_in_candidate_choice(self):
        """SV39 (검토 2) '첫 번째 말고 두 번째'는 두 번째, 부정만 있거나 둘 이상을 말하면 확정하지 않고 다시 묻는다."""
        candidates = [item_of(self.doc, "시장 장보기"), item_of(self.doc, "복지관 방문")]
        chosen = {"첫 번째 말고 두 번째": 1, "아니 두 번째요": 1, "1번 말고 2번": 1, "시장 말고 복지관": 1,
                  "두 번째 말고 첫 번째": 0}
        for text, index in chosen.items():
            with self.subTest(text):
                self.assertEqual(sv.choose_from_candidates(candidates, text, FIXED_NOW), candidates[index])
        for text in ("첫 번째 말고", "첫 번째 아니야", "시장 말고", "첫 번째랑 두 번째", "1번 2번 둘 다", "둘 다 싫어"):
            with self.subTest(text):
                self.assertIsNone(sv.choose_from_candidates(candidates, text, FIXED_NOW))

    def test_SV40_incomplete_or_invalid_forecast_is_unknown(self):
        """SV40 (검토 3) 빈 예보·항목 누락·자료형·결측값·범위·missing_fields 불일치는 모두 unknown. 신호 없음은 안전 보장이 아님"""
        base = {"temperature_c": 12.0, "apparent_temperature_c": 10.0, "wind_speed_mps": 3.0,
                "precipitation_mm": 0.0, "precipitation_probability_pct": 10, "missing_fields": []}
        bad_weathers = {
            "empty": {}, "missing_key": {k: v for k, v in base.items() if k != "wind_speed_mps"},
            "string": dict(base, temperature_c="12"), "bool": dict(base, precipitation_mm=False),
            "nan": dict(base, wind_speed_mps=float("nan")), "inf": dict(base, temperature_c=float("inf")),
            "negative_rain": dict(base, precipitation_mm=-1.0), "probability_150": dict(base, precipitation_probability_pct=150),
            "temperature_100": dict(base, temperature_c=100.0),
            "none_not_listed": dict(base, precipitation_mm=None),
            "listed_not_none": dict(base, missing_fields=["precipitation_mm"]),
            "missing_fields_not_list": dict(base, missing_fields="none"), "no_missing_fields_key": {k: v for k, v in base.items()
                                                                                                    if k != "missing_fields"},
            "partial_value": dict(base, precipitation_mm=None, missing_fields=["precipitation_mm"]),
        }
        for name, weather in bad_weathers.items():
            with self.subTest(name):
                result = sv.forecast_status({"status": "ok", "weather": weather})
                self.assertEqual(result["status"], "unknown")
                self.assertTrue(result["reason"])
        self.assertEqual(sv.forecast_status({"status": "partial", "weather": base})["status"], "unknown")
        ok = sv.forecast_status({"status": "ok", "weather": base})
        self.assertEqual(ok["status"], "no_signal")
        self.assertIn("안전 보장 아님", ok["reason"])

    def test_SV41_midnight_crossing_clock(self):
        """SV41 (검토 4) 23:00~다음 날 01:00 일정은 다음 날 00:30을 포함한다. 날짜를 알면 그 날짜로만 비교한다."""
        night = with_item(self.doc, "야간 기도회", at(26, 23), minutes=120)
        self.assertTrue(sv.covers_clock(night, "00:30"))
        self.assertTrue(sv.covers_clock(night, "00:30", "2026-11-27"))
        self.assertFalse(sv.covers_clock(night, "00:30", "2026-11-26"))
        self.assertTrue(sv.covers_clock(night, "23:30", "2026-11-26"))
        found = sv.find_targets(self.doc, now=FIXED_NOW, operation="update", day="2026-11-27", clock="00:30")
        self.assertEqual((found["status"], found["matches"]), ("single", [night]))
        self.assertEqual(sv.find_targets(self.doc, now=FIXED_NOW, operation="update", day="2026-11-26",
                                         clock="00:30")["status"], "none")
        # 기존의 정상적인 시각 식별은 그대로다.
        market = sv.find_targets(self.doc, now=FIXED_NOW, operation="update", day="2026-11-26", clock="14:30")
        self.assertEqual([s["title"] for s in market["matches"]], ["시장 장보기"])

    def test_SV42_cancel_blocked_when_schedule_ended_while_waiting(self):
        """SV42 (검토 5) 취소 제안 뒤 승인을 기다리는 사이 일정이 끝나면 저장 직전에 차단한다."""
        store = self.ready()
        cancel = approved(sv.propose_cancel(item_of(store.load(), "시장 장보기"), FIXED_NOW)["action"])
        self.now = at(26, 15, 30)
        before = self.raw()
        result = sv.apply_schedule_action(cancel, store)
        self.assertEqual((result["status"], result["applied"]), ("invalid_action", False))
        self.assertIn("이미 끝나", result["answer"])
        self.assertEqual(self.raw(), before)

    def test_SV43_time_is_read_after_lock_is_acquired(self):
        """SV43 (검토 5) 현재 시각은 저장소 잠금을 잡은 뒤에 읽는다. 잠금을 기다리는 동안 일정이 끝나면 취소하지 않는다."""
        store = self.ready()
        cancel = approved(sv.propose_cancel(item_of(store.load(), "시장 장보기"), FIXED_NOW)["action"])
        before = self.raw()
        results: list[dict[str, Any]] = []
        holder = ss.FileLock(store.lock_path, 1)
        holder.__enter__()
        try:
            worker = threading.Thread(target=lambda: results.append(sv.apply_schedule_action(cancel, store)))
            worker.start()
            worker.join(timeout=0.5)
            self.assertTrue(worker.is_alive())           # 잠금을 기다리는 중
            self.now = at(26, 15, 30)                     # 기다리는 사이 일정 종료 시각이 지남
        finally:
            holder.__exit__(None, None, None)
        worker.join(timeout=10)
        self.assertEqual((results[0]["status"], results[0]["applied"]), ("invalid_action", False))
        self.assertEqual(self.raw(), before)

    def test_SV44_tampered_approved_content_is_not_saved(self):
        """SV44 (검토 6) 소요 시간이 바뀐 변경, 바꿀 수 없는 항목 변경, 범위 밖 등록, 승인 내용과 다른 결과는 저장하지 않는다."""
        store = self.ready()
        doc = store.load()
        market = item_of(doc, "시장 장보기")
        update = sv.propose_update(doc, market, at(26, 17), FIXED_NOW)["action"]
        create = sv.propose_create(doc, title="친구 만나기", start=at(27, 10), minutes=60, now=FIXED_NOW)["action"]
        cancel = sv.propose_cancel(market, FIXED_NOW)["action"]

        def changed(action, **after):
            payload = dict(action["payload"], after=dict(action["payload"]["after"], **after))
            return approved(dict(action, payload=payload))

        tampered = {
            "update_duration": changed(update, end_at=ss.iso(at(26, 18, 30))),
            "update_title": changed(update, title="다른 이름"),
            "update_protect_flag": changed(update, reschedulable=False),
            "update_status": changed(update, status="completed"),
            "cancel_moves_time": changed(cancel, start_at=ss.iso(at(26, 18))),
            "cancel_wrong_status": changed(cancel, status="completed"),
            "create_too_long": changed(create, end_at=ss.iso(at(27, 10) + timedelta(minutes=800))),
            "create_too_short": changed(create, end_at=ss.iso(at(27, 10, 5))),
            "create_completed": changed(create, status="completed"),
            "create_spaces": changed(create, title=" 친구 만나기 "),
            "create_bad_category": changed(create, category="party"),
            "create_extra_key": approved(dict(create, payload=dict(create["payload"],
                                                                   after=dict(create["payload"]["after"], note="x")))),
        }
        before = self.raw()
        for name, action in tampered.items():
            with self.subTest(name):
                result = sv.apply_schedule_action(action, store)
                self.assertEqual((result["status"], result["applied"]), ("invalid_action", False), result["answer"])
        self.assertEqual(self.raw(), before)
        # 손대지 않은 같은 제안은 정상 저장된다. (검사가 정상 경로를 막지 않음)
        self.assertEqual(sv.apply_schedule_action(approved(update), store)["status"], "ok")

    def test_SV45_huge_or_conflicting_expressions(self):
        """SV45 (검토 7) 매우 큰 상대 날짜는 예외 없이 다시 묻고, 서로 다른 날·시각을 함께 말하면 임의로 확정하지 않는다."""
        for text in ("999999999주 뒤", "99999999999일 뒤", "9999999일 후", "200주 뒤"):
            with self.subTest(text):
                self.assertIn("ambiguous", sv.parse_day(text, FIXED_NOW) or {})
        for text in ("999999999주 뒤로 미뤄", "99999999999일 뒤로 미뤄 줘"):
            with self.subTest(text):
                self.assertIn("ambiguous", sv.resolve_shift(text, at(26, 14)) or {})
        # ('내일 모레'는 '내일모레'와 같은 말로 3단계에서도 모레로 읽으므로 충돌 사례에서 뺐다)
        for text in ("오늘 내일", "오늘 모레", "내일 금요일"):
            with self.subTest(text):
                self.assertIn("ambiguous", sv.parse_day(text, FIXED_NOW) or {})
        self.assertEqual(got(sv.parse_day("내일 목요일", FIXED_NOW), "date"), "2026-11-26")
        self.assertEqual(got(sv.parse_day("내일모레", FIXED_NOW), "date"), "2026-11-27")
        self.assertEqual(got(sv.parse_day("100주 뒤", FIXED_NOW), "date"), "2028-10-25")
        self.assertIn("ambiguous", sv.parse_clock("오후 2시나 3시") or {})


# %% 3단계 의도 분류 보완 테스트 도우미 (구현 단위 3)
# 핵심 요약: 3단계 테스트의 Harness·FakeLLM·FakeWeather·FakeRAG로 Ollama·날씨·RAG를 가상으로 바꾸고, 실제 3단계 코드를 실행한다.
OLD_INTENTS = ("weather_question", "outing_consultation", "safety_guidance", "schedule_change", "daily_conversation",
               "living_support")
SCHEDULE_HINT = [{"title": "시장 장보기", "date": "2026-11-25", "time": "14:00"},
                 {"title": "복지관 방문", "date": "2026-11-25", "time": "16:00"}]


def run_with_schedule(h: Any, request: Any, state: Any = None, **kwargs) -> dict[str, Any]:
    """Harness 안에서 run_outing_decision_with_schedule을 실행하고 JSON 직렬화를 확인한다."""
    kwargs.setdefault("device", "xpu")
    kwargs.setdefault("index", tod.FAKE_INDEX)
    result = od.run_outing_decision_with_schedule(request, conversation_state=state, **kwargs)
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result


# %% M-OD1. 3단계 공개 계약 유지 (R1 해시 비교를 대신하는 계약 검사, 요구 32)
class M_OD1_Contract(unittest.TestCase):

    def test_OD01_public_contract_is_backward_compatible(self):
        """OD01 기존 run_outing_decision 시그니처·반환 키·decision 키·이해 스키마 키는 그대로이고, 의도는 뒤에만 덧붙였다."""
        params = inspect.signature(od.run_outing_decision).parameters
        self.assertEqual(list(params), ["request", "device", "index", "conversation_state"])
        new = inspect.signature(od.run_outing_decision_with_schedule).parameters
        self.assertEqual(list(new), ["request", "device", "index", "conversation_state", "schedule_hint", "schedule_handoff"])
        self.assertEqual(od.INTENTS[:6], OLD_INTENTS)
        self.assertEqual(od.INTENTS[6:], ("schedule_query", "schedule_create", "schedule_cancel", "schedule_complete",
                                         "schedule_history"))
        self.assertEqual(od.ROUTES, od.INTENTS + ("needs_clarification", "error"))
        self.assertEqual(set(od.UNDERSTANDING_SCHEMA["properties"]),
                         {"intent", "intent_clear", "answers_previous_question", "user_goal", "needs_weather",
                          "safety_topics", "city_text", "date_text", "time_text", "appointment_time_text",
                          "departure_lead_text", "alternative_time_text", "purpose", "can_reschedule",
                          "user_reported_conditions"})
        self.assertEqual(od.UNDERSTANDING_SCHEMA["properties"]["intent"]["enum"], list(od.INTENTS))
        self.assertEqual(len(od.empty_result()), 26)

    def test_OD02_prompt_defines_schedule_intents(self):
        """OD02 이해 프롬프트가 새 의도 정의와, 실측에서 틀렸던 문장·부정·생활 지원 구분 예시를 담는다."""
        prompt = od.UNDERSTANDING_PROMPT
        for name in od.SCHEDULE_INTENTS:
            self.assertIn(f"- {name}:", prompt)
        for text in ("비 오면 시간 바꿀까?\" → intent=schedule_change", "바꾸지 마", "네가 알아서 정해 줘",
                     "우유하고 계란이 떨어졌어\" → intent=living_support", "날씨만 묻는 말은 weather_question"):
            self.assertIn(text, prompt)


# %% M-OD2. 일정 넘김 경로 (요구 18, LLM은 의도만, 일정 변경·승인은 하지 않음)
class M_OD2_Handoff(unittest.TestCase):

    def test_OD03_new_schedule_intents_hand_off_without_tools(self):
        """OD03 새 일정 의도 5개: 이해 LLM 1회만, 날씨·RAG·답변 LLM 없음, route=의도, 승인 제안 없음, 외출 slots 오염 없음"""
        cases = {"schedule_query": "내가 내일 뭐 하기로 했지?", "schedule_create": "내일 아침 여덟 시에 산책할 거야. 기억해 줘.",
                 "schedule_cancel": "내일 장보기는 안 갈래", "schedule_complete": "오늘 시장 다녀왔어",
                 "schedule_history": "아까 바꾼 일정 몇 시였지?"}
        for intent, message in cases.items():
            for runner in ("old", "new"):
                with self.subTest(intent=intent, runner=runner), tod.Harness(llm=tod.FakeLLM(
                        {message: tod.u(intent=intent, date_text="내일" if "내일" in message else "")})) as h:
                    result = h.run(message) if runner == "old" else run_with_schedule(h, message, schedule_hint=SCHEDULE_HINT)
                    self.assertEqual((result["status"], result["route"], result["intent"]), ("ok", intent, intent))
                    self.assertEqual((result["usage"]["llm_calls"], h.weather.calls, h.rag.calls), (1, [], []))
                    self.assertEqual((result["requires_user_confirmation"], result["proposed_action"]), (False, None))
                    self.assertEqual(result["decision"]["kind"], "schedule_handoff")
                    self.assertEqual(result["decision"]["answer_source"], "handoff")
                    self.assertEqual(set(result["decision"]), {"kind", "suggest_schedule_change", "official_points",
                                                               "used_fact_ids", "question", "verified", "answer_source"})
                    self.assertEqual(result["answer"], od.HANDOFF_ANSWER)
                    state = result["conversation_state"]
                    self.assertEqual(state["slots"], od.new_conversation_state()["slots"])
                    self.assertEqual((state["intent"], state["turn"], len(state["history"])), (intent, 1, 2))
                    self.assertEqual(od.load_conversation_state(state)["intent"], intent)

    def test_OD04_schedule_change_hands_off_only_when_requested(self):
        """OD04 schedule_change: 기존 함수는 기존대로 예보 비교·제안(3단계 S08), 저장소 연결용 함수는 일정 단계로 넘김"""
        message = "내일 오후 2시에 인천 가는데 비 오면 시간 바꿀까?"
        understanding = tod.u(intent="schedule_change", needs_weather=True, city_text="인천", date_text="내일",
                              time_text="오후 2시")
        rainy = {14: {"precipitation_mm": 2.0, "precipitation_probability_pct": 80}}
        with tod.Harness(llm=tod.FakeLLM({message: understanding}), weather=tod.FakeWeather(by_hour=rainy)) as h:
            old = h.run(message)
        self.assertEqual((old["route"], old["requires_user_confirmation"]), ("schedule_change", True))
        self.assertEqual(old["proposed_action"]["type"], "propose_outing_time_change")
        with tod.Harness(llm=tod.FakeLLM({message: understanding}), weather=tod.FakeWeather(by_hour=rainy)) as h:
            kept = run_with_schedule(h, message, schedule_handoff=False)
        self.assertEqual((kept["route"], kept["requires_user_confirmation"]), ("schedule_change", True))
        with tod.Harness(llm=tod.FakeLLM({message: understanding}), weather=tod.FakeWeather(by_hour=rainy)) as h:
            handed = run_with_schedule(h, message)
            self.assertEqual((handed["route"], handed["decision"]["kind"], h.weather.calls), ("schedule_change", "schedule_handoff", []))
            self.assertEqual((handed["requires_user_confirmation"], handed["proposed_action"]), (False, None))

    def test_OD05_structured_schedule_intent_needs_no_llm(self):
        """OD05 문장 없이 intent만 넘긴 구조화 요청도 넘김 경로를 쓰고, LLM을 부르지 않는다."""
        with tod.Harness() as h:
            result = h.run({"intent": "schedule_query"})
        self.assertEqual((result["route"], result["usage"]["llm_calls"]), ("schedule_query", 0))

    def test_OD06_other_intents_keep_their_paths(self):
        """OD06 날씨 질문·안전수칙·생활 지원·일상 대화는 기존 경로 그대로다. (넘김 없음, 저장소 연결용 함수에서도 같음)"""
        cases = {"내일 인천 비 와?": (tod.u(**tod.WQ, city_text="인천", date_text="내일"), "weather_question"),
                 "눈길에서는 어떻게 걸어야 해?": (tod.u(intent="safety_guidance", safety_topics=["대설"]), "safety_guidance"),
                 "우유하고 계란이 떨어졌어": (tod.u(intent="living_support", user_goal="생필품 부족"), "living_support"),
                 "오늘 심심하다": (tod.u(intent="daily_conversation"), "daily_conversation")}
        for message, (understanding, route) in cases.items():
            with self.subTest(message), tod.Harness(llm=tod.FakeLLM({message: understanding})) as h:
                result = run_with_schedule(h, message, schedule_hint=SCHEDULE_HINT)
                self.assertEqual(result["route"], route)
                self.assertNotEqual(result["decision"]["kind"], "schedule_handoff")
                self.assertNotIn("schedule_handoff", result["llm"]["rules_applied"])


# %% M-OD3. 일정 요약 전달과 재확인 신호 (보완 ②: 동사만으로 의도를 확정하지 않음)
class M_OD3_HintAndRecheck(unittest.TestCase):

    def test_OD07_schedule_hint_reaches_understanding_only_when_given(self):
        """OD07 저장된 일정 요약은 제목·날짜·시각만, 최대 8개, 형식이 틀린 항목은 버리고 이해 단계 자료에 넣는다."""
        message = "시장 가는 거 몇 시였지?"
        hint = SCHEDULE_HINT + [{"title": "x" * 50, "date": "d", "time": "t"}, ["잘못된"], {"title": "id만"}] + \
            [{"title": f"일정{i}", "date": "2026-11-26", "time": "09:00", "schedule_id": "schedule_001"} for i in range(10)]
        with tod.Harness(llm=tod.FakeLLM({message: tod.u(intent="schedule_query")})) as h:
            run_with_schedule(h, message, schedule_hint=hint)
            context = h.llm.stage_calls("understanding")[0]["context"]
        listed = context["등록된_일정"]
        self.assertEqual(len(listed), 8)
        self.assertEqual(listed[0], {"제목": "시장 장보기", "날짜": "2026-11-25", "시각": "14:00"})
        self.assertTrue(all(set(item) == {"제목", "날짜", "시각"} for item in listed))
        with tod.Harness(llm=tod.FakeLLM({message: tod.u(intent="schedule_query")})) as h:
            h.run(message)
            self.assertNotIn("등록된_일정", h.llm.stage_calls("understanding")[0]["context"])

    def test_OD08_recheck_signal_never_changes_intent(self):
        """OD08 모델이 날씨 질문으로 분류한 일정 변경 모양 문장은 재확인 신호만 남기고, 의도·경로는 바꾸지 않는다."""
        cases = {"내일 시장 가는 시간 바꿔 줘, 비 와?": True, "내일 일정 바꾸지 마, 비 온대?": True,
                 "시장 장보기 미뤄야 하나? 비 오나?": True, "옷 바꿔 입어야 할까?": False, "내일 비 와?": False}
        for message, expected in cases.items():
            with self.subTest(message), tod.Harness(llm=tod.FakeLLM(
                    {message: tod.u(**tod.WQ, city_text="인천", date_text="내일")})) as h:
                result = run_with_schedule(h, message, schedule_hint=SCHEDULE_HINT)
                # 의도는 모델이 고른 그대로다. (지역이 발화에 없으면 기존 규칙대로 지역을 묻는 경로가 될 수 있다)
                self.assertEqual(result["intent"], "weather_question")
                self.assertNotEqual(result["decision"]["kind"], "schedule_handoff")
                self.assertEqual("schedule_recheck_suggested" in result["llm"]["rules_applied"], expected)
        message = "내일 시장 가는 시간 바꿔 줘, 비 와?"
        with tod.Harness(llm=tod.FakeLLM({message: tod.u(**tod.WQ, city_text="인천", date_text="내일")})) as h:
            old = h.run(message)
            off = run_with_schedule(h, message, schedule_handoff=False)
        for result in (old, off):
            self.assertNotIn("schedule_recheck_suggested", result["llm"]["rules_applied"])
        self.assertFalse(od.needs_schedule_recheck("schedule_change", "시간 바꿔 줘"))
        self.assertFalse(od.needs_schedule_recheck("weather_question", None))

    def test_OD09_multi_turn_intent_change_is_not_an_answer(self):
        """OD09 다중 턴: 지역을 묻는 중 '아니 날씨 말고 내일 일정 바꿔 줘' → 대답으로 묶지 않고 일정 변경으로 넘긴다."""
        first, second = tod.LATER_MEAL_Q, "아니 날씨 말고 내일 일정 바꿔 줘"
        llm = tod.FakeLLM({first: tod.LATER_MEAL_U,
                           second: tod.u(intent="schedule_change", answers_previous_question=True, date_text="내일")})
        with tod.Harness(llm=llm, now=tod.NOV25_1500) as h:
            asked = run_with_schedule(h, first, schedule_hint=SCHEDULE_HINT)
            self.assertEqual(asked["route"], "needs_clarification")
            moved = run_with_schedule(h, second, asked["conversation_state"], schedule_hint=SCHEDULE_HINT)
        self.assertEqual((moved["route"], moved["decision"]["kind"]), ("schedule_change", "schedule_handoff"))
        self.assertIn("not_an_answer_to_pending_question", moved["llm"]["rules_applied"])
        self.assertIsNone(moved["conversation_state"]["pending_question"])
        self.assertEqual(h.weather.calls, [])


# %% L-OD. 실제 qwen2.5:3b 의도 분류 측정 (--live-ollama 지정 시에만)
# 핵심 요약: 정해진 문장 20개를 정해진 횟수(3회)만큼 실행해 모델의 원래 의도와 최종 경로를 모두 기록한다. 결과를 고르지 않는다.
# 모델이 기대와 다르게 분류해도 이 테스트는 실패로 처리하지 않고 표로 보고한다. (모델 성능과 프로그램 결함을 구분하기 위함)
# 프로그램 계약(예외 없음, 경로 값, 승인 제안 없음)은 assert로 확인한다.
LIVE_INTENT_CASES = [
    ("내일 시장 가는 시간 좀 바꿔 줘", {"schedule_change"}),
    ("내일 오후 2시에 인천 가는데 비 오면 시간 바꿀까?", {"schedule_change"}),
    ("내일 일정 좀 바꿔 줘", {"schedule_change"}),
    ("시간은 네가 알아서 정해 줘", {"schedule_change"}),
    ("오후 네 시 말고 저녁 일곱 시로 해 줘", {"schedule_change"}),
    ("내일 일정 바꾸지 마", {"schedule_change"}),
    ("시간을 바꿀 필요는 없어", {"schedule_change"}),
    ("비가 오면 시간을 바꿀까?", {"schedule_change"}),
    ("아까 바꾼 일정 몇 시였지?", {"schedule_history", "schedule_query"}),
    ("원래 시장에 몇 시에 가려고 했지?", {"schedule_history", "schedule_query"}),
    ("내가 내일 뭐 하기로 했지?", {"schedule_query"}),
    ("다음 약속이 뭐야?", {"schedule_query"}),
    ("내일 아침 여덟 시에 산책할 거야. 기억해 줘.", {"schedule_create"}),
    ("내일 장보기는 안 갈래", {"schedule_cancel"}),
    ("오늘 시장 다녀왔어", {"schedule_complete"}),
    ("내일 인천 비 와?", {"weather_question"}),
    ("내일 시장 가는데 뭐 입고 가?", {"outing_consultation"}),
    ("폭염 때는 어떻게 해야 돼?", {"safety_guidance"}),
    ("우유하고 계란이 떨어졌어", {"living_support"}),
    ("오늘 심심하다", {"daily_conversation"}),
]
LIVE_REPEATS = 3


class L_OD_LiveIntent(unittest.TestCase):

    def setUp(self):
        if not OPTIONS.live_ollama:
            raise unittest.SkipTest("--live-ollama 를 지정하면 실행합니다. (ollama serve와 qwen2.5:3b 필요)")
        report = od.ollama_status(od.ollama_settings())
        if not (report["server_running"] and report["model_installed"]):
            raise unittest.SkipTest(f"Ollama 서버 또는 모델이 준비되지 않았습니다: {report}")

    def test_L_OD01_intent_classification_table(self):
        """L-OD01 실제 Qwen 의도 분류: 모델 원래 의도, 재확인 신호, 최종 경로를 모두 기록한다. (날씨·RAG는 넘김 경로에서 실행 안 함)"""
        rows = []
        for message, expected in LIVE_INTENT_CASES:
            for attempt in range(1, LIVE_REPEATS + 1):
                result = od.run_outing_decision_with_schedule(message, schedule_hint=SCHEDULE_HINT)
                json.dumps(result, ensure_ascii=False, allow_nan=False)
                self.assertIn(result["route"], od.ROUTES)
                if result["route"] in od.SCHEDULE_INTENTS + ("schedule_change",):
                    self.assertIsNone(result["proposed_action"])
                understanding = result["llm"].get("understanding") or {}
                raw = understanding.get("intent")
                rows.append({"message": message, "attempt": attempt, "expected": sorted(expected), "model_intent": raw,
                             "route": result["route"], "status": result["status"],
                             "recheck": "schedule_recheck_suggested" in result["llm"]["rules_applied"],
                             "model_ok": raw in expected, "route_ok": result["route"] in expected})
        model_ok = sum(r["model_ok"] for r in rows)
        route_ok = sum(r["route_ok"] for r in rows)
        rechecked = [r for r in rows if r["recheck"]]
        print(f"\n  [L-OD01] 실행 {len(rows)}회 | 모델 원래 의도 일치 {model_ok} | 최종 경로 일치 {route_ok} | 재확인 신호 {len(rechecked)}")
        for r in rows:
            mark = "O" if r["model_ok"] else "X"
            print(f"  {mark} #{r['attempt']} {r['message'][:24]:<24} 기대={'/'.join(r['expected'])} 모델={r['model_intent']} "
                  f"경로={r['route']} 상태={r['status']} 재확인={'예' if r['recheck'] else '-'}")


# %% 5-4단계 일정 에이전트·그래프 테스트 도우미 (구현 단위 4)
# 핵심 요약: 3단계 Harness로 Ollama·날씨·RAG를 가상으로 바꾸고, 일정 이해·시간 선택 단계만 정한 JSON을 돌려주는 가상 LLM을 더한다.
# 3단계 이해·답변 단계는 기존 FakeLLM 규칙을 그대로 쓴다. 모든 저장소는 임시 폴더에 있다.
S_BASE = {"operation": "other", "query_scope": "none", "answers_previous_question": False, "target_ids": [],
          **{key: "" for key in sa.TEXT_FIELDS}, "shift_base": "unknown", "delegate_choice": False,
          "other_day_ok": False, "check_weather": False, "multiple_requests": False, "preferences": [],
          "category_guess": "unknown", "request_mode": "request", "exclude_previous_target": False,
          "corrects_field": "none"}
RAIN = {"precipitation_mm": 2.0, "precipitation_probability_pct": 80}
WINDY = {"wind_speed_mps": 12.0}
SID = {"산책": "schedule_001", "시장": "schedule_002", "복지관": "schedule_003", "병원": "schedule_004"}
SCHEDULE_STAGES = ("schedule_understanding", "schedule_choice")


class ScheduleLLM(tod.FakeLLM):
    """schedule: {발화: 일정 이해 값 | [시도별 값] | 원문}, choices: [시간 선택 답 | 함수], fail: 일정 단계에서 낼 오류"""

    def __init__(self, understanding=None, schedule=None, choices=None, answers=None, fail=None):
        super().__init__(understanding=understanding, answers=answers)
        self.schedule = schedule or {}
        self.choices = list(choices or [])
        self.fail = fail

    def __call__(self, messages, schema, settings, *, stage, max_tokens=None, temperature=None):
        if stage not in SCHEDULE_STAGES:
            return super().__call__(messages, schema, settings, stage=stage, max_tokens=max_tokens, temperature=temperature)
        context = json.loads(messages[1]["content"])
        with self.lock:
            self.calls.append({"stage": stage, "context": context, "schema": schema, "temperature": temperature})
            if self.fail:
                raise od.DecisionError(*self.fail)
            if stage == "schedule_understanding":
                script = self.schedule.get(context["사용자_발화"], {})
                if isinstance(script, list):
                    attempt = sum(1 for c in self.calls if c["stage"] == stage and c["context"]["사용자_발화"] == context["사용자_발화"])
                    script = script[min(attempt, len(script)) - 1]
                raw = script if isinstance(script, str) else json.dumps(dict(S_BASE, **script), ensure_ascii=False)
            else:
                item = self.choices.pop(0) if self.choices else {"choice": "none", "reasons": []}
                item = item(context) if callable(item) else item
                raw = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return raw, {"stage": stage, "model": settings["model"], "eval_count": 10, "prompt_eval_count": 20,
                     "total_duration_ms": 5}


class FakeA:
    """저장소 모드에서도 주입한 A 함수를 그대로 부르는지 확인하는 가짜 A (기존 4개 인자만 받음)"""

    def __init__(self, *results):
        self.results, self.calls = list(results), []

    def __call__(self, request, *, device=None, index=None, conversation_state=None):
        self.calls.append({"request": request, "device": device, "index": index})
        return copy.deepcopy(self.results.pop(0) if len(self.results) > 1 else self.results[0])


def outing_result(original: str, proposed: str, *, purpose: str = "시장", city: str = "인천", route="outing_consultation"):
    """3단계가 날씨 비교로 만든 외출 시간 변경 제안 결과 (A 계약 모양)"""
    proposal = {"type": "propose_outing_time_change", "original_outing_at": original, "proposed_outing_at": proposed,
                "reason": "두 예보의 차이입니다. 강수확률 80 → 10%", "requires_approval": True, "applied": False}
    return {"status": "ok", "detail_code": None, "message": "", "route": route, "intent": route,
            "answer": "비 예보가 있어 시간을 바꾸는 것을 권해요.", "conversation_state": {"version": 1, "marker": "A"},
            "missing_fields": [], "clarification": None, "requires_user_confirmation": True, "proposed_action": proposal,
            "outing_request": {"city": city, "purpose": purpose}, "official_guidance": [], "weather": None,
            "comparison": None, "llm": {"understanding": {"intent": route}, "rules_applied": []},
            "decision": {"answer_source": "llm"}, "tool_trace": [], "usage": {"llm_calls": 2}}


class AgentCase(StoreCase):
    """임시 저장소(초기 일정) + 가상 LLM·예보·RAG. 기준 시각 2026-11-25(수) 10:00"""

    def setUp(self):
        super().setUp()
        self.schedules = self.ready()
        self.thread = sg.new_thread_id()

    def use(self, llm=None, weather=None, rag=None):
        self.llm = llm or ScheduleLLM()
        self.weather = weather or tod.FakeWeather()
        self.rag = rag or tod.FakeRAG()
        harness = tod.Harness(llm=self.llm, weather=self.weather, rag=self.rag, now=self.now)
        harness.__enter__()
        self.addCleanup(harness.__exit__, None, None, None)
        return harness

    def agent(self, message, state=None, **kwargs):
        """run_schedule_request를 실행하고 그래프 계약(질문 ⇔ needs_clarification, 제안 ⇔ 승인 요청)과 JSON 직렬화를 확인한다."""
        kwargs.setdefault("device", "xpu")
        kwargs.setdefault("index", tod.FAKE_INDEX)
        result = sa.run_schedule_request(message, store=self.schedules, conversation_state=state, now=self.now, **kwargs)
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        self.assertIn(result["route"], sa.ROUTES)
        self.assertEqual(result["status"] == "needs_input", result["route"] == "needs_clarification")
        self.assertEqual(result["requires_user_confirmation"], result["proposed_action"] is not None)
        self.assertEqual(bool(result["missing_fields"]), result["clarification"] is not None)
        if result["proposed_action"]:
            self.assertEqual(sg.check_feature_result(result, "S"), [])
            self.assertNotIn("approval", result["proposed_action"])
        return result

    def graph(self, **kwargs):
        kwargs.setdefault("a_device", "xpu")
        kwargs.setdefault("a_index", tod.FAKE_INDEX)
        return sg.build_graph(schedule_store=self.schedules, **kwargs)

    def say(self, graph, text):
        return sg.send_message(graph, self.thread, text)

    def values(self, graph):
        return graph.get_state(sg.run_config(self.thread)).values

    def doc(self):
        return self.schedules.load()

    def start(self, title):
        return item_of(self.doc(), title)["start_at"]

    def s_calls(self, stage="schedule_understanding"):
        return self.llm.stage_calls(stage)

    def apply_update(self, title, new_start):
        """서비스로 일정 하나를 바로 바꾼다. (승인 대기 중 다른 요청이 먼저 저장된 상황을 만들 때)"""
        doc = self.doc()
        made = sv.propose_update(doc, item_of(doc, title), new_start, self.now)
        return sv.apply_schedule_action(approved(got(made, "action")), self.schedules)


# %% M-AG1. 일정 이해 LLM 출력 검사 (요구 18, 보완 2: 모델이 고른 id·지어낸 표현을 그대로 믿지 않음)
class M_AG1_Understanding(AgentCase):

    def test_AG01_schema_matches_plan_and_limits_ids(self):
        """AG01 일정 이해 스키마는 확정한 24개 키이고, target_ids는 이번에 넘긴 일정 id만 고를 수 있다."""
        schema = sa.schedule_schema([SID["시장"]])
        self.assertEqual(list(schema["properties"]), list(sa.SCHEMA_KEYS))
        self.assertEqual(len(sa.SCHEMA_KEYS), 24)
        self.assertEqual(schema["properties"]["target_ids"]["items"]["enum"], [SID["시장"]])
        self.assertEqual(schema["properties"]["request_mode"]["enum"],
                         ["request", "negated", "conditional", "question", "correction"])
        self.assertEqual(schema["properties"]["operation"]["enum"],
                         ["query", "create", "update", "cancel", "complete", "other"])
        self.assertEqual(sa.schedule_schema([])["properties"]["target_ids"]["maxItems"], 0)
        self.assertFalse(schema["additionalProperties"])

    def test_AG02_unknown_id_dropped_and_llm_pick_not_confirmed(self):
        """AG02 모델이 없는 id를 내면 버리고, 있는 id라도 발화 근거가 없으면 확정하지 않고 '~ 말씀이신가요?'로 확인한다."""
        self.use(ScheduleLLM(schedule={"그 일정 바꿔 줘": {"operation": "update",
                                                         "target_ids": ["schedule_999", SID["시장"]]}}))
        before = self.raw()
        result = self.agent("그 일정 바꿔 줘")
        self.assertIn({"field": "target_ids", "value": "schedule_999"}, result["llm"]["discarded"])
        self.assertEqual(result["status"], "needs_input")
        self.assertIn("'시장 장보기'", result["answer"])
        self.assertIn("말씀이신가요", result["answer"])
        self.assertIsNone(result["proposed_action"])
        self.assertEqual(self.raw(), before)

    def test_AG03_ungrounded_text_is_discarded(self):
        """AG03 사용자가 말하지 않은 시각(모델이 지어낸 '오후 5시')은 버리고, 제안 대신 몇 시인지 묻는다."""
        self.use(ScheduleLLM(schedule={"시장 가는 시간 바꿔 줘": {"operation": "update", "target_text": "시장 가는",
                                                             "new_time_text": "오후 5시", "reason_text": "비 때문에"}}))
        result = self.agent("시장 가는 시간 바꿔 줘")
        fields = {d["field"] for d in result["llm"]["discarded"]}
        self.assertEqual(fields, {"new_time_text", "reason_text"})
        self.assertEqual(result["status"], "needs_input")
        self.assertIn("몇 시로 바꿀까요", result["answer"])
        self.assertIsNone(result["proposed_action"])

    def test_AG04_invalid_json_twice_keeps_previous_state(self):
        """AG04 형식이 틀린 답을 두 번 내면 llm_invalid_output으로 멈추고, 이전 일정 상담 상태를 그대로 돌려준다."""
        self.use(ScheduleLLM(schedule={"시장 시간 바꿔 줘": {"operation": "update", "target_text": "시장"},
                                       "오후 네 시": ["{깨진", "{\"operation\": \"update\"}"]}))
        first = self.agent("시장 시간 바꿔 줘")
        result = self.agent("오후 네 시", first["conversation_state"])
        self.assertEqual((result["status"], result["detail_code"], result["route"]),
                         ("llm_invalid_output", "schedule_understanding", "error"))
        self.assertEqual(result["conversation_state"], first["conversation_state"])
        self.assertEqual(len([c for c in self.s_calls() if c["context"]["사용자_발화"] == "오후 네 시"]), 2)

    def test_AG05_ollama_unavailable_and_store_errors(self):
        """AG05 Ollama 연결 실패는 llm_unavailable, 손상된 저장소는 store_error. 어느 경우도 파일을 초기화하거나 바꾸지 않는다."""
        self.use(ScheduleLLM(fail=("llm_unavailable", "connection_refused", "Ollama 서버에 연결하지 못했습니다.")))
        before = self.raw()
        result = self.agent("내일 일정 알려 줘")
        self.assertEqual((result["status"], result["detail_code"], result["route"]),
                         ("llm_unavailable", "connection_refused", "error"))
        self.assertEqual(self.raw(), before)
        self.path.write_bytes(b"{broken")
        result = self.agent("내일 일정 알려 줘")
        self.assertEqual((result["status"], result["detail_code"]), ("store_error", "corrupt"))
        self.assertEqual(self.raw(), b"{broken")
        self.assertEqual(len(self.s_calls()), 1)    # 저장소를 읽지 못하면 LLM을 부르지 않는다

    def test_AG06_multiple_targets_keep_llm_pick_as_suggestion(self):
        """AG06 '내일 오후 일정 바꿔 줘' → 후보 2개. 모델이 고른 일정은 (추천) 표시만 하고 확정하지 않는다. (multiple + llm_suggested)"""
        self.use(ScheduleLLM(schedule={"내일 오후 일정 바꿔 줘": {"operation": "update", "date_text": "내일", "time_text": "오후",
                                                             "target_ids": [SID["복지관"]]}}))
        result = self.agent("내일 오후 일정 바꿔 줘")
        self.assertEqual(result["status"], "needs_input")
        self.assertEqual(result["llm"]["target_lookup"]["status"], "multiple")
        self.assertEqual(result["llm"]["target_lookup"]["llm_suggested"], [SID["복지관"]])
        self.assertIn("1) ", result["answer"])
        self.assertIn("복지관 방문 [예정] (추천)", result["answer"])
        self.assertEqual(result["conversation_state"]["draft"]["candidates"], [SID["시장"], SID["복지관"]])


# %% M-AG6. 실측 결함 회귀 (5-4 실제 qwen2.5:3b 1차 실측에서 나온 실제 출력을 그대로 입력으로 씀)
# 핵심 요약: 3B 모델은 '등록된_일정' 표기를 텍스트 칸에 옮겨 적는다. 수정 전에는 근거 확인이 '내일(11월 26일 목요일)'·'시장 장보기'를
# 통째로 버려 날짜·대상을 잃고 확인 질문으로 빠졌고, 직전 발화의 '내일'이 새 날짜로 섞여 '이미 오후 2시예요'라고 답했다.
LIVE_QUERY_U = {"operation": "query", "query_scope": "date", "target_ids": [SID["시장"], SID["복지관"], SID["병원"]],
                "target_text": "내일", "date_text": "내일(11월 26일 목요일)",
                "time_text": "오늘(11월 25일 수요일) 오전 8시~오전 8시 30분 아침 산책 [시간 지남(완료 기록 없음)]",
                "new_date_text": "내일(11월 26일 목요일)", "new_time_text": "내일(11월 26일 목요일) 오후 2시~오후 3시 시장 장보기 [예정]",
                "title": "아침 산책", "city_text": "수요일", "reason_text": "아침 산책", "shift_base": "schedule",
                "category_guess": "outing"}
LIVE_TARGET_U = {"operation": "query", "query_scope": "target", "answers_previous_question": True, "target_ids": [SID["시장"]],
                 "target_text": "시장 장보기", "date_text": "내일(11월 26일 목요일)",
                 "time_text": "오후 2시~오후 3시 시장 장보기 [예정]", "shift_text": "없음", "duration_text": "없음",
                 "title": "시장 장보기", "city_text": "시장", "reason_text": "시장 장보기", "corrects_field": "time"}
LIVE_CHANGE_U = {"operation": "update", "query_scope": "target", "answers_previous_question": True, "target_ids": [SID["시장"]],
                 "target_text": "시장 가는", "date_text": "내일", "time_text": "오후 2시~오후 3시", "new_date_text": "내일",
                 "new_time_text": "오후 2시~오후 3시", "shift_text": "한 시간 늦게", "duration_text": "한 시간",
                 "title": "시장 가는", "city_text": "시장", "reason_text": "시장 가는 시간 바꾸기", "shift_base": "schedule",
                 "corrects_field": "time"}


class M_AG6_LiveRegressions(AgentCase):

    def test_AG31_system_labels_are_stripped_before_grounding(self):
        """AG31 '내일(11월 26일 목요일)'은 괄호 표기를 떼면 사용자가 말한 '내일'이다 → 날짜 조회로 답한다. 지어낸 시각·이름은 버린다."""
        self.use(ScheduleLLM(schedule={"내가 내일 뭐 하기로 했지?": LIVE_QUERY_U}))
        result = self.agent("내가 내일 뭐 하기로 했지?")
        self.assertEqual(result["llm"]["understanding"]["date_text"], "내일")
        self.assertIn("내일(11월 26일 목요일) 일정이에요.", result["answer"])
        self.assertEqual(result["status"], "ok")
        for key in ("time_text", "new_time_text", "title", "city_text", "reason_text"):
            self.assertEqual(result["llm"]["understanding"][key], "")

    def test_AG32_copied_title_keeps_user_words_only(self):
        """AG32 사용자는 '시장'이라고 말했고 모델은 저장된 제목 '시장 장보기'를 옮겼다 → '시장'만 남겨 대상을 찾고 시각을 답한다."""
        self.use(ScheduleLLM(schedule={"시장 몇 시였지?": LIVE_TARGET_U}))
        result = self.agent("시장 몇 시였지?")
        self.assertEqual(result["llm"]["understanding"]["target_text"], "시장")
        self.assertIn("'시장 장보기' 일정은 내일(11월 26일 목요일) 오후 2시~오후 3시예요.", result["answer"])

    def test_AG33_previous_utterance_is_not_grounding(self):
        """AG33 직전 발화의 '내일'을 이번 발화의 새 날짜로 쓰지 않는다. 같은 날짜로의 '날짜만 변경'은 변경 정보가 아니므로 몇 시인지 묻는다."""
        self.use(ScheduleLLM(schedule={"내가 내일 뭐 하기로 했지?": LIVE_QUERY_U, "시장 가는 시간 좀 바꿔 줘": LIVE_CHANGE_U}))
        state = self.agent("내가 내일 뭐 하기로 했지?")["conversation_state"]
        result = self.agent("시장 가는 시간 좀 바꿔 줘", state)
        self.assertEqual(result["llm"]["understanding"]["new_date_text"], "")
        self.assertIn("몇 시로 바꿀까요?", result["answer"])
        self.assertNotIn("이미 오후 2시", result["answer"])
        same_day = {"operation": "update", "target_text": "시장", "new_date_text": "내일"}
        self.use(ScheduleLLM(schedule={"시장 내일로 해 줘": same_day}))
        self.assertIn("몇 시로 바꿀까요?", self.agent("시장 내일로 해 줘")["answer"])


# %% I-AG8. 외부 코드 검토 결함 회귀 (커밋 전 검토에서 재현된 문제 1~4)
# 핵심 요약: 수정 전에는 조건을 무시한 제안, 모호한 시각을 버린 대상 확정, 날씨 조건 삭제, 행정구역 확대 해석이 일어났다.
class I_AG8_ReviewRegressions(AgentCase):

    def test_AG34_conditional_cancel_complete_create_are_not_proposed(self):
        """AG34 문제 1: '비 오면 취소해 줘'처럼 날씨 조건이 붙은 취소·완료·등록은 예보 확인 없이 제안하지 않고 지원하지 않는다고 안내한다.
        모델이 request로 분류해도 취소에 비 선호가 붙어 있으면 조건부로 본다. 조건 없는 취소는 그대로 제안한다."""
        cases = {"내일 시장 가는 거 비 오면 취소해 줘": {"operation": "cancel", "request_mode": "conditional",
                                                "target_text": "시장 가는", "date_text": "내일", "preferences": ["avoid_rain"]},
                 "비 오면 내일 장보기 취소해 줘": {"operation": "cancel", "target_text": "장보기", "date_text": "내일",
                                          "preferences": ["avoid_rain"]},
                 "비 오면 산책 완료로 해 줘": {"operation": "complete", "request_mode": "conditional", "target_text": "산책"},
                 "비 오면 내일 아침 여덟 시에 산책 등록해 줘": {"operation": "create", "request_mode": "conditional", "title": "산책",
                                                    "date_text": "내일", "time_text": "아침 여덟 시"},
                 "내일 장보기 취소해 줘": {"operation": "cancel", "target_text": "장보기", "date_text": "내일"}}
        self.use(ScheduleLLM(schedule=cases), weather=tod.FakeWeather(by_hour={14: RAIN}))
        for message in list(cases)[:4]:
            with self.subTest(message):
                result = self.agent(message)
                self.assertIsNone(result["proposed_action"])
                self.assertEqual(result["decision"]["kind"], "schedule_condition_unsupported")
                self.assertIn("기능은 아직 없어서 일정은 바꾸지 않았어요", result["answer"])
                self.assertIsNone(result["conversation_state"]["draft"])
        self.assertEqual(self.weather.calls, [])     # 조건을 판단한 척 예보를 조회하지도 않는다
        self.assertEqual(self.agent("내일 장보기 취소해 줘")["proposed_action"]["type"], "cancel_schedule")

    def test_AG35_ambiguous_clock_asks_before_choosing_target(self):
        """AG35 문제 2: '모레 두 시 일정 취소해 줘' → 오전·오후가 모호하면 날짜만으로 대상을 정하지 않고 묻는다. 대답하면 그 시각으로 찾는다."""
        self.schedules.transact(uuid.uuid4().hex, "d" * 64, add_schedule("독서 모임", at(27, 14), 60))
        self.use(ScheduleLLM(schedule={"모레 두 시 일정 취소해 줘": {"operation": "cancel", "date_text": "모레", "time_text": "두 시"},
                                       "오후 두 시": {"operation": "cancel", "answers_previous_question": True,
                                                  "time_text": "오후 두 시"},
                                       "오전 두 시": {"operation": "cancel", "answers_previous_question": True,
                                                  "time_text": "오전 두 시"}}))
        first = self.agent("모레 두 시 일정 취소해 줘")
        self.assertIsNone(first["proposed_action"])
        self.assertEqual(first["missing_fields"], ["target_time"])
        self.assertIn("오전인지 오후인지", first["answer"])
        self.assertEqual(first["conversation_state"]["draft"]["query"]["date"], "2026-11-27")
        afternoon = self.agent("오후 두 시", first["conversation_state"])
        self.assertEqual(afternoon["proposed_action"]["payload"]["schedule_id"], "schedule_005")
        night = self.agent("오전 두 시", first["conversation_state"])
        self.assertIsNone(night["proposed_action"])        # 02:00에는 일정이 없다 (14:00 일정을 대신 고르지 않음)

    def test_AG36_ambiguous_new_time_with_named_target(self):
        """AG36 문제 2 경계: '시장 가는 시간 다섯 시로 바꿔 줘'(모델이 '다섯 시'를 time_text에 넣음) → 대상은 이름으로 찾고,
        바꿀 시각의 오전·오후를 묻는다. 정상 시각·시간대 대상 식별은 그대로다."""
        message = "시장 가는 시간 다섯 시로 바꿔 줘"
        self.use(ScheduleLLM(schedule={message: {"operation": "update", "target_text": "시장 가는", "time_text": "다섯 시"},
                                       "내일 오후 2시 일정 취소해 줘": {"operation": "cancel", "date_text": "내일",
                                                                "time_text": "오후 2시"}}))
        result = self.agent(message)
        self.assertIn("ambiguous_time_text_used_as_new_time", result["llm"]["rules_applied"])
        self.assertEqual(result["conversation_state"]["draft"]["target_id"], SID["시장"])
        self.assertIn("오전인지 오후인지", result["answer"])
        exact = self.agent("내일 오후 2시 일정 취소해 줘")
        self.assertEqual(exact["proposed_action"]["payload"]["schedule_id"], SID["시장"])

    def test_AG37_weather_preference_is_kept(self):
        """AG37 문제 3: 사용자가 말한 날씨 조건을 지우지 않는다. 지역이 없으면 묻고, 실내 일정이면 확인할 수 없다고 안내하며,
        시간대만 말했어도 예보를 확인해 비 오는 시각을 고르지 않는다. 날씨 조건이 없는 시간대 요청은 기존처럼 가장 가까운 빈 시간이다."""
        self.schedules.transact(uuid.uuid4().hex, "e" * 64, add_schedule("독서 모임", at(26, 10), 60))
        create = "내일 비 안 오는 시간에 산책 등록해 줘"
        self.use(ScheduleLLM(schedule={
            create: {"operation": "create", "title": "산책", "date_text": "내일", "delegate_choice": True,
                     "preferences": ["avoid_rain"], "category_guess": "outing"},
            "인천": {"operation": "create", "answers_previous_question": True, "city_text": "인천"},
            "독서 모임 비 안 오는 시간으로 알아서 바꿔 줘": {"operation": "update", "target_text": "독서 모임",
                                                "delegate_choice": True, "preferences": ["avoid_rain"]},
            "시장 저녁쯤 비 안 오는 시간으로 바꿔 줘": {"operation": "update", "target_text": "시장", "new_time_text": "저녁쯤",
                                             "preferences": ["avoid_rain"]},
            "시장 저녁쯤으로 바꿔 줘": {"operation": "update", "target_text": "시장", "new_time_text": "저녁쯤"}}),
            weather=tod.FakeWeather(by_hour={13: RAIN, 14: RAIN, 18: RAIN}))
        first = self.agent(create)
        self.assertEqual((first["missing_fields"], first["proposed_action"]), (["city"], None))
        self.assertEqual(first["conversation_state"]["draft"]["preferences"], ["avoid_rain"])
        second = self.agent("인천", first["conversation_state"])
        start = got(second, "proposed_action")["payload"]["after"]["start_at"]
        self.assertNotIn(start[11:13], ("13", "14"))
        self.assertEqual(second["proposed_action"]["payload"]["after"]["city"], "인천")
        indoor = self.agent("독서 모임 비 안 오는 시간으로 알아서 바꿔 줘")
        self.assertIsNone(indoor["proposed_action"])
        self.assertIn("실외 외출 일정으로 저장되어 있지 않아 비 예보로 시간을 고를 수 없어요", indoor["answer"])
        # 저녁(18~21시) 후보는 원래 시각(14시)에 가까운 18:00·18:30·19:00. 18시 구간에 비 예보가 있다.
        evening = self.agent("시장 저녁쯤 비 안 오는 시간으로 바꿔 줘")
        self.assertEqual(evening["proposed_action"]["payload"]["after"]["start_at"], ss.iso(at(26, 19)))
        plain = self.agent("시장 저녁쯤으로 바꿔 줘")
        self.assertEqual(plain["proposed_action"]["payload"]["after"]["start_at"], ss.iso(at(26, 18)))

    def test_AG38_city_is_not_promoted_to_other_region(self):
        """AG38 문제 4: 사용자는 '경기도 광주'라고 했는데 모델이 '광주광역시'를 내면 그 행정구역을 확정하지 않는다.
        핵심 이름 '광주'만 남기고, 1단계 AMBIGUOUS_CITY_NAMES 규칙에 따라 어느 광주인지 묻는다. 그대로 말한 지역은 유지한다."""
        blank = {**{k: "" for k in sa.TEXT_FIELDS}, "preferences": []}
        grounded, discarded = sa.ground_texts(dict(blank, city_text="광주광역시"), "경기도 광주에서 친구 만나기로 했어", [])
        self.assertEqual(grounded["city_text"], "광주")
        self.assertIn({"field": "city_text", "value": "광주광역시", "kept": "광주"}, discarded)
        for said, model in (("인천광역시에서 친구 만나", "인천광역시"), ("인천에서 친구 만나", "인천광역시"),
                            ("광주광역시에서 친구 만나", "광주광역시")):
            with self.subTest(said):
                expected = model if model in said else model.replace("광역시", "")
                self.assertEqual(sa.ground_texts(dict(blank, city_text=model), said, [])[0]["city_text"], expected)
        message = "모레 오후 세 시에 경기도 광주에서 친구 만나기로 했어"
        self.use(ScheduleLLM(schedule={message: {"operation": "create", "title": "친구 만나기", "date_text": "모레",
                                                 "time_text": "오후 세 시", "city_text": "광주광역시", "category_guess": "outing"}}))
        result = self.agent(message)
        self.assertIsNone(result["proposed_action"])
        self.assertEqual(result["missing_fields"], ["city"])
        self.assertIn("'광주광역시' 또는 '광주, 경기도'", result["answer"])


# %% I-AG9. 2차 외부 검토 결함 회귀 (직접 정한 시각의 날씨 조건 우회, 지역 질문 반복)
# 핵심 요약: 수정 전에는 시각을 직접·상대 시간·충돌 대안으로 정하면 choose_slot()을 거치지 않아 날씨 조건 없이 제안했고,
# 지역이 없는 기존 실외 일정은 사용자가 지역을 알려 줘도 같은 질문을 반복했다.
class I_AG9_PreferenceGate(AgentCase):
    WALK = {"operation": "update", "target_text": "공원 걷기", "preferences": ["avoid_rain"]}
    CITY = {"operation": "update", "answers_previous_question": True, "city_text": "인천"}

    def add_walk(self):
        """지역이 저장되지 않은 실외 일정 '공원 걷기' (내일 10:00~11:00)"""
        made = sv.propose_create(self.doc(), title="공원 걷기", start=at(26, 10), minutes=60, now=self.now, city=None,
                                 category="outing")
        sv.apply_schedule_action(approved(got(made, "action")), self.schedules)

    def test_AG41_create_at_given_time_checks_rain(self):
        """AG41 '내일 오후 두 시에 비 안 오면 산책 등록해 줘'(모델: request + avoid_rain) → 지역을 묻고, 알려 주면 그 시각의 예보로 확인한다.
        오후 2시는 시장과 겹쳐 3시 대안이 되고, 3시에 비 예보가 있으면 제안하지 않는다. 비 예보가 없으면 예보와 함께 제안한다."""
        message = "내일 오후 두 시에 비 안 오면 산책 등록해 줘"
        script = {message: {"operation": "create", "title": "산책", "date_text": "내일", "time_text": "오후 두 시",
                            "preferences": ["avoid_rain"], "category_guess": "outing"},
                  "인천": {"operation": "create", "answers_previous_question": True, "city_text": "인천"}}
        for name, by_hour in (("비", {15: RAIN}), ("맑음", {})):
            with self.subTest(name):
                self.use(ScheduleLLM(schedule=script), weather=tod.FakeWeather(by_hour=by_hour))
                first = self.agent(message)
                self.assertEqual((first["missing_fields"], first["proposed_action"]), (["city"], None))
                # (3차 검토) 날씨 확인 전의 대체 시각은 '후보'로만 안내한다. (승인 제안 없음 검사는 위에서 그대로 유지)
                self.assertIn("그래서 비어 있는 오후 3시을(를) 대체 후보로 골라 예보를 확인해요", first["answer"])
                self.assertNotIn("오후 3시로 제안해요", first["answer"])
                second = self.agent("인천", first["conversation_state"])
                self.assertEqual(self.weather.times(), [("2026-11-26", "15:00")])
                if name == "비":
                    self.assertIsNone(second["proposed_action"])
                    self.assertIn("비 예보가 있어 제안하지 않았어요", second["answer"])
                else:
                    after = got(second, "proposed_action")["payload"]["after"]
                    self.assertEqual((after["start_at"], after["city"]), (ss.iso(at(26, 15)), "인천"))
                    self.assertIn("프로젝트 기준 위험 신호 없음(안전 보장 아님)", second["answer"])

    def test_AG42_update_at_given_time_without_stored_city(self):
        """AG42 '공원 걷기 비 안 오는 시간으로 오후 다섯 시에 바꿔 줘'(일정에 지역 없음) → 지역을 묻고, 대답한 지역으로 5시 예보를 확인한다.
        일정에 저장된 지역은 바꾸지 않는다. 날씨 조건이 없는 같은 요청은 기존처럼 바로 제안한다(예보 조회 없음)."""
        self.add_walk()
        message = "공원 걷기 비 안 오는 시간으로 오후 다섯 시에 바꿔 줘"
        self.use(ScheduleLLM(schedule={message: dict(self.WALK, new_time_text="오후 다섯 시"), "인천": self.CITY,
                                       "공원 걷기 오후 다섯 시로 바꿔 줘": {"operation": "update", "target_text": "공원 걷기",
                                                                    "new_time_text": "오후 다섯 시"}}))
        first = self.agent(message)
        self.assertEqual((first["missing_fields"], first["proposed_action"]), (["city"], None))
        second = self.agent("인천", first["conversation_state"])
        action = got(second, "proposed_action")
        self.assertEqual(action["payload"]["after"]["start_at"], ss.iso(at(26, 17)))
        self.assertIsNone(action["payload"]["after"]["city"])
        self.assertIn("예보는 말씀하신 지역 '인천' 기준이에요. 일정에 저장된 지역은 바꾸지 않아요.", second["answer"])
        self.assertEqual(sorted(self.weather.times()), [("2026-11-26", "10:00"), ("2026-11-26", "17:00")])
        calls = len(self.weather.calls)
        plain = self.agent("공원 걷기 오후 다섯 시로 바꿔 줘")       # 날씨 조건 없는 일반 변경은 기존처럼 바로 제안
        self.assertEqual(plain["proposed_action"]["payload"]["after"]["start_at"], ss.iso(at(26, 17)))
        self.assertEqual(len(self.weather.calls), calls)
        self.assertEqual(sv.apply_schedule_action(approved(action), self.schedules)["status"], "ok")
        self.assertIsNone(item_of(self.doc(), "공원 걷기")["city"])

    def test_AG43_relative_and_conflict_alternative_keep_rain_condition(self):
        """AG43 상대 시간('한 시간 늦춰')과 충돌 대안(4시 → 5시)도 비 조건을 확인한다. 대안 시각에 비 예보가 있으면 제안하지 않는다."""
        self.add_walk()
        self.use(ScheduleLLM(schedule={
            "공원 걷기 비 안 오게 한 시간 늦춰 줘": dict(self.WALK, shift_text="한 시간 늦춰", shift_base="schedule"),
            "시장 비 안 오는 시간으로 오후 네 시로 바꿔 줘": {"operation": "update", "target_text": "시장", "new_time_text": "오후 네 시",
                                                "preferences": ["avoid_rain"]}}),
            weather=tod.FakeWeather(by_hour={17: RAIN}))
        shifted = self.agent("공원 걷기 비 안 오게 한 시간 늦춰 줘")
        self.assertEqual((shifted["missing_fields"], shifted["proposed_action"]), (["city"], None))
        conflict = self.agent("시장 비 안 오는 시간으로 오후 네 시로 바꿔 줘")
        self.assertIsNone(conflict["proposed_action"])
        # (3차 검토) '5시로 제안해요'와 '5시에는 비 예보가 있어 제안하지 않았어요'가 함께 나오지 않게 한다.
        self.assertIn("그래서 비어 있는 오후 5시을(를) 대체 후보로 골라 예보를 확인해요", conflict["answer"])
        self.assertNotIn("오후 5시로 제안해요", conflict["answer"])
        self.assertIn("오후 5시에는 비 예보가 있어 제안하지 않았어요", conflict["answer"])

    def test_AG44_city_answer_is_used_for_delegated_choice(self):
        """AG44 지역이 없는 실외 일정의 선택 위임: 지역 질문 → '인천' → 같은 질문을 반복하지 않고 인천 예보로 비 없는 시간을 제안한다."""
        self.add_walk()
        self.use(ScheduleLLM(schedule={"공원 걷기 비 안 오는 시간으로 알아서 바꿔 줘": dict(self.WALK, delegate_choice=True),
                                       "인천": self.CITY}),
                 weather=tod.FakeWeather(by_hour={10: RAIN, 11: RAIN}))
        first = self.agent("공원 걷기 비 안 오는 시간으로 알아서 바꿔 줘")
        self.assertEqual(first["missing_fields"], ["city"])
        second = self.agent("인천", first["conversation_state"])
        action = got(second, "proposed_action")
        self.assertNotIn(action["payload"]["after"]["start_at"][11:13], ("10", "11"))
        self.assertIsNone(action["payload"]["after"]["city"])
        self.assertTrue(all(call["city"] == "인천" for call in self.weather.calls))


# %% I-AG10. 3차 외부 검토 결함 회귀 (지역 질문 뒤 시간대 소실)
# 핵심 요약: 수정 전에는 '저녁'을 말한 뒤 지역 질문에 답하면 시간대가 사라져, 몇 시인지 다시 묻거나(시간 선택을 맡기지 않은 경우)
# 07~21시 전체에서 후보를 골라 저녁이 아닌 시각을 제안했다(맡긴 경우). 등록·변경 모두 같았다.
class I_AG10_PeriodAfterQuestion(AgentCase):
    CITY = {"answers_previous_question": True, "city_text": "인천"}

    def create_flow(self, delegate: bool, follow: str = "인천", follow_u: dict | None = None):
        message = "내일 저녁 비 안 오는 시간" + ("으로 알아서 " if delegate else "에 ") + "산책 등록해 줘"
        self.use(ScheduleLLM(schedule={message: {"operation": "create", "title": "산책", "date_text": "내일", "time_text": "저녁",
                                                 "delegate_choice": delegate, "preferences": ["avoid_rain"],
                                                 "category_guess": "outing"},
                                       follow: follow_u or dict(self.CITY, operation="create")}),
                 weather=tod.FakeWeather(by_hour={18: RAIN}))
        first = self.agent(message)
        self.assertEqual((first["missing_fields"], first["proposed_action"]), (["city"], None))
        self.assertEqual(first["conversation_state"]["draft"]["pending"], {"period": "저녁"})
        return self.agent(follow, first["conversation_state"])

    def test_AG45_create_keeps_evening_after_city_question(self):
        """AG45 사례 A·B: '내일 저녁 비 안 오는 시간에(으로 알아서) 산책 등록해 줘' → 지역 질문 → '인천' →
        다시 묻지 않고 저녁(18~21시) 안에서, 비 예보가 있는 18시 구간을 피해 제안한다. 날씨 조건·선택 위임도 그대로다."""
        for delegate in (False, True):
            with self.subTest(delegate=delegate):
                second = self.create_flow(delegate)
                after = got(second, "proposed_action")["payload"]["after"]
                self.assertEqual(after["start_at"], ss.iso(at(26, 19)))
                self.assertEqual(after["city"], "인천")
                self.assertIn("period_kept_after_follow_up_question", second["llm"]["rules_applied"])
                self.assertTrue(all(c["outing_time"][:2] in ("18", "19", "20") for c in self.weather.calls))

    def test_AG46_correction_after_question_wins(self):
        """AG46 지역을 답하면서 시각을 정정하면('인천, 아침 아홉 시로') 정정한 시각이 우선한다. 날짜만 고치면 시간대(저녁)는 이어 쓴다."""
        timed = self.create_flow(False, "인천, 아침 아홉 시로", dict(self.CITY, operation="create", time_text="아침 아홉 시"))
        self.assertEqual(timed["proposed_action"]["payload"]["after"]["start_at"], ss.iso(at(26, 9)))
        dated = self.create_flow(False, "인천, 모레로", dict(self.CITY, operation="create", date_text="모레"))
        self.assertEqual(dated["proposed_action"]["payload"]["after"]["start_at"], ss.iso(at(27, 19)))

    def test_AG47_update_keeps_evening_after_city_question(self):
        """AG47 변경도 같다: 지역이 없는 실외 일정을 '저녁 비 안 오는 시간으로' → 지역 질문 → '인천' → 저녁 안에서 비를 피해 제안한다.
        일정의 지역은 바꾸지 않는다."""
        made = sv.propose_create(self.doc(), title="공원 걷기", start=at(26, 10), minutes=60, now=self.now, city=None,
                                 category="outing")
        sv.apply_schedule_action(approved(got(made, "action")), self.schedules)
        message = "공원 걷기 저녁 비 안 오는 시간으로 바꿔 줘"
        self.use(ScheduleLLM(schedule={message: {"operation": "update", "target_text": "공원 걷기", "new_time_text": "저녁",
                                                 "preferences": ["avoid_rain"]},
                                       "인천": dict(self.CITY, operation="update")}),
                 weather=tod.FakeWeather(by_hour={18: RAIN}))
        first = self.agent(message)
        self.assertEqual(first["missing_fields"], ["city"])
        second = self.agent("인천", first["conversation_state"])
        after = got(second, "proposed_action")["payload"]["after"]
        self.assertEqual((after["start_at"], after["city"]), (ss.iso(at(26, 19)), None))


# %% M-AG7. 실제 LLM 테스트 판정 함수 (문제 5: 일정 id만으로 '유효한 제안'·'저장 성공'을 판정하지 않음)
class M_AG7_LiveJudge(StoreCase):

    def test_AG39_proposal_judge_checks_time_duration_and_target(self):
        """AG39 기대 제안(시장 17:00~18:00, 60분, 대상·버전·변경 전 내용 일치)만 유효하다. 같은 일정 id라도 18시·소요 시간 변경은 실패다."""
        doc = seed_doc()
        market = item_of(doc, "시장 장보기")
        good = got(sv.propose_update(doc, market, at(26, 17), FIXED_NOW), "action")
        self.assertEqual(judge_live_proposal(good, doc), [])
        late = got(sv.propose_update(doc, market, at(26, 18), FIXED_NOW), "action")
        self.assertEqual(late["payload"]["schedule_id"], LIVE_TARGET_ID)          # 수정 전 판정(id만)으로는 통과했던 제안
        self.assertTrue(judge_live_proposal(late, doc))
        stretched = copy.deepcopy(good)
        stretched["payload"]["after"]["end_at"] = ss.iso(at(26, 19))
        self.assertIn("소요 시간이 원래와 다름", judge_live_proposal(stretched, doc))
        stale = copy.deepcopy(good)
        stale["payload"]["expected_version"] = 2
        self.assertIn("버전·변경 전 내용이 저장된 일정과 다름", judge_live_proposal(stale, doc))
        self.assertTrue(judge_live_proposal(None, doc))

    def test_AG40_saved_judge_checks_file_history_and_reload(self):
        """AG40 저장 성공은 파일의 시각·버전·다른 일정 불변·이력 1건·새 저장소 객체로 다시 읽은 값까지 같아야 한다."""
        store = self.ready()
        original = store.load()
        self.assertTrue(judge_live_saved(original, original, original))           # 저장 전: 실패
        action = got(sv.propose_update(original, item_of(original, "시장 장보기"), at(26, 17), FIXED_NOW), "action")
        self.assertEqual(sv.apply_schedule_action(approved(action), store)["status"], "ok")
        saved = store.load()
        self.assertEqual(judge_live_saved(saved, ss.ScheduleStore(self.path, clock=lambda: self.now).load(), original), [])
        self.assertIn("새 저장소 객체로 다시 읽은 값이 다름", judge_live_saved(saved, original, original))


# %% I-AG2. 조회 (요구 07·14·26·27·41·42, 시나리오 3·10)
class I_AG2_Queries(AgentCase):

    def test_AG07_forgotten_schedule_queries(self):
        """AG07 시나리오 3: '내가 내일 뭐 하기로 했지?' → '시장 가는 거 몇 시였지?' 저장 기록만으로 답하고, 승인·저장·날씨·RAG가 없다."""
        self.use(ScheduleLLM(schedule={"내가 내일 뭐 하기로 했지?": {"operation": "query", "query_scope": "date", "date_text": "내일"},
                                       "시장 가는 거 몇 시였지?": {"operation": "query", "query_scope": "target",
                                                               "target_text": "시장 가는"}}))
        before = self.raw()
        first = self.agent("내가 내일 뭐 하기로 했지?")
        self.assertIn("오후 2시~오후 3시 시장 장보기 [예정]", first["answer"])
        self.assertIn("오후 4시~오후 5시 복지관 방문 [예정]", first["answer"])
        second = self.agent("시장 가는 거 몇 시였지?", first["conversation_state"])
        self.assertIn("'시장 장보기' 일정은 내일(11월 26일 목요일) 오후 2시~오후 3시예요. (상태: 예정)", second["answer"])
        for result in (first, second):
            self.assertIsNone(result["proposed_action"])
        self.assertEqual((self.weather.calls, self.rag.calls), ([], []))
        self.assertEqual(self.raw(), before)

    def test_AG08_next_and_remaining(self):
        """AG08 다음 일정은 지금 이후의 예정 일정(시간이 지난 산책은 제외), 오늘 남은 일정은 없다고 안내한다."""
        self.use(ScheduleLLM(schedule={"다음 약속이 뭐야?": {"operation": "query", "query_scope": "next"},
                                       "오늘 남은 일정 있어?": {"operation": "query", "query_scope": "remaining"}}))
        self.assertIn("다음 일정: 내일(11월 26일 목요일) 오후 2시~오후 3시 시장 장보기 [예정]", self.agent("다음 약속이 뭐야?")["answer"])
        self.assertEqual(self.agent("오늘 남은 일정 있어?")["answer"], "오늘 남은 일정이 없어요.")

    def test_AG09_daily_summary_with_and_without_b(self):
        """AG09 시나리오 10: 하루 정리. 지난 산책은 '시간 지남(완료 기록 없음)'으로만 표시, B 미연결이면 확인할 수 없다고 안내(가짜 기록 없음)."""
        message = "오늘 한 일하고 내일 일정 알려 줘"
        self.use(ScheduleLLM(schedule={message: {"operation": "query", "query_scope": "summary"}}))
        result = self.agent(message)
        self.assertIn("아침 산책 [시간 지남(완료 기록 없음)]", result["answer"])
        self.assertIn("시장 장보기 [예정]", result["answer"])
        self.assertIn(sa.B_NOT_CONNECTED, result["answer"])
        record = {"action_id": "a" * 32, "message": "우유 1개 구매 도움 요청", "executed_at": ss.iso(self.now)}
        connected = self.agent(message, context={"b_records": [record]})
        self.assertIn("• 우유 1개 구매 도움 요청", connected["answer"])
        self.assertNotIn(sa.B_NOT_CONNECTED, connected["answer"])

    def test_AG10_history_from_store(self):
        """AG10 변경 이력은 LLM 기억이 아니라 저장된 action_history에서 찾는다. 원래 시각도 기록에서 알려 준다."""
        self.use(ScheduleLLM(schedule={"아까 몇 시로 바꿨더라?": {"operation": "query", "query_scope": "history"},
                                       "원래 시장에 몇 시에 가려고 했지?": {"operation": "query", "query_scope": "history",
                                                                     "target_text": "시장"}}))
        self.assertEqual(self.agent("아까 몇 시로 바꿨더라?")["answer"], "오늘 바꾸거나 등록한 기록이 없어요.")
        self.assertEqual(got(self.apply_update("시장 장보기", at(26, 17)), "status"), "ok")
        result = self.agent("아까 몇 시로 바꿨더라?")
        self.assertIn("'시장 장보기' 내일(11월 26일 목요일) 오후 2시 → 내일(11월 26일 목요일) 오후 5시(으)로 변경", result["answer"])
        original = self.agent("원래 시장에 몇 시에 가려고 했지?")
        self.assertIn("처음에는 내일(11월 26일 목요일) 오후 2시 일정이었어요.", original["answer"])


# %% I-AG3. 부정·조건부·정정·화제 전환 (보완 2의 v2 문장 8개, 요구 05)
class I_AG3_RequestModes(AgentCase):

    def test_AG11_change_tomorrow_asks_which(self):
        """AG11 '내일 일정을 바꿔 줘.' → 내일 일정이 2개라 번호로 묻고, '첫 번째'(모델이 other로 분류해도) 대답은 후보 안에서만 고른다."""
        self.use(ScheduleLLM(schedule={"내일 일정을 바꿔 줘.": {"operation": "update", "date_text": "내일"},
                                       "첫 번째": {"operation": "other"}}))
        first = self.agent("내일 일정을 바꿔 줘.")
        self.assertEqual(first["status"], "needs_input")
        self.assertIn("1) 내일(11월 26일 목요일) 오후 2시~오후 3시 시장 장보기", first["answer"])
        second = self.agent("첫 번째", first["conversation_state"])
        self.assertIn("short_answer_linked_to_draft", second["llm"]["rules_applied"])
        self.assertIn("'시장 장보기' 일정은 지금", second["answer"])
        self.assertEqual(second["conversation_state"]["draft"]["target_id"], SID["시장"])

    def test_AG12_negated_sentences_make_no_proposal(self):
        """AG12 '내일 일정 바꾸지 마.', '시간을 바꿀 필요는 없어.' → 제안 없음, 진행 중인 초안도 버린다."""
        self.use(ScheduleLLM(schedule={"시장 시간 바꿔 줘": {"operation": "update", "target_text": "시장"},
                                       "내일 일정 바꾸지 마.": {"operation": "update", "request_mode": "negated", "date_text": "내일"},
                                       "시간을 바꿀 필요는 없어.": {"operation": "update", "request_mode": "negated"}}))
        draft = self.agent("시장 시간 바꿔 줘")["conversation_state"]
        self.assertIsNotNone(draft["draft"])
        for message, state in (("내일 일정 바꾸지 마.", draft), ("시간을 바꿀 필요는 없어.", None)):
            with self.subTest(message):
                result = self.agent(message, state)
                self.assertIsNone(result["proposed_action"])
                self.assertIsNone(result["conversation_state"]["draft"])
                self.assertEqual(result["decision"]["kind"], "schedule_kept")
                self.assertTrue(result["answer"].startswith("알겠어요."))

    def test_AG13_conditional_rain_uses_real_condition(self):
        """AG13 '비가 오면 바꿀까?': 비 예보가 확인되면 비 없는 빈 시간으로 제안, 비 신호가 없으면 그대로,
        조회 실패는 '비가 안 온다'로 보지 않고, 바람 신호만 있으면 비 조건으로 보지 않는다."""
        message = "비가 오면 시장 시간을 바꿀까?"
        script = {message: {"operation": "update", "request_mode": "conditional", "target_text": "시장",
                            "preferences": ["avoid_rain"], "check_weather": True}}
        cases = [("비", {13: RAIN, 14: RAIN}, None, "schedule_proposal"), ("맑음", {}, None, "schedule_condition_not_met"),
                 ("조회 실패", {}, "network_error", "schedule_condition_unknown"),
                 ("바람만", {14: WINDY}, None, "schedule_condition_not_met")]
        for name, by_hour, fail, kind in cases:
            with self.subTest(name):
                self.use(ScheduleLLM(schedule=script, choices=[{"choice": "C3", "reasons": ["less_rain", "no_conflict"]}]),
                         weather=tod.FakeWeather(by_hour=by_hour, fail=fail))
                result = self.agent(message)
                self.assertEqual(result["decision"]["kind"], kind)
                if kind == "schedule_proposal":
                    self.assertEqual(result["proposed_action"]["payload"]["after"]["start_at"], ss.iso(at(26, 15)))
                    self.assertIn("비 예보가 있어요", result["answer"])
                else:
                    self.assertIsNone(result["proposed_action"])
                if name == "조회 실패":
                    self.assertIn("판단할 수 없어요", result["answer"])
                    self.assertNotIn("보이지 않아", result["answer"])
                if name == "바람만":
                    self.assertIn("비 신호가 보이지 않아", result["answer"])
                    self.assertIn("위험 신호 있음(강풍)", result["answer"])

    def test_AG14_back_to_original_time_discards_draft(self):
        """AG14 제안 뒤 '아니, 원래 시간에 갈래.' → 원래 시각을 그대로 두고 초안을 버린다."""
        self.use(ScheduleLLM(schedule={"시장 시간 저녁 일곱 시로 바꿔 줘": {"operation": "update", "target_text": "시장",
                                                                     "new_time_text": "저녁 일곱 시"},
                                       "아니, 원래 시간에 갈래.": {"operation": "update", "request_mode": "negated"}}))
        first = self.agent("시장 시간 저녁 일곱 시로 바꿔 줘")
        self.assertEqual(first["proposed_action"]["payload"]["after"]["start_at"], ss.iso(at(26, 19)))
        result = self.agent("아니, 원래 시간에 갈래.", first["conversation_state"])
        self.assertEqual(result["answer"], "알겠어요. '시장 장보기' 일정은 원래대로 내일(11월 26일 목요일) 오후 2시에 그대로 둘게요.")
        self.assertIsNone(result["conversation_state"]["draft"])
        self.assertIsNone(result["proposed_action"])

    def test_AG15_other_schedule_excludes_previous_target(self):
        """AG15 '아까 말한 일정 말고 다른 일정이야.' → 직전에 고른 시장을 빼고 다시 찾는다. 남은 복지관은 변경 제한이라 바꾸지 않는다."""
        self.use(ScheduleLLM(schedule={"내일 오후 일정 바꿔 줘": {"operation": "update", "date_text": "내일", "time_text": "오후"},
                                       "첫 번째": {"operation": "update", "answers_previous_question": True},
                                       "아까 말한 일정 말고 다른 일정이야.": {
                                           "operation": "update", "request_mode": "correction", "corrects_field": "target",
                                           "exclude_previous_target": True}}))
        state = self.agent("내일 오후 일정 바꿔 줘")["conversation_state"]
        state = self.agent("첫 번째", state)["conversation_state"]
        self.assertEqual(state["draft"]["target_id"], SID["시장"])
        result = self.agent("아까 말한 일정 말고 다른 일정이야.", state)
        self.assertIn("previous_target_excluded", result["llm"]["rules_applied"])
        self.assertIn("'복지관 방문'은(는) 변경 제한 일정", result["answer"])
        self.assertIsNone(result["proposed_action"])

    def test_AG16_time_correction_inherits_ampm_only_with_basis(self):
        """AG16 '저녁 일곱 시 아니고 여덟 시야.' → 직전 시각의 오후를 이어 20시로 다시 계산한다. 이어받을 근거가 없으면 묻는다."""
        self.use(ScheduleLLM(schedule={
            "시장 시간 저녁 일곱 시로 바꿔 줘": {"operation": "update", "target_text": "시장", "new_time_text": "저녁 일곱 시"},
            "저녁 일곱 시 아니고 여덟 시야.": {"operation": "update", "request_mode": "correction", "corrects_field": "time",
                                         "new_time_text": "여덟 시"},
            "시장은 여덟 시 아니고 아홉 시야": {"operation": "update", "request_mode": "correction", "corrects_field": "time",
                                          "target_text": "시장", "new_time_text": "아홉 시"}}))
        first = self.agent("시장 시간 저녁 일곱 시로 바꿔 줘")
        result = self.agent("저녁 일곱 시 아니고 여덟 시야.", first["conversation_state"])
        self.assertEqual(result["proposed_action"]["payload"]["after"]["start_at"], ss.iso(at(26, 20)))
        self.assertIn("앞에서 말씀하신 시간대를 이어 오후 8시로 계산했어요.", result["answer"])
        self.assertNotEqual(result["proposed_action"]["action_id"], first["proposed_action"]["action_id"])
        unsure = self.agent("시장은 여덟 시 아니고 아홉 시야")
        self.assertEqual(unsure["status"], "needs_input")
        self.assertIn("오전인지 오후인지", unsure["answer"])

    def test_AG17_topic_switch_hands_off_and_keeps_draft(self):
        """AG17 일정 상담 중 '오늘 날씨 어때?', '우유가 떨어졌어' → handoff(A·B로 넘김). 진행 중인 일정 초안은 그대로 남긴다."""
        self.use(ScheduleLLM(schedule={"시장 시간 바꿔 줘": {"operation": "update", "target_text": "시장"},
                                       "오늘 날씨 어때?": {"operation": "other"}, "우유가 떨어졌어": {"operation": "other"}}))
        state = self.agent("시장 시간 바꿔 줘")["conversation_state"]
        for message in ("오늘 날씨 어때?", "우유가 떨어졌어"):
            with self.subTest(message):
                result = self.agent(message, state)
                self.assertEqual((result["route"], result["decision"]["kind"]), ("handoff", "handoff"))
                self.assertEqual(result["conversation_state"]["draft"], state["draft"])

    def test_AG29_date_and_period_correction(self):
        """AG29 요구 05: 5시 제안 뒤 '아니, 모레 오전으로 해 줘' → 바꾼 항목(날짜·시간대)만 갱신해 모레 오전의 빈 시간으로 다시 제안한다."""
        self.use(ScheduleLLM(schedule={
            "시장 시간 오후 다섯 시로 바꿔 줘": {"operation": "update", "target_text": "시장", "new_time_text": "오후 다섯 시"},
            "아니, 모레 오전으로 해 줘": {"operation": "update", "request_mode": "correction", "corrects_field": "date",
                                   "new_date_text": "모레", "new_time_text": "오전"}}))
        first = self.agent("시장 시간 오후 다섯 시로 바꿔 줘")
        self.assertEqual(first["proposed_action"]["payload"]["after"]["start_at"], ss.iso(at(26, 17)))
        result = self.agent("아니, 모레 오전으로 해 줘", first["conversation_state"])
        self.assertEqual(result["proposed_action"]["payload"]["schedule_id"], SID["시장"])
        self.assertEqual(result["proposed_action"]["payload"]["after"]["start_at"], ss.iso(at(27, 10)))
        self.assertIn("말씀하신 시간대(오전) 안에서", result["answer"])

    def test_AG30_new_time_in_time_text_and_multiple_requests(self):
        """AG30 모델이 바꿀 시각을 time_text에 넣어도(찾을 일정의 시각으로 보면 없음) 이름으로 다시 찾아 새 시각으로 쓴다.
        한 문장에 여러 작업을 요청하면 첫 작업만 제안하고 나머지는 나중에 말해 달라고 안내한다."""
        message = "시장 가는 시간을 오후 5시로 바꾸고 병원도 취소해 줘"
        self.use(ScheduleLLM(schedule={message: {"operation": "update", "target_text": "시장 가는", "time_text": "오후 5시",
                                                 "multiple_requests": True}}))
        result = self.agent(message)
        self.assertIn("time_text_used_as_new_time", result["llm"]["rules_applied"])
        self.assertEqual(result["proposed_action"]["payload"]["after"]["start_at"], ss.iso(at(26, 17)))
        self.assertIn("한 번에 하나씩 처리할게요.", result["answer"])


# %% I-AG4. 등록·취소·완료 (요구 38·39·40, 시나리오 4·5·6)
class I_AG4_CreateCancelComplete(AgentCase):

    def test_AG18_create_then_store_after_approval(self):
        """AG18 시나리오 4: '내일 아침 여덟 시에 산책할 거야. 기억해 줘.' → 등록 제안(소요 시간 기본 60분 표시), 승인 뒤에만 저장"""
        message = "내일 아침 여덟 시에 산책할 거야. 기억해 줘."
        self.use(ScheduleLLM(schedule={message: {"operation": "create", "title": "산책", "date_text": "내일",
                                                 "time_text": "아침 여덟 시", "category_guess": "outing"}}))
        before = self.raw()
        result = self.agent(message)
        action = got(result, "proposed_action")
        self.assertEqual(action["type"], "create_schedule")
        self.assertEqual(action["payload"]["after"]["start_at"], ss.iso(at(26, 8)))
        self.assertIn("소요 시간을 말씀하지 않아 60분으로 잡았어요.", result["answer"])
        self.assertEqual(self.weather.calls, [])     # 지역을 말하지 않아 예보를 조회하지 않음 (지어내지 않음)
        self.assertEqual(self.raw(), before)
        self.assertEqual(got(sv.apply_schedule_action(approved(action), self.schedules), "status"), "ok")
        self.assertEqual(item_of(self.doc(), "산책")["start_at"], ss.iso(at(26, 8)))

    def test_AG19_create_asks_one_missing_field_at_a_time(self):
        """AG19 '산책 일정 등록해 줘' → 날짜 질문 → '내일' → 시각 질문 → '아침 여덟 시' → 등록 제안"""
        self.use(ScheduleLLM(schedule={"산책 일정 등록해 줘": {"operation": "create", "title": "산책"},
                                       "내일": {"operation": "create", "answers_previous_question": True, "date_text": "내일"},
                                       "아침 여덟 시": {"operation": "create", "answers_previous_question": True,
                                                    "time_text": "아침 여덟 시"}}))
        first = self.agent("산책 일정 등록해 줘")
        self.assertEqual((first["missing_fields"], first["answer"]), (["date"], "'산책' 일정은 어느 날인가요?"))
        second = self.agent("내일", first["conversation_state"])
        self.assertEqual((second["missing_fields"], second["answer"]), (["time"], "'산책'은(는) 몇 시에 시작하나요?"))
        third = self.agent("아침 여덟 시", second["conversation_state"])
        self.assertEqual(third["proposed_action"]["payload"]["after"]["start_at"], ss.iso(at(26, 8)))

    def test_AG20_cancel_keeps_record_and_protects(self):
        """AG20 시나리오 5: '내일 장보기는 안 갈래' → 취소 제안 → 승인 시 cancelled(기록은 남음). 변경 제한 일정(병원)은 취소하지 않는다."""
        self.use(ScheduleLLM(schedule={"내일 장보기는 안 갈래": {"operation": "cancel", "target_text": "장보기", "date_text": "내일"},
                                       "병원 예약 취소해 줘": {"operation": "cancel", "target_text": "병원 예약"}}))
        result = self.agent("내일 장보기는 안 갈래")
        action = got(result, "proposed_action")
        self.assertEqual(action["type"], "cancel_schedule")
        self.assertEqual(got(sv.apply_schedule_action(approved(action), self.schedules), "status"), "ok")
        self.assertEqual(item_of(self.doc(), "시장 장보기")["status"], "cancelled")
        self.assertEqual(len(self.doc()["schedules"]), 4)
        protected = self.agent("병원 예약 취소해 줘")
        self.assertIsNone(protected["proposed_action"])
        self.assertIn("변경 제한 일정이라 SafeMate가 취소하지 않아요", protected["answer"])

    def test_AG21_complete_rules_by_state(self):
        """AG21 시나리오 6: 사용자가 끝냈다고 말한 일정만 완료 제안. 내일 일정은 완료로 기록하지 않고, 이미 완료면 그렇게 안내한다."""
        self.use(ScheduleLLM(schedule={"아까 산책 끝냈어": {"operation": "complete", "target_text": "산책"},
                                       "내일 시장 다녀왔어": {"operation": "complete", "target_text": "시장", "date_text": "내일"}}))
        result = self.agent("아까 산책 끝냈어")
        action = got(result, "proposed_action")
        self.assertEqual(action["type"], "complete_schedule")
        self.assertIsNone(self.agent("내일 시장 다녀왔어")["proposed_action"])
        self.assertIn("아직 날짜가 오지 않은 일정이에요.", self.agent("내일 시장 다녀왔어")["answer"])
        self.assertEqual(got(sv.apply_schedule_action(approved(action), self.schedules), "status"), "ok")
        again = self.agent("아까 산책 끝냈어")
        self.assertIsNone(again["proposed_action"])
        self.assertEqual(again["answer"], "이미 완료로 기록된 일정이에요.")


# %% I-AG5. 날씨·선택 위임·상대 시간 (요구 03·11·13·15·16·17, 시나리오 2·7·8)
class I_AG5_WeatherAndDelegation(AgentCase):
    DELEGATE = "내일 비 안 오는 시간으로 시장 가는 거 바꿔 줘"
    DELEGATE_U = {"operation": "update", "target_text": "시장 가는", "date_text": "내일", "delegate_choice": True,
                  "preferences": ["avoid_rain"]}

    def test_AG22_delegated_choice_verified(self):
        """AG22 시나리오 2: 선택 위임 → Python 후보 3개(14:30·13:30·15:00)와 예보 → LLM이 15:00·less_rain 선택 → Python 검증 통과"""
        self.use(ScheduleLLM(schedule={self.DELEGATE: self.DELEGATE_U},
                             choices=[{"choice": "C3", "reasons": ["less_rain", "no_conflict"]}]),
                 weather=tod.FakeWeather(by_hour={13: RAIN, 14: RAIN}))
        result = self.agent(self.DELEGATE)
        self.assertEqual(result["proposed_action"]["payload"]["after"]["start_at"], ss.iso(at(26, 15)))
        self.assertEqual(result["decision"]["answer_source"], "llm_choice")
        self.assertEqual(self.weather.times(), [("2026-11-26", t) for t in ("14:00", "14:30", "13:30", "15:00")])
        choice = self.s_calls("schedule_choice")[0]["context"]
        self.assertEqual([c["예보_상태"] for c in choice["후보"]], ["signal", "signal", "no_signal"])
        self.assertIn("원래 시각보다 비 예보가 적어요", result["answer"])
        self.assertIn("안전을 보장한다는 뜻은 아니에요", result["answer"])
        self.assertEqual(self.rag.calls, [])     # 바꿀 시각에 위험 신호가 없으면 공식 수칙을 검색하지 않는다

    def test_AG23_wrong_llm_reason_falls_back_to_python(self):
        """AG23 LLM이 비 오는 후보(C1)를 '비가 적다'고 고르면 Python 검증에서 버리고 규칙으로 15:00을 고른다. (python_fallback)"""
        self.use(ScheduleLLM(schedule={self.DELEGATE: self.DELEGATE_U},
                             choices=[{"choice": "C1", "reasons": ["less_rain"]}]),
                 weather=tod.FakeWeather(by_hour={13: RAIN, 14: RAIN}))
        result = self.agent(self.DELEGATE)
        self.assertEqual(result["proposed_action"]["payload"]["after"]["start_at"], ss.iso(at(26, 15)))
        self.assertEqual(result["decision"]["answer_source"], "python_fallback")
        self.assertIn("llm_choice_rejected_by_python_check", result["llm"]["rules_applied"])

    def test_AG24_no_good_candidate_no_recommendation(self):
        """AG24 예보를 확인하지 못한 후보(unknown)는 '비가 안 오는' 근거로 쓰지 않는다. 맞는 후보가 없으면 제안하지 않는다."""
        self.use(ScheduleLLM(schedule={self.DELEGATE: self.DELEGATE_U},
                             choices=[{"choice": "C1", "reasons": ["no_conflict"]}]),
                 weather=tod.FakeWeather(fail="network_error"))
        result = self.agent(self.DELEGATE)
        self.assertIsNone(result["proposed_action"])
        self.assertEqual(result["status"], "needs_input")
        self.assertIn("비 신호가 없다고 확인된 빈 시간을 찾지 못해 바꾸지 않았어요.", result["answer"])

    def test_AG25_relative_shift_from_schedule(self):
        """AG25 시나리오 7: '한 시간 늦게' → 원래 일정 기준 15:00, '두 시간 늦게' → 16:00 충돌 → 17:00 대안, 기준을 모르면 묻는다."""
        self.use(ScheduleLLM(schedule={
            "시장 가는 거 한 시간 늦게 가려고": {"operation": "update", "target_text": "시장 가는", "shift_text": "한 시간 늦게",
                                        "shift_base": "schedule"},
            "시장 가는 거 두 시간 늦게 가려고": {"operation": "update", "target_text": "시장 가는", "shift_text": "두 시간 늦게",
                                        "shift_base": "schedule"},
            "시장 두 시간 늦춰 줘": {"operation": "update", "target_text": "시장", "shift_text": "두 시간 늦춰"},
            "원래 시간 기준으로": {"operation": "update", "answers_previous_question": True, "shift_base": "schedule"}}))
        one = self.agent("시장 가는 거 한 시간 늦게 가려고")
        self.assertEqual(one["proposed_action"]["payload"]["after"]["start_at"], ss.iso(at(26, 15)))
        two = self.agent("시장 가는 거 두 시간 늦게 가려고")
        self.assertEqual(two["proposed_action"]["payload"]["after"]["start_at"], ss.iso(at(26, 17)))
        self.assertIn("오후 4시에는 다른 일정(오후 4시 복지관 방문)이 있어요. 그래서 비어 있는 오후 5시로 제안해요.", two["answer"])
        unsure = self.agent("시장 두 시간 늦춰 줘")
        self.assertEqual(unsure["missing_fields"], ["shift_base"])
        resolved = self.agent("원래 시간 기준으로", unsure["conversation_state"])
        self.assertEqual(resolved["proposed_action"]["payload"]["after"]["start_at"], ss.iso(at(26, 17)))

    def test_AG26_weather_only_when_needed(self):
        """AG26 늦잠 같은 개인 사정 변경·실내 일정 등록은 날씨를 조회하지 않는다. 실외 일정 변경은 원래·바꿀 시각 2구간만 조회한다."""
        self.use(ScheduleLLM(schedule={
            "늦잠 자서 시장 가는 거 한 시간 늦출게": {"operation": "update", "target_text": "시장 가는", "shift_text": "한 시간 늦출게",
                                           "shift_base": "schedule", "reason_text": "늦잠"},
            "모레 오후 세 시에 친구 집에 갈 거야": {"operation": "create", "title": "친구 집", "date_text": "모레",
                                          "time_text": "오후 세 시", "category_guess": "home"},
            "시장 가는 거 한 시간 늦게 가려고": {"operation": "update", "target_text": "시장 가는", "shift_text": "한 시간 늦게",
                                        "shift_base": "schedule"}}))
        personal = self.agent("늦잠 자서 시장 가는 거 한 시간 늦출게")
        self.assertEqual(personal["proposed_action"]["payload"]["reason"], "늦잠")
        indoor = self.agent("모레 오후 세 시에 친구 집에 갈 거야")
        self.assertEqual(indoor["proposed_action"]["type"], "create_schedule")
        self.assertEqual(self.weather.calls, [])
        outdoor = self.agent("시장 가는 거 한 시간 늦게 가려고")
        self.assertEqual(self.weather.times(), [("2026-11-26", "14:00"), ("2026-11-26", "15:00")])
        self.assertEqual([f["label"] for f in outdoor["proposed_action"]["payload"]["forecast"]], ["원래 시각", "바꿀 시각"])

    def test_AG27_recheck_forecast_is_fresh(self):
        """AG27 시나리오 8: '아까 바꾼 장보기 일정 괜찮겠지?' → 실제 변경 기록에서 일정을 찾고 매번 예보를 새로 조회한다. 바꾸려면 새 승인"""
        message = "아까 바꾼 장보기 일정 괜찮겠지?"
        self.use(ScheduleLLM(schedule={message: {"operation": "query", "query_scope": "target", "check_weather": True,
                                                 "target_text": "아까 바꾼 장보기 일정"}}),
                 weather=tod.FakeWeather(by_hour={17: RAIN}))
        self.apply_update("시장 장보기", at(26, 17))
        first = self.agent(message)
        second = self.agent(message)
        self.assertEqual(self.weather.times(), [("2026-11-26", "17:00")] * 2)
        self.assertIn("'시장 장보기' 일정은 내일(11월 26일 목요일) 오후 5시~오후 6시", first["answer"])
        self.assertIn("위험 신호 있음(호우)", first["answer"])
        self.assertIn("바꾸려면 새로 승인이 필요해요", first["answer"])
        self.assertIsNone(second["proposed_action"])
        self.assertEqual(second["conversation_state"]["draft"]["awaiting"], "time")

    def test_AG28_beyond_forecast_range(self):
        """AG28 예보 기간(16일) 밖 날짜로 옮기면 일정 제안은 만들고, 날씨는 조회하지 않은 채 '확인 못함'으로 안내한다."""
        message = "시장 가는 거 다음 달 20일로 옮겨 줘"
        self.use(ScheduleLLM(schedule={message: {"operation": "update", "target_text": "시장 가는", "new_date_text": "다음 달 20일"}}))
        result = self.agent(message)
        self.assertEqual(result["proposed_action"]["payload"]["after"]["start_at"], "2026-12-20T14:00:00+09:00")
        self.assertEqual(self.weather.times(), [("2026-11-26", "14:00")])     # 12월 20일은 조회하지 않음
        self.assertIn("예보 조회 기간(오늘부터 16일) 밖", result["answer"])


# %% I-GR1. 그래프 구조 (보완 4: 저장소 미연결이면 4단계 그래프와 같음)
def previous_graph_module():
    """5-4단계 직전 커밋(3a8aaec)의 safemate_graph.py를 임시 모듈로 불러온다. git을 쓸 수 없으면 None"""
    try:
        source = subprocess.run(["git", "show", "3a8aaec:safemate_graph.py"], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    folder = tempfile.mkdtemp()
    path = Path(folder) / "safemate_graph_before_5_4.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("safemate_graph_before_5_4", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # LangGraph가 State의 타입 힌트를 읽을 때 모듈 이름으로 전역 이름을 찾으므로 잠시 등록한다.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class I_GR1_Structure(AgentCase):

    def test_GR01_unconnected_graph_is_unchanged(self):
        """GR01 저장소를 연결하지 않으면 노드 6개이고, Mermaid 그림이 5-4단계 직전 그래프와 글자 하나까지 같다."""
        before = previous_graph_module()
        if before is None:
            self.skipTest("git으로 이전 그래프를 읽지 못했습니다.")
        self.addCleanup(sys.modules.pop, "safemate_graph_before_5_4", None)
        now_graph = sg.build_graph().get_graph()
        self.assertEqual(set(now_graph.nodes) - {"__start__", "__end__"},
                         {"intake", "a_node", "b_node", "approval", "execute", "close"})
        self.assertEqual(now_graph.draw_mermaid(), before.build_graph().get_graph().draw_mermaid())

    def test_GR02_connected_graph_adds_schedule_node(self):
        """GR02 저장소를 연결하면 schedule_node가 추가되고, 일정 작업 4종이 허용 표에 있다. (기존 A·B 작업 종류 유지)"""
        graph = self.graph().get_graph()
        self.assertEqual(set(graph.nodes) - {"__start__", "__end__"},
                         {"intake", "a_node", "b_node", "schedule_node", "approval", "execute", "close"})
        mermaid = graph.draw_mermaid()
        for edge in ("intake -.-> schedule_node", "a_node -.-> schedule_node", "schedule_node -.-> approval",
                     "schedule_node -.-> a_node", "approval -.-> execute", "approval -.-> close"):
            self.assertIn(edge, mermaid)
        self.assertEqual(sg.ALLOWED_ACTIONS["propose_outing_time_change"], {"source": "outing_decision", "feature": "A"})
        self.assertEqual(sg.ALLOWED_ACTIONS["request_caregiver_help"], {"source": "living_support", "feature": "B"})
        for kind in ("create_schedule", "update_schedule", "cancel_schedule", "complete_schedule"):
            self.assertEqual(sg.ALLOWED_ACTIONS[kind], {"source": "schedule_service", "feature": "S"})


# %% I-GR2. 대표 시나리오 1 전체 흐름 (완료 기준: 조회 → 4시 충돌 → 5시 제안 → 승인 → 파일 변경 → 재조회 → 재실행 후 유지)
FLOW_A = {"내가 내일 뭐 하기로 했지?": tod.u(intent="schedule_query"), "시장 가는 시간 좀 바꿔 줘": tod.u(intent="schedule_change"),
          "시장 몇 시였지?": tod.u(intent="schedule_query")}
FLOW_S = {"내가 내일 뭐 하기로 했지?": {"operation": "query", "query_scope": "date", "date_text": "내일"},
          "시장 가는 시간 좀 바꿔 줘": {"operation": "update", "target_text": "시장 가는"},
          "오후 네 시쯤": {"operation": "update", "answers_previous_question": True, "new_time_text": "오후 네 시쯤"},
          "시장 몇 시였지?": {"operation": "query", "query_scope": "target", "target_text": "시장"}}


class I_GR2_Scenario1(AgentCase):

    def test_GR03_full_change_flow_and_restart(self):
        """GR03 시나리오 1 전체: 저장은 승인 뒤 execute에서만, 시장 일정만 바뀌고 이력 1건, 다시 물으면 오후 5시, 새 그래프·저장소에서도 유지"""
        self.use(ScheduleLLM(understanding=FLOW_A, schedule=FLOW_S), weather=tod.FakeWeather(by_hour={14: RAIN}))
        graph = self.graph()
        before = self.raw()

        # 1. 조회: 승인·저장·날씨 없음
        view = self.say(graph, "내가 내일 뭐 하기로 했지?")
        self.assertIn("시장 장보기 [예정]", view["answer"])
        self.assertEqual((view["pending"], view["active_feature"], self.weather.calls), (None, None, []))
        self.assertEqual(view["debug"]["routing"]["model_intent"], "schedule_query")

        # 2. 변경 요청 → 새 시각 질문 (일정 상담 진행 중 표시)
        view = self.say(graph, "시장 가는 시간 좀 바꿔 줘")
        self.assertIn("몇 시로 바꿀까요?", view["answer"])
        self.assertEqual(view["active_feature"], "S")

        # 3. '오후 네 시쯤' → 복지관과 충돌 → 오후 5시 제안 (A를 거치지 않고 일정 상담으로 바로 감)
        a_calls = len(self.llm.stage_calls("understanding"))
        view = self.say(graph, "오후 네 시쯤")
        self.assertEqual(len(self.llm.stage_calls("understanding")), a_calls)
        pending = got(view, "pending")
        self.assertEqual(pending["type"], "update_schedule")
        for text in ("[변경] 시장 장보기: 11월 26일 오후 2시~오후 3시 → 11월 26일 오후 5시~오후 6시", "소요 시간 60분",
                     "다른 일정과 겹치지 않음", "위험 신호 있음(호우)"):
            self.assertIn(text, pending["preview"])
        self.assertIn("오후 4시에는 다른 일정(오후 4시 복지관 방문)이 있어요.", view["answer"])
        self.assertEqual(self.raw(), before)                     # 승인 전 파일 불변

        # 4. 승인 → execute에서만 저장. 시장만 바뀌고 이력 1건, A 상태 초기화
        view = sg.press_button(graph, self.thread, "approve", pending["action_id"])
        self.assertEqual(view["answer"], "'시장 장보기' 일정을 내일(11월 26일 목요일) 오후 5시로 바꿔 저장했어요.")
        doc = self.doc()
        self.assertEqual((self.start("시장 장보기"), item_of(doc, "시장 장보기")["version"]), (ss.iso(at(26, 17)), 2))
        self.assertEqual(self.start("복지관 방문"), ss.iso(at(26, 16)))
        self.assertEqual([h["operation"] for h in doc["action_history"]], ["update"])
        values = self.values(graph)
        self.assertIsNone(values["a_conversation_state"])
        self.assertIsNone(values["schedule_conversation_state"]["draft"])
        self.assertEqual(values["executed_action_ids"], [pending["action_id"]])
        self.assertIn("a_state_reset_after_schedule_change", [e["event"] for e in values["action_log"]])

        # 5. 다시 물으면 저장소를 다시 읽어 오후 5시
        self.assertIn("오후 5시~오후 6시", self.say(graph, "시장 몇 시였지?")["answer"])

        # 6. 프로그램 재실행 흉내: 새 저장소 객체·새 그래프(새 InMemorySaver)·새 대화방
        reopened = ss.ScheduleStore(self.path, clock=lambda: self.now)
        self.assertEqual(item_of(reopened.load(), "시장 장보기")["start_at"], ss.iso(at(26, 17)))
        fresh = sg.build_graph(schedule_store=reopened, a_device="xpu", a_index=tod.FAKE_INDEX)
        view = sg.send_message(fresh, sg.new_thread_id(), "시장 몇 시였지?")
        self.assertIn("오후 5시~오후 6시", view["answer"])


# %% I-GR3. 승인·거절 안전 (요구 20~24, 보완 5)
class I_GR3_ApprovalSafety(AgentCase):

    def proposal(self, graph):
        self.say(graph, "시장 가는 시간 좀 바꿔 줘")
        return got(self.say(graph, "오후 네 시쯤"), "pending")

    def setUp(self):
        super().setUp()
        self.use(ScheduleLLM(understanding=FLOW_A, schedule={
            **FLOW_S, "오늘 인천 날씨 어때?": {"operation": "other"}}))

    def test_GR04_reject_keeps_file(self):
        """GR04 거절 → 파일·이력 그대로, 작업에 맞는 거절 문장, 일정 상담 끝"""
        graph = self.graph()
        before = self.raw()
        pending = self.proposal(graph)
        view = sg.press_button(graph, self.thread, "reject", pending["action_id"])
        self.assertEqual(view["answer"], "알겠어요, 그대로 둘게요. 원래 일정(오후 2시)을 유지합니다.")
        self.assertEqual(self.raw(), before)
        self.assertEqual((view["active_feature"], self.values(graph)["schedule_conversation_state"]["draft"]), (None, None))

    def test_GR05_new_question_cancels_without_saving(self):
        """GR05 승인 대기 중 새 질문 → 이전 제안 취소(저장 없음). 일정 상담 중이라도 날씨 질문은 A기능이 답한다."""
        graph = self.graph()
        before = self.raw()
        self.proposal(graph)
        view = self.say(graph, "오늘 인천 날씨 어때?")
        self.assertEqual(view["notice"], sg.CANCELLED)
        self.assertEqual(view["debug"]["feature"], "A")
        self.assertEqual(self.raw(), before)
        self.assertIn("cancelled", [e["event"] for e in self.values(graph)["action_log"]])

    def test_GR06_double_and_stale_buttons_apply_once(self):
        """GR06 승인 두 번 → 한 번만 저장. 새 제안이 나온 뒤 이전 제안의 버튼은 무시한다."""
        graph = self.graph()
        old = self.proposal(graph)
        new = got(self.say(graph, "오후 네 시쯤"), "pending")
        self.assertNotEqual(old["action_id"], new["action_id"])
        self.assertEqual(sg.press_button(graph, self.thread, "approve", old["action_id"])["answer"], sg.STALE_BUTTON)
        sg.press_button(graph, self.thread, "approve", new["action_id"])
        self.assertEqual(sg.press_button(graph, self.thread, "approve", new["action_id"])["answer"], sg.NO_PENDING)
        self.assertEqual(len(self.doc()["action_history"]), 1)

    def test_GR07_tampered_proposal_is_blocked(self):
        """GR07 승인 대기 중 State의 제안 내용(바꿀 시각)이 바뀌면 승인 카드와 다르므로 실행하지 않는다."""
        graph = self.graph()
        before = self.raw()
        pending = self.proposal(graph)
        state = copy.deepcopy(self.values(graph)["pending_action"])
        state["payload"]["after"].update(start_at=ss.iso(at(26, 19)), end_at=ss.iso(at(26, 20)))
        graph.update_state(sg.run_config(self.thread), {"pending_action": state})
        view = sg.press_button(graph, self.thread, "approve", pending["action_id"])
        self.assertEqual(view["answer"], sg.EXECUTION_BLOCKED)
        self.assertEqual(view["debug"]["execution_result"]["reason"], "content_changed")
        self.assertEqual(self.raw(), before)

    def test_GR08_changes_while_waiting_block_saving(self):
        """GR08 승인 대기 중 같은 일정이 다른 요청으로 바뀌면 stale, 그 시간에 새 일정이 생기면 conflict → 저장하지 않고 성공으로 기록하지 않는다."""
        for case in ("stale", "conflict"):
            with self.subTest(case):
                self.setUp()
                graph = self.graph()
                pending = self.proposal(graph)
                if case == "stale":
                    self.apply_update("시장 장보기", at(26, 12))
                else:
                    made = sv.propose_create(self.doc(), title="독서 모임", start=at(26, 17), minutes=60, now=self.now)
                    sv.apply_schedule_action(approved(got(made, "action")), self.schedules)
                before = self.raw()
                view = sg.press_button(graph, self.thread, "approve", pending["action_id"])
                self.assertEqual(view["debug"]["execution_result"]["status"], case)
                self.assertIn("저장하지 않았어요", view["answer"])
                self.assertEqual(self.raw(), before)
                self.assertEqual(self.values(graph).get("executed_action_ids") or [], [])

    def test_GR09_corrupted_store_at_approval(self):
        """GR09 승인 직전에 파일이 손상되면 저장하지 않고, 손상된 파일을 초기화하거나 덮어쓰지 않는다."""
        graph = self.graph()
        pending = self.proposal(graph)
        self.path.write_bytes(b"{broken")
        view = sg.press_button(graph, self.thread, "approve", pending["action_id"])
        self.assertEqual(view["debug"]["execution_result"]["status"], "error")
        self.assertIn("저장하지 않았어요", view["answer"])
        self.assertEqual(self.raw(), b"{broken")


# %% I-GR4. 경로·상태 분리 (요구 18·28, 보완 2·4: 재확인 1회, 무한 순환 방지, A·B·S 상태 분리)
class I_GR4_Routing(AgentCase):

    def test_GR10_switch_to_weather_and_b_keeps_schedule_draft(self):
        """GR10 일정 상담 중 날씨 질문 → A, 생활 지원 → B(미연결 안내). 일정 초안은 S 상태에 그대로 남는다."""
        self.use(ScheduleLLM(understanding={"시장 가는 시간 좀 바꿔 줘": tod.u(intent="schedule_change"),
                                            "우유가 떨어졌어": tod.u(intent="living_support"),
                                            "오늘 인천 날씨 어때?": tod.u(intent="weather_question", needs_weather=True,
                                                                     city_text="인천", date_text="오늘")},
                             schedule={"시장 가는 시간 좀 바꿔 줘": FLOW_S["시장 가는 시간 좀 바꿔 줘"],
                                       "오늘 인천 날씨 어때?": {"operation": "other"}, "우유가 떨어졌어": {"operation": "other"}}))
        graph = self.graph()
        self.say(graph, "시장 가는 시간 좀 바꿔 줘")
        draft = self.values(graph)["schedule_conversation_state"]["draft"]
        view = self.say(graph, "오늘 인천 날씨 어때?")
        values = self.values(graph)
        self.assertEqual((view["debug"]["feature"], values["turn_hops"]), ("A", 1))
        self.assertEqual(values["routing"]["handoff_from"], "schedule_node")
        self.assertEqual(values["schedule_conversation_state"]["draft"], draft)
        self.assertEqual(view["active_feature"], "S")
        view = self.say(graph, "우유가 떨어졌어")
        self.assertEqual((view["answer"], view["debug"]["feature"]), (sg.B_UNAVAILABLE, "B"))
        self.assertEqual(self.values(graph)["schedule_conversation_state"]["draft"]["target_id"], SID["시장"])

    def test_GR11_recheck_uses_schedule_llm_once(self):
        """GR11 3단계가 날씨 질문으로 분류했지만 재확인 신호가 있으면 일정 이해 LLM에게 1회 더 맡긴다.
        일정 변경 요청이면 일정 답변을, 부정이면 이미 만든 A 답변을 그대로 쓴다. (A 답변을 다시 만들지 않음)"""
        change = "내일 오후 2시에 인천 가는데 비 오면 시간 바꿀까?"
        keep = "내일 오후 2시 인천 날씨 보고 시간 바꾸지 말까?"
        a_u = dict(intent="weather_question", needs_weather=True, city_text="인천", date_text="내일", time_text="오후 2시")
        self.use(ScheduleLLM(understanding={change: tod.u(**a_u), keep: tod.u(**a_u)},
                             schedule={change: {"operation": "update", "request_mode": "conditional", "date_text": "내일",
                                                "time_text": "오후 2시", "preferences": ["avoid_rain"], "check_weather": True},
                                       keep: {"operation": "update", "request_mode": "negated", "date_text": "내일"}},
                             choices=[{"choice": "C3", "reasons": ["less_rain"]}]),
                 weather=tod.FakeWeather(by_hour={13: RAIN, 14: RAIN}))
        graph = self.graph()
        view = self.say(graph, change)
        routing = view["debug"]["routing"]
        self.assertEqual((routing["model_intent"], routing["recheck"], routing["final_route"]),
                         ("weather_question", True, "schedule"))
        self.assertEqual(got(view, "pending")["type"], "update_schedule")
        self.assertEqual(len(self.llm.stage_calls("answer")), 1)
        sg.press_button(graph, self.thread, "reject", view["pending"]["action_id"])
        view = self.say(graph, keep)
        values = self.values(graph)
        self.assertEqual(values["routing"]["recheck_result"], "kept_outing_answer")
        self.assertEqual(view["answer"], values["last_result"]["answer"])
        self.assertEqual(view["debug"]["feature"], "A")
        self.assertIsNone(view["pending"])

    def test_GR12_no_infinite_loop_between_a_and_s(self):
        """GR12 3단계는 일정, 일정 이해는 '일정 아님'으로 서로 다르게 보면 되돌려 보내지 않고 다시 묻는다. (턴당 넘김 1회)"""
        self.use(ScheduleLLM(understanding={"그거 있잖아": tod.u(intent="schedule_query"),
                                            "시장 가는 시간 좀 바꿔 줘": tod.u(intent="schedule_change"),
                                            "음 그게": tod.u(intent="schedule_query")},
                             schedule={"그거 있잖아": {"operation": "other"}, "음 그게": {"operation": "other"},
                                       "시장 가는 시간 좀 바꿔 줘": FLOW_S["시장 가는 시간 좀 바꿔 줘"]}))
        graph = self.graph()
        self.assertEqual(self.say(graph, "그거 있잖아")["answer"], sg.SCHEDULE_UNSURE)
        self.assertEqual(self.values(graph)["turn_hops"], 1)
        self.say(graph, "시장 가는 시간 좀 바꿔 줘")
        view = self.say(graph, "음 그게")          # S(일정 아님) → A(일정) → 다시 S로 보내지 않음
        self.assertEqual(view["answer"], sg.SCHEDULE_UNSURE)
        self.assertEqual(len([c for c in self.s_calls() if c["context"]["사용자_발화"] == "음 그게"]), 1)

    def test_GR13_end_words_and_cancel_sentence(self):
        """GR13 일정 상담 중 '그만'만 말하면 S 상담만 끝낸다. '내일 장보기 취소해 줘'처럼 문장이면 일정 취소 요청이다."""
        self.use(ScheduleLLM(understanding={"시장 가는 시간 좀 바꿔 줘": tod.u(intent="schedule_change")},
                             schedule={"시장 가는 시간 좀 바꿔 줘": FLOW_S["시장 가는 시간 좀 바꿔 줘"],
                                       "내일 장보기 취소해 줘": {"operation": "cancel", "target_text": "장보기", "date_text": "내일"}}))
        graph = self.graph()
        self.say(graph, "시장 가는 시간 좀 바꿔 줘")
        view = self.say(graph, "그만")
        self.assertEqual((view["answer"], view["active_feature"]), (sg.S_CONSULTATION_ENDED, None))
        self.assertIsNone(self.values(graph).get("b_conversation_state"))
        self.say(graph, "시장 가는 시간 좀 바꿔 줘")
        view = self.say(graph, "내일 장보기 취소해 줘")
        self.assertEqual(got(view, "pending")["type"], "cancel_schedule")

    def test_GR19_living_support_still_goes_to_b(self):
        """GR19 시나리오 9: 저장소를 연결해도 '우유하고 계란이 떨어졌어'는 B로 간다. B 호출 모양·B 승인·B 상태는 4단계와 같고 일정은 만들지 않는다."""
        living = tsg.FakeLiving(tsg.b_question(1), tsg.b_proposal_result(2))
        executor = tsg.FakeExecutor()
        self.use(ScheduleLLM(understanding={"우유하고 계란이 떨어졌어": tod.u(intent="living_support", user_goal="장보기 도움")}))
        before = self.raw()
        unconnected = self.graph()
        self.assertEqual(sg.send_message(unconnected, sg.new_thread_id(), "우유하고 계란이 떨어졌어")["answer"], sg.B_UNAVAILABLE)
        graph = self.graph(b_runner=living, b_executor=executor)
        first = self.say(graph, "우유하고 계란이 떨어졌어")
        self.assertEqual((first["active_feature"], first["debug"]["feature"]), ("B", "B"))
        self.assertEqual(living.calls[0]["request"], {"user_message": "우유하고 계란이 떨어졌어",
                                                      "context": {"source": "outing_decision", "user_goal": "장보기 도움"}})
        second = self.say(graph, "우유 하나, 계란 한 판")
        self.assertEqual(got(second, "pending")["type"], "request_caregiver_help")
        sg.press_button(graph, self.thread, "approve", second["pending"]["action_id"])
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(self.raw(), before)
        self.assertEqual(self.s_calls(), [])
        self.assertIsNone(self.values(graph).get("schedule_conversation_state"))

    def test_GR20_life_record_reader_injection(self):
        """GR20 요구 30: B 기록 읽기 함수를 주입하지 않으면 '확인할 수 없음', 주입하면(명세 2.3 기록 형식) 하루 정리에 그 기록만 보여 준다."""
        message = "오늘 한 일 정리해 줘"
        self.use(ScheduleLLM(understanding={message: tod.u(intent="schedule_query")},
                             schedule={message: {"operation": "query", "query_scope": "summary"}}))
        self.assertIn(sa.B_NOT_CONNECTED, self.say(self.graph(), message)["answer"])
        log = Path(self.temp.name) / "mock_action_log.jsonl"
        log.write_text(json.dumps({"action_id": "c" * 32, "type": "request_caregiver_help", "mock": True,
                                   "payload": {"message": "우유 1개 구매 도움을 부탁드립니다."},
                                   "approved_at": ss.iso(self.now), "executed_at": ss.iso(self.now)},
                                  ensure_ascii=False) + "\n", encoding="utf-8")
        graph = self.graph(life_record_reader=lambda: sv.read_living_action_log(log))
        view = sg.send_message(graph, sg.new_thread_id(), message)
        self.assertIn("• 우유 1개 구매 도움을 부탁드립니다.", view["answer"])
        self.assertNotIn(sa.B_NOT_CONNECTED, view["answer"])

    def test_GR14_states_are_separate_and_custom_runner_kept(self):
        """GR14 A·B·S 대화 상태는 서로 다른 칸에 따로 저장된다. 저장소 모드에서도 주입한 A 함수를 기존 인자로 그대로 부른다."""
        fake = FakeA(dict(outing_result("2026-11-26T14:00:00+09:00", "2026-11-26T17:00:00+09:00"),
                          requires_user_confirmation=False, proposed_action=None, answer="외출 상담 답변"))
        self.use()
        graph = self.graph(a_runner=fake)
        self.say(graph, "외출 상담")
        self.assertEqual(fake.calls[0]["device"], "xpu")
        values = self.values(graph)
        self.assertEqual(values["a_conversation_state"], {"version": 1, "marker": "A"})
        self.assertIsNone(values.get("b_conversation_state"))
        self.assertIsNone(values.get("schedule_conversation_state"))


# %% I-GR5. 기존 A 외출 시간 변경 제안의 저장소 연결 (보완 1, 요구 13)
class I_GR5_OutingLink(AgentCase):

    def test_GR15_single_match_becomes_checked_update(self):
        """GR15 목적·날짜·시각·지역이 맞는 일정이 하나면, 대상 id·버전·변경 전 내용을 담은 update_schedule 카드로 바꾼다."""
        self.use()
        graph = self.graph(a_runner=FakeA(outing_result("2026-11-26T14:00:00+09:00", "2026-11-26T17:00:00+09:00")))
        view = self.say(graph, "내일 오후 2시 시장 가는데 비 오면 어떡해?")
        pending = got(view, "pending")
        action = self.values(graph)["pending_action"]
        self.assertEqual((pending["type"], pending["source"]), ("update_schedule", "schedule_service"))
        self.assertEqual((action["payload"]["schedule_id"], action["payload"]["expected_version"]), (SID["시장"], 1))
        self.assertIn("저장된 '시장 장보기' 일정과 연결해", view["answer"])
        sg.press_button(graph, self.thread, "approve", pending["action_id"])
        self.assertEqual(self.start("시장 장보기"), ss.iso(at(26, 17)))

    def test_GR16_no_stored_outing_proposes_create(self):
        """GR16 저장된 일정에 없는 외출이면 기존 일정을 고치지 않고 새 일정 등록을 제안한다."""
        self.use()
        graph = self.graph(a_runner=FakeA(outing_result("2026-11-27T10:00:00+09:00", "2026-11-27T12:00:00+09:00",
                                                        purpose="공원 산책")))
        view = self.say(graph, "모레 공원 산책 가려는데")
        self.assertEqual(got(view, "pending")["type"], "create_schedule")
        self.assertIn("[새 일정] 공원 산책", view["pending"]["preview"])
        self.assertIn("새 일정으로 등록하는 것을 제안해요", view["answer"])

    def test_GR17_ambiguous_asks_then_uses_proposed_time(self):
        """GR17 시각이 겹치는 일정이 여럿이면 카드 없이 묻고, 사용자가 고르면 A가 제안한 시각으로 변경을 제안한다."""
        self.schedules.transact(uuid.uuid4().hex, "b" * 64, add_schedule("독서 모임", at(26, 14), 60))
        self.use(ScheduleLLM(schedule={"1번": {"operation": "update", "answers_previous_question": True}}))
        graph = self.graph(a_runner=FakeA(outing_result("2026-11-26T14:00:00+09:00", "2026-11-26T17:00:00+09:00",
                                                        purpose="친구 만나기")))
        view = self.say(graph, "내일 두 시에 나가는데")
        self.assertIsNone(view["pending"])
        self.assertEqual(view["active_feature"], "S")
        self.assertIn("어느 것을 바꿀지 확인이 필요해요", view["answer"])
        view = self.say(graph, "1번")
        action = self.values(graph)["pending_action"]
        self.assertEqual(action["payload"]["schedule_id"], SID["시장"])
        self.assertEqual(action["payload"]["after"]["start_at"], ss.iso(at(26, 17)))

    def test_GR18_refused_link_explains_without_card(self):
        """GR18 A가 제안한 시각이 다른 일정과 겹치면 승인 카드를 만들지 않고, 이유와 비어 있는 시간을 안내한다."""
        self.use()
        graph = self.graph(a_runner=FakeA(outing_result("2026-11-26T14:00:00+09:00", "2026-11-26T16:00:00+09:00")))
        before = self.raw()
        view = self.say(graph, "내일 시장 비 오면?")
        self.assertIsNone(view["pending"])
        self.assertIn("[일정 확인] 오후 4시에는 다른 일정(오후 4시 복지관 방문)이 있어요.", view["answer"])
        self.assertIn("비어 있는 시간: 오후 5시", view["answer"])
        self.assertEqual(self.raw(), before)


# %% L층. 실제 qwen2.5:3b + 명시된 가상 예보 + 임시 저장소로 대표 흐름 (요구 33, 보완 7)
# 핵심 요약: 정해진 횟수(3회)만 실행하고 결과를 고르지 않는다. 그래프 성공·모델 판단·제안·저장·안전(잘못된 저장 없음)을 따로 센다.
# 모델이 기대와 다르게 판단해도 이 테스트는 실패로 처리하지 않고 표로 보고한다. 잘못된 저장·예외만 assert로 실패 처리한다.
LIVE_FLOW_RUNS = 3
LIVE_FORECAST = {14: RAIN}      # 명시된 가상 예보: 14시 구간만 비 (실제 예보가 아님)
LIVE_LOG_DIR = ROOT / ".safemate" / "test_logs"
# 대표 시나리오의 기대 결과: 시장 장보기(14:00~15:00, 60분) → '오후 네 시쯤'은 복지관(16:00~17:00)과 충돌
# → 같은 날 가장 가까운 늦은 빈 시간 17:00~18:00 (소요 시간 60분 유지)
LIVE_TARGET_ID = "schedule_002"
LIVE_EXPECTED = {"start_at": "2026-11-26T17:00:00+09:00", "end_at": "2026-11-26T18:00:00+09:00"}
LIVE_EXPECTED_WORDS = "오후 5시~오후 6시"


# %% L층 판정 함수 (오프라인 단위 테스트 M-AG7로도 검증)
# 핵심 요약: '유효한 제안'과 '실제 저장 성공'을 일정 id만이 아니라 종류·대상·버전·변경 전 내용·기대 시각·소요 시간·
# 바꿀 수 없는 항목·이력·재실행 후 값까지 비교해 판정한다. 문제 목록이 비어 있어야 성공이다.
def judge_live_proposal(action: Any, original_doc: dict[str, Any]) -> list[str]:
    """대표 시나리오의 기대 제안(시장 장보기 → 17:00~18:00)과 다른 점 목록"""
    if not isinstance(action, dict) or action.get("type") != "update_schedule" or action.get("source") != sv.SOURCE:
        return ["카드 종류가 update_schedule(schedule_service)가 아님"]
    payload = action.get("payload") or {}
    item = next(s for s in original_doc["schedules"] if s["schedule_id"] == LIVE_TARGET_ID)
    problems = []
    if payload.get("schedule_id") != LIVE_TARGET_ID:
        problems.append(f"대상 {payload.get('schedule_id')}")
    if payload.get("expected_version") != item["version"] or payload.get("before") != sv.snapshot(item):
        problems.append("버전·변경 전 내용이 저장된 일정과 다름")
    after = payload.get("after") or {}
    if {k: after.get(k) for k in LIVE_EXPECTED} != LIVE_EXPECTED:
        problems.append(f"제안 시각 {after.get('start_at')}~{after.get('end_at')}")
    start, end = ss.parse_iso(after.get("start_at")), ss.parse_iso(after.get("end_at"))
    if start is None or end is None or end - start != sv.interval(item)[1] - sv.interval(item)[0]:
        problems.append("소요 시간이 원래와 다름")
    if {k: v for k, v in after.items() if k not in ("start_at", "end_at")} != \
            {k: v for k, v in sv.snapshot(item).items() if k not in ("start_at", "end_at")}:
        problems.append("시각 외 항목이 바뀜")
    return problems


def judge_live_saved(doc: dict[str, Any], reopened: dict[str, Any], original_doc: dict[str, Any]) -> list[str]:
    """승인 뒤 실제 JSON 저장 결과가 기대(시장만 17:00~18:00, 버전 2, 이력 1건, 재실행 후 같음)와 다른 점 목록"""
    problems = []
    item = next(s for s in doc["schedules"] if s["schedule_id"] == LIVE_TARGET_ID)
    if {k: item[k] for k in LIVE_EXPECTED} != LIVE_EXPECTED or item["version"] != 2 or item["status"] != "scheduled":
        problems.append(f"저장된 시장 일정 {item['start_at']}~{item['end_at']} v{item['version']} {item['status']}")
    others = [s for s in doc["schedules"] if s["schedule_id"] != LIVE_TARGET_ID]
    if others != [s for s in original_doc["schedules"] if s["schedule_id"] != LIVE_TARGET_ID]:
        problems.append("다른 일정이 바뀜")
    history = doc["action_history"]
    if len(history) != 1 or history[0]["operation"] != "update" or history[0]["schedule_id"] != LIVE_TARGET_ID \
            or history[0]["before"]["start_at"] != "2026-11-26T14:00:00+09:00" \
            or history[0]["after"]["start_at"] != LIVE_EXPECTED["start_at"]:
        problems.append(f"이력 {len(history)}건이 기대와 다름")
    if reopened["schedules"] != doc["schedules"] or reopened["action_history"] != history:
        problems.append("새 저장소 객체로 다시 읽은 값이 다름")
    return problems


def live_log_path() -> Path:
    """실측 로그 경로. 이미 있는 로그는 덮어쓰지 않고 시각을 붙인 새 이름을 쓴다. (이전 측정 기록 보존)"""
    path = Path(OPTIONS.live_log) if OPTIONS.live_log else LIVE_LOG_DIR / f"5-4_live_flow_{datetime.now(KST):%Y%m%d_%H%M%S}.txt"
    if path.exists():
        path = path.with_name(f"{path.stem}_{datetime.now(KST):%H%M%S}{path.suffix}")
    return path


class LiveLog:
    """진행 상황을 한 줄씩 바로 쓰는 UTF-8 로그 (연결이 끊겨도 어디까지 했는지 남음). 경로·개인정보는 쓰지 않는다."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("x", encoding="utf-8")     # 있는 파일은 열지 않는다 (덮어쓰기 방지)

    def write(self, text: str) -> None:
        self.file.write(text + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class CountingLLM:
    """실제 call_ollama를 그대로 부르고, 단계·걸린 시간만 기록한다. (응답은 바꾸지 않음)"""

    def __init__(self, real):
        self.real, self.calls = real, []

    def __call__(self, messages, schema, settings, *, stage, max_tokens=None, temperature=None):
        started = time.perf_counter()
        try:
            return self.real(messages, schema, settings, stage=stage, max_tokens=max_tokens, temperature=temperature)
        finally:
            self.calls.append({"stage": stage, "seconds": round(time.perf_counter() - started, 1)})


class L_AG_LiveFlow(unittest.TestCase):

    def setUp(self):
        if not OPTIONS.live_ollama:
            raise unittest.SkipTest("--live-ollama 를 지정하면 실행합니다. (ollama serve와 qwen2.5:3b 필요)")
        report = od.ollama_status(od.ollama_settings())
        if not (report["server_running"] and report["model_installed"]):
            raise unittest.SkipTest(f"Ollama 서버 또는 모델이 준비되지 않았습니다: {report}")

    def turn(self, log, graph, thread, llm, label, text):
        """한 턴을 실행하고 모델 판단·경로·답변 앞부분·LLM 호출 수·시간을 로그에 남긴다."""
        before, started = len(llm.calls), time.perf_counter()
        view = sg.send_message(graph, thread, text)
        routing = view["debug"].get("routing") or {}
        log.write(f"  [{label}] {text} | 3단계 의도={routing.get('model_intent')} 재확인={routing.get('recheck')} "
                  f"일정 작업={routing.get('schedule_operation')}/{routing.get('schedule_request_mode')} "
                  f"최종={routing.get('final_route')} 카드={'있음' if view['pending'] else '없음'} "
                  f"LLM {len(llm.calls) - before}회 {time.perf_counter() - started:.1f}초")
        log.write("      답변: " + view["answer"].replace("\n", " / ")[:240])
        return view, routing

    def run_flow(self, log, run: int, approve: bool) -> dict:
        """판정 항목 (서로 섞지 않음)
        - graph_ok: 예외 없이 끝남 / model_ok: 모델이 고른 의도·일정 작업이 단계마다 기대와 같음(경로 기록만 봄)
        - query_ok: 조회 답변에 내일 일정 2개 / proposal_ok: judge_live_proposal 통과(17:00~18:00, 60분, 대상·버전·변경 전 내용)
        - saved_ok: judge_live_saved 통과 / requery_ok: 재조회·재실행 후 답변이 17:00~18:00 / safe_ok: 잘못된 저장 없음"""
        row = {"run": run, "kind": "승인" if approve else "거절", "graph_ok": True, "model_ok": False, "query_ok": None,
               "proposal_ok": False, "saved_ok": None, "requery_ok": None, "safe_ok": False, "proposed_at": None, "notes": []}
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ, clean_env(), clear=True):
            path = Path(folder) / "schedule_store.json"
            store = ss.ScheduleStore(path, clock=lambda: FIXED_NOW)
            store.ensure_initialized()
            original_doc, original_raw = store.load(), path.read_bytes()
            llm = CountingLLM(od.call_ollama)
            with tod.Harness(llm=llm, weather=tod.FakeWeather(by_hour=LIVE_FORECAST), now=FIXED_NOW):
                graph = sg.build_graph(schedule_store=store, a_device="xpu", a_index=tod.FAKE_INDEX)
                thread = sg.new_thread_id()
                try:
                    model = []      # 모델 판단만: 경로 기록(3단계 의도·일정 작업)이 기대와 같은지

                    def schedule_routed(routing: dict[str, Any]) -> bool:
                        return routing.get("model_intent") in sg.SCHEDULE_ROUTES or routing.get("schedule_mode") == "continue"

                    # 1. 조회 (승인 흐름만): 모델 판단과 답변 내용을 따로 본다.
                    if approve:
                        view, routing = self.turn(log, graph, thread, llm, "조회", "내가 내일 뭐 하기로 했지?")
                        model.append(schedule_routed(routing) and routing.get("schedule_operation") == "query")
                        row["query_ok"] = "시장 장보기 [예정]" in view["answer"] and "복지관 방문 [예정]" in view["answer"]
                    # 2. 변경 요청 → 시각 질문
                    view, routing = self.turn(log, graph, thread, llm, "변경 요청", "시장 가는 시간 좀 바꿔 줘")
                    model.append(schedule_routed(routing) and routing.get("schedule_operation") == "update")
                    if not (view["active_feature"] == "S" and "몇 시로" in view["answer"]):
                        row["notes"].append("변경 요청에 시각 질문이 나오지 않음")
                    # 3. '오후 네 시쯤' → 충돌 → 대체 제안. 제안 내용은 judge_live_proposal로 정확히 비교한다.
                    view, routing = self.turn(log, graph, thread, llm, "새 시각", "오후 네 시쯤")
                    model.append(routing.get("schedule_operation") == "update")
                    action = graph.get_state(sg.run_config(thread)).values.get("pending_action")
                    problems = judge_live_proposal(action, original_doc)
                    row["proposal_ok"] = not problems
                    if isinstance(action, dict):
                        row["proposed_at"] = ((action.get("payload") or {}).get("after") or {}).get("start_at")
                    row["notes"] += [f"제안: {p}" for p in problems]
                    # 4. 승인 또는 거절 (기대와 다른 제안은 승인하지 않는다)
                    if row["proposal_ok"] and isinstance(action, dict):
                        decision = "approve" if approve else "reject"
                        result = sg.press_button(graph, thread, decision, action["action_id"])
                        log.write(f"  [{decision}] " + result["answer"].replace("\n", " / ")[:200])
                    doc = store.load()
                    if approve and row["proposal_ok"]:
                        # 5. 다시 묻기 + 6. 새 저장소·새 그래프에서 확인 (프로그램 재실행 흉내)
                        view, routing = self.turn(log, graph, thread, llm, "재조회", "시장 몇 시였지?")
                        model.append(schedule_routed(routing) and routing.get("schedule_operation") == "query")
                        reopened = ss.ScheduleStore(path, clock=lambda: FIXED_NOW)
                        fresh = sg.build_graph(schedule_store=reopened, a_device="xpu", a_index=tod.FAKE_INDEX)
                        again, routing = self.turn(log, fresh, sg.new_thread_id(), llm, "재실행 후 조회", "시장 몇 시였지?")
                        model.append(schedule_routed(routing) and routing.get("schedule_operation") == "query")
                        saved = judge_live_saved(doc, reopened.load(), original_doc)
                        row["saved_ok"] = not saved
                        row["notes"] += [f"저장: {p}" for p in saved]
                        row["requery_ok"] = LIVE_EXPECTED_WORDS in view["answer"] and LIVE_EXPECTED_WORDS in again["answer"]
                        row["safe_ok"] = not saved      # 승인한 기대 변경 외에는 아무것도 바뀌지 않음
                    else:
                        row["safe_ok"] = path.read_bytes() == original_raw
                    row["model_ok"] = all(model)
                except Exception as exc:  # noqa: BLE001 - 그래프 예외는 실패로 기록하고 아래 assert에서 실패 처리
                    row["graph_ok"] = False
                    row["notes"].append(f"{type(exc).__name__}: {exc}"[:200])
            row["llm_calls"], row["llm_seconds"] = len(llm.calls), round(sum(c["seconds"] for c in llm.calls), 1)
        return row

    def test_L_AG01_change_flow_with_real_qwen(self):
        """L-AG01 실제 Qwen: 조회 → 변경 요청 → 시각 질문 → 충돌 → 대체 제안 → 승인 → JSON 수정 → 재조회 → 재실행 확인 (3회) + 거절 (3회)"""
        path = live_log_path()
        log = LiveLog(path)
        self.addCleanup(log.close)
        print(f"\n  [L-AG01] 진행 로그: .safemate/test_logs/{path.name}" if path.parent == LIVE_LOG_DIR else "")
        log.write(f"[L-AG01] 5-4단계 실제 Ollama 대표 흐름 | 모델 {od.ollama_settings()['model']} | 기준 시각 {FIXED_NOW:%Y-%m-%d %H:%M} "
                  f"(고정) | 가상 예보: 14시 구간 비(강수 2mm·80%), 그 밖 맑음 | 임시 저장소 | 반복 {LIVE_FLOW_RUNS}회")
        rows = []
        for run in range(1, LIVE_FLOW_RUNS + 1):
            for approve in (True, False):
                log.write(f"[회차 {run} · {'승인' if approve else '거절'}] 시작 {datetime.now(KST):%H:%M:%S}")
                row = self.run_flow(log, run, approve)
                rows.append(row)
                log.write(f"[회차 {run} · {row['kind']}] 그래프 {row['graph_ok']} | 모델 판단 {row['model_ok']} | 조회 답변 {row['query_ok']} | 제안 {row['proposal_ok']}"
                          f"({row['proposed_at']}) | 저장 {row['saved_ok']} | 재조회 {row['requery_ok']} | 안전 {row['safe_ok']} | LLM {row['llm_calls']}회 "
                          f"{row['llm_seconds']}초 | {'; '.join(row['notes'])}")
        total = len(rows)
        approves = [r for r in rows if r["kind"] == "승인"]
        summary = (f"[요약] 실행 {total}회(승인 {len(approves)}·거절 {total - len(approves)}) | 그래프 성공 {sum(r['graph_ok'] for r in rows)} | "
                   f"모델 판단 성공 {sum(r['model_ok'] for r in rows)} | 조회 답변 성공 {sum(bool(r['query_ok']) for r in approves)}/{len(approves)} | "
                   f"유효 제안(17:00~18:00·60분·대상 일치) {sum(r['proposal_ok'] for r in rows)} | "
                   f"실제 저장 성공 {sum(bool(r['saved_ok']) for r in approves)}/{len(approves)} | "
                   f"재조회·재실행 답변 성공 {sum(bool(r['requery_ok']) for r in approves)}/{len(approves)} | "
                   f"잘못된 저장 없음 {sum(r['safe_ok'] for r in rows)}")
        log.write(summary)
        print("\n  " + summary)
        for r in rows:
            print(f"  회차 {r['run']} {r['kind']}: 그래프 {r['graph_ok']} 모델 {r['model_ok']} 조회 {r['query_ok']} 제안 {r['proposal_ok']}({r['proposed_at']}) "
                  f"저장 {r['saved_ok']} 재조회 {r['requery_ok']} 안전 {r['safe_ok']} LLM {r['llm_calls']}회 {r['llm_seconds']}초 {'; '.join(r['notes'])}")
        # 프로그램 계약: 예외 없이 끝나고, 어떤 경우에도 잘못된 저장이 없어야 한다. (모델 판단 실패는 표로만 보고)
        self.assertTrue(all(r["graph_ok"] for r in rows), [r["notes"] for r in rows])
        self.assertTrue(all(r["safe_ok"] for r in rows))


# %% R층. 기존 파일 보존과 운영 저장소 격리 (요구 31·34)
class R_Regression(unittest.TestCase):

    def test_R1_stage_1_2_files_match_base_commit(self):
        """R1 1·2단계 운영 파일과 3단계 테스트 파일이 기준 커밋과 같다. (줄바꿈 차이는 같게 봄)"""
        for name in PROTECTED_FILES:
            with self.subTest(name):
                try:
                    base = subprocess.run(["git", "show", f"{BASE_COMMIT}:{name}"], cwd=ROOT, capture_output=True,
                                          check=True).stdout
                except (OSError, subprocess.CalledProcessError) as exc:
                    raise unittest.SkipTest(f"git으로 기준 파일을 읽지 못했습니다: {exc}")
                current = (ROOT / name).read_bytes()

                def normalize(data: bytes) -> str:
                    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()

                self.assertEqual(normalize(current), normalize(base))

    def test_R2_operational_store_untouched(self):
        """R2 테스트가 운영 일정 파일(.safemate/schedule_store.json)을 만들거나 바꾸지 않았다."""
        after = {str(p): file_state(p) for p in OPERATIONAL_FILES}
        self.assertEqual(after, OPERATIONAL_BEFORE)


# %% 명령행 옵션 읽기와 실행
# 핵심 요약: 이 파일 전용 옵션(--device, --live-ollama, --live-log)을 먼저 읽고, 나머지는 unittest에 넘긴다.
def main():
    """unittest가 모르는 옵션을 꺼내 OPTIONS에 담은 뒤 테스트를 실행한다."""
    argv = sys.argv[1:]
    rest = []
    while argv:
        item = argv.pop(0)
        if item == "--device":
            OPTIONS.device = argv.pop(0)
        elif item == "--live-ollama":
            OPTIONS.live_ollama = True
        elif item == "--live-log":
            OPTIONS.live_log = argv.pop(0)
        else:
            rest.append(item)
    if OPTIONS.device not in (None, "auto", "xpu", "cuda", "cpu"):
        raise SystemExit("--device 는 auto, xpu, cuda, cpu 중 하나입니다.")
    unittest.main(argv=[sys.argv[0]] + rest)


if __name__ == "__main__":
    main()
