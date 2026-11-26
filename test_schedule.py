"""SafeMate 5단계 일정 관리 테스트

실행 방법 (.venv-xpu)
- 기본 (M층 단위·I층 통합·R층 회귀, 네트워크·Ollama 불필요):
    & .\\.venv-xpu\\Scripts\\python.exe test_schedule.py -v
- 실제 연동 (L층, 이후 구현 단위에서 추가):
    & .\\.venv-xpu\\Scripts\\python.exe test_schedule.py --live-ollama --device auto -v

테스트 층
- M층: 일정 저장소·서비스·장치 선택의 단위 테스트 (가상 시각, 임시 폴더)
- I층: 실제 코드 통합 + 가상 LLM·가상 예보 (이후 구현 단위에서 추가)
- L층: 실제 qwen2.5:3b·Open-Meteo (이후 구현 단위에서 추가)
- R층: 1·2단계 파일 보존, 운영 일정 파일을 테스트가 건드리지 않았는지 확인

모든 테스트는 TemporaryDirectory 안의 임시 저장소만 쓴다. 운영 저장소(.safemate/schedule_store.json)는 읽거나 쓰지 않는다.
"""

# %% 공통 라이브러리와 검사 대상 불러오기
# 핵심 요약: 테스트 대상 모듈과 표준 라이브러리만 쓴다. 테스트 시작 시점의 운영 저장소 상태를 기록해 두었다가 끝에 비교한다.
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import schedule_service as sv
import schedule_store as ss
from a_outing import KST

ROOT = Path(__file__).resolve().parent
OPTIONS = SimpleNamespace(device=None, live_ollama=False)
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
# 핵심 요약: 이 파일 전용 옵션(--device, --live-ollama)을 먼저 읽고, 나머지는 unittest에 넘긴다.
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
        else:
            rest.append(item)
    if OPTIONS.device not in (None, "auto", "xpu", "cuda", "cpu"):
        raise SystemExit("--device 는 auto, xpu, cuda, cpu 중 하나입니다.")
    unittest.main(argv=[sys.argv[0]] + rest)


if __name__ == "__main__":
    main()
