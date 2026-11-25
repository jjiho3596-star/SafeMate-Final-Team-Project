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
