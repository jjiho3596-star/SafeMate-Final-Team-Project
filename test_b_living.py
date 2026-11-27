import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import b_living


class LivingActionTests(unittest.TestCase):
    def make_action(self):
        items = [{"name": "우유", "quantity": 1, "unit": "개"}]
        note = "문 앞에 두어 주세요."
        return {
            "type": "request_caregiver_help",
            "action_id": "3f2a9c0e5b7d4e1f8a6b2c9d0e1f2a3b",
            "summary": "보호자께 장보기 도움을 요청합니다.",
            "requires_approval": True,
            "applied": False,
            "created_at": "2026-11-27T18:00:00+09:00",
            "source": "living_support",
            "payload": {
                "recipient_role": "caregiver",
                "category": "groceries",
                "items": items,
                "note": note,
                "urgency": "normal",
                "message": b_living.build_help_message("groceries", items, note),
            },
        }

    def assert_result_keys(self, result):
        self.assertEqual(
            set(result),
            {"status", "message", "action_id", "applied", "mock", "record", "answer"},
        )
        self.assertIs(result["mock"], True)

    def test_build_help_message_uses_only_confirmed_details(self):
        message = b_living.build_help_message(
            "groceries",
            [{"name": "우유", "quantity": 1, "unit": "개"}],
            "문 앞에 두어 주세요.",
        )
        self.assertIn("장보기", message)
        self.assertIn("우유 1개", message)
        self.assertIn("참고: 문 앞에 두어 주세요.", message)
        with self.assertRaisesRegex(ValueError, "item_quantity_missing"):
            b_living.build_help_message(
                "groceries", [{"name": "우유", "quantity": None, "unit": ""}]
            )
        with self.assertRaisesRegex(ValueError, "category_invalid"):
            b_living.build_help_message([], [])

    def test_validation_approval_idempotency_and_jsonl_record(self):
        action = self.make_action()
        action["action_id"] = "living-request-1"
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "mock_action_log.jsonl"
            with patch.dict(os.environ, {"SAFEMATE_ACTION_LOG": str(log_path)}):
                invalid = b_living.execute_living_action({**action, "applied": True})
                self.assert_result_keys(invalid)
                self.assertEqual(invalid["status"], "invalid_action")

                rejected = b_living.execute_living_action(action)
                self.assert_result_keys(rejected)
                self.assertEqual(rejected["status"], "rejected")

                action["approval"] = {
                    "status": "approved",
                    "at": "2026-11-27T18:01:00+09:00",
                }
                executed = b_living.execute_living_action(action)
                self.assert_result_keys(executed)
                self.assertEqual(executed["status"], "ok")
                self.assertIs(executed["applied"], True)
                self.assertEqual(
                    executed["record"]["payload"]["message"],
                    action["payload"]["message"],
                )

                repeated = b_living.execute_living_action(action)
                self.assert_result_keys(repeated)
                self.assertEqual(repeated["status"], "already_executed")
                self.assertEqual(
                    len(log_path.read_text(encoding="utf-8").splitlines()), 1
                )
                json.loads(log_path.read_text(encoding="utf-8"))

    def test_log_write_failure_returns_error_shape(self):
        action = self.make_action()
        action["approval"] = {
            "status": "approved",
            "at": "2026-11-27T18:01:00+09:00",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            log_directory = Path(temp_dir) / "not_a_file.jsonl"
            log_directory.mkdir()
            with patch.dict(
                os.environ, {"SAFEMATE_ACTION_LOG": str(log_directory)}
            ):
                result = b_living.execute_living_action(action)
        self.assert_result_keys(result)
        self.assertEqual(result["status"], "error")
        self.assertIs(result["applied"], False)
        self.assertIsNone(result["record"])


if __name__ == "__main__":
    unittest.main()
