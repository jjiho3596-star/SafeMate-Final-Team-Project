import json
import os
import tempfile
import unittest
from pathlib import Path
from urllib.error import URLError
from unittest.mock import patch

import b_living


def check_living_result(result):
    required = {
        "status",
        "detail_code",
        "message",
        "route",
        "answer",
        "conversation_state",
        "missing_fields",
        "clarification",
        "requires_user_confirmation",
        "proposed_action",
        "usage",
        "tool_trace",
        "limitations",
        "living_evidence",
    }
    assert required <= set(result), required - set(result)
    assert "approval" not in result
    assert result["status"] in {
        "ok",
        "needs_input",
        "llm_unavailable",
        "llm_invalid_output",
        "tool_error",
    }
    assert result["route"] in {"living_support", "needs_clarification", "error"}
    assert isinstance(result["answer"], str) and result["answer"].strip()
    assert isinstance(result["conversation_state"], dict)
    assert isinstance(result["missing_fields"], list)
    assert isinstance(result["living_evidence"], list)
    assert (result["status"] == "needs_input") == (
        result["route"] == "needs_clarification"
    )
    assert bool(result["missing_fields"]) == (result["clarification"] is not None)
    if result["clarification"] is not None:
        assert result["clarification"]["fields"] == result["missing_fields"]
    for evidence in result["living_evidence"]:
        assert isinstance(evidence["content"], str) and evidence["content"].strip()
        assert {"title", "publisher", "url", "pdf_page"} <= set(evidence)
        assert evidence["pdf_page"] is None or (
            isinstance(evidence["pdf_page"], int)
            and not isinstance(evidence["pdf_page"], bool)
        )

    action = result["proposed_action"]
    assert result["requires_user_confirmation"] == (action is not None)
    if action is not None:
        assert action["type"] == "request_caregiver_help"
        assert isinstance(action["action_id"], str) and action["action_id"]
        assert action["requires_approval"] is True and action["applied"] is False
        assert result["status"] == "ok" and result["route"] == "living_support"
        assert "approval" not in action
        payload = action["payload"]
        assert payload["recipient_role"] == "caregiver"
        assert payload["items"]
        assert isinstance(payload["message"], str)
        assert 0 < len(payload["message"]) <= 300
        for item in payload["items"]:
            assert item["name"] in payload["message"]
    json.dumps(result, ensure_ascii=False, allow_nan=False)


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


class LivingSupportTests(unittest.TestCase):
    evidence = [
        {
            "content": "생활지원 안내 자료 원문",
            "title": "생활지원 안내",
            "publisher": "공공기관",
            "url": None,
            "pdf_page": None,
        }
    ]

    def model_output(self, **overrides):
        response = {
            "intent": "living_support",
            "category": "other",
            "items": [],
            "urgency": "normal",
            "answer": "자료에서 확인한 생활지원 안내를 참고해 주세요.",
        }
        response.update(overrides)
        return json.dumps(response, ensure_ascii=False)

    def test_normal_consultation_and_no_evidence_limitation(self):
        with (
            patch.object(
                b_living, "_search_living_evidence", return_value=self.evidence
            ),
            patch.object(
                b_living, "_ollama_chat", return_value=self.model_output()
            ),
        ):
            result = b_living.run_living_support(
                {"user_message": "생활지원 안내를 알려 주세요", "context": None},
                device="cpu",
            )
        check_living_result(result)
        self.assertEqual(result["living_evidence"], self.evidence)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["route"], "living_support")

        with (
            patch.object(b_living, "_search_living_evidence", return_value=[]),
            patch.object(
                b_living, "_ollama_chat", return_value=self.model_output()
            ),
        ):
            no_evidence = b_living.run_living_support("생활지원 안내")
        check_living_result(no_evidence)
        self.assertEqual(no_evidence["living_evidence"], [])
        self.assertTrue(no_evidence["limitations"])
        self.assertIn("공식 생활지원 자료", no_evidence["answer"])

    def test_missing_quantity_follow_up_becomes_approved_action_proposal(self):
        first_output = self.model_output(
            intent="request_caregiver_help",
            category="groceries",
            items=[{"name": "우유", "quantity": None, "unit": ""}],
            answer="우유 수량을 확인할게요.",
        )
        follow_up_output = self.model_output(
            intent="living_support",
            category="groceries",
            items=[{"name": "우유", "quantity": 2, "unit": "개"}],
            answer="요청 내용을 정리했어요.",
        )
        with (
            patch.object(
                b_living, "_search_living_evidence", return_value=self.evidence
            ),
            patch.object(
                b_living,
                "_ollama_chat",
                side_effect=[first_output, follow_up_output],
            ),
            patch.object(b_living, "execute_living_action") as execute,
        ):
            needs_input = b_living.run_living_support("우유가 필요해")
            check_living_result(needs_input)
            self.assertEqual(needs_input["status"], "needs_input")
            self.assertEqual(needs_input["missing_fields"], ["quantity"])
            self.assertIsNone(needs_input["proposed_action"])

            proposed = b_living.run_living_support(
                "두 개",
                conversation_state=needs_input["conversation_state"],
            )
        check_living_result(proposed)
        self.assertEqual(proposed["status"], "ok")
        action = proposed["proposed_action"]
        self.assertEqual(action["type"], "request_caregiver_help")
        self.assertEqual(action["payload"]["items"][0]["quantity"], 2)
        self.assertEqual(
            action["payload"]["message"],
            b_living.build_help_message(
                "groceries", action["payload"]["items"], action["payload"]["note"]
            ),
        )
        execute.assert_not_called()

    def test_search_and_ollama_errors_obey_result_contract(self):
        previous_state = {
            "turn": 2,
            "pending_category": "groceries",
            "pending_items": [{"name": "우유", "quantity": None, "unit": ""}],
        }
        with patch.object(
            b_living,
            "_search_living_evidence",
            side_effect=OSError("index unavailable"),
        ):
            search_error = b_living.run_living_support(
                "두 개", conversation_state=previous_state
            )
        check_living_result(search_error)
        self.assertEqual(search_error["status"], "tool_error")
        self.assertEqual(search_error["route"], "error")
        self.assertEqual(search_error["conversation_state"], previous_state)
        self.assertIsNone(search_error["proposed_action"])

        with (
            patch.object(
                b_living, "_search_living_evidence", return_value=self.evidence
            ),
            patch.object(
                b_living, "_ollama_chat", side_effect=URLError("Ollama offline")
            ),
        ):
            unavailable = b_living.run_living_support(
                "두 개", conversation_state=previous_state
            )
        check_living_result(unavailable)
        self.assertEqual(unavailable["status"], "llm_unavailable")
        self.assertEqual(unavailable["conversation_state"], previous_state)

        with (
            patch.object(
                b_living, "_search_living_evidence", return_value=self.evidence
            ),
            patch.object(b_living, "_ollama_chat", return_value="{invalid json"),
        ):
            invalid_output = b_living.run_living_support(
                {"user_message": "안내를 알려 주세요", "context": {}},
                conversation_state=previous_state,
            )
        check_living_result(invalid_output)
        self.assertEqual(invalid_output["status"], "llm_invalid_output")
        self.assertEqual(invalid_output["conversation_state"], previous_state)

    def test_malformed_ollama_envelopes_and_model_fields_are_reported(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"message":{"content":[]}}'

        with (
            patch.dict(
                os.environ, {"OLLAMA_BASE_URL": "http://127.0.0.1:11434"}
            ),
            patch.object(
                b_living, "_search_living_evidence", return_value=self.evidence
            ),
            patch.object(b_living, "urlopen", return_value=FakeResponse()),
        ):
            bad_envelope = b_living.run_living_support("생활지원 안내")
        check_living_result(bad_envelope)
        self.assertEqual(bad_envelope["status"], "llm_invalid_output")

        malformed_fields = self.model_output(intent=[])
        with (
            patch.object(
                b_living, "_search_living_evidence", return_value=self.evidence
            ),
            patch.object(b_living, "_ollama_chat", return_value=malformed_fields),
        ):
            bad_model_data = b_living.run_living_support("생활지원 안내")
        check_living_result(bad_model_data)
        self.assertEqual(bad_model_data["status"], "llm_invalid_output")

    def test_ollama_environment_precedence_does_not_mutate_environment(self):
        with patch.dict(
            os.environ,
            {
                "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
                "B_OLLAMA_MODEL": "living-model",
                "OLLAMA_MODEL": "shared-model",
            },
        ):
            before = dict(os.environ)
            self.assertEqual(
                b_living._local_ollama_config(),
                ("http://127.0.0.1:11434", "living-model"),
            )
            self.assertEqual(dict(os.environ), before)


if __name__ == "__main__":
    unittest.main()
