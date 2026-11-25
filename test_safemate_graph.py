"""SafeMate 4단계 공통 LangGraph 테스트

실행 방법 (.venv-xpu)
- 기본 (M층 + I층 + R층, 네트워크·Ollama 불필요):
    & .\\.venv-xpu\\Scripts\\python.exe test_safemate_graph.py -v
- 실제 연동 (L층, Ollama qwen2.5:3b 실행 중 + 인터넷 필요):
    & .\\.venv-xpu\\Scripts\\python.exe test_safemate_graph.py --live-ollama --device xpu -v

테스트 층
- M층 (G1~G13): 가짜 A·B·실행 함수로 그래프 흐름·승인 안전성만 확인한다.
- I층 (I1~I3): 실제 run_outing_decision() + 3단계 테스트의 가상 예보·가상 LLM(Harness)으로 확인한다.
- L층 (L1~L3): 실제 qwen2.5:3b, 실제 Open-Meteo로 확인한다. --live-ollama 를 줄 때만 실행한다.
- R층 (R1): 1·2·3단계 파일이 기준 커밋(c4e2ca0)과 같은지 SHA-256으로 확인한다.
  기존 147개 회귀 테스트는 test_outing_decision.py 를 따로 실행한다.
"""

# %% 공통 라이브러리와 검사 대상 불러오기
# 핵심 요약: 그래프 모듈과, I층에서 다시 쓸 3단계 테스트의 가상 함수(Harness·FakeLLM·FakeWeather)를 불러온다. (읽기만 함)
import copy
import hashlib
import json
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import outing_decision as od
import safemate_graph as sg
import test_outing_decision as tod

ROOT = Path(__file__).resolve().parent
OPTIONS = SimpleNamespace(device=None, live_ollama=False)
BASE_COMMIT = "c4e2ca0"
PROTECTED_FILES = ("a_outing.py", "safety_rag.py", "outing_decision.py", "test_outing_decision.py")


# %% 가상 A 결과 (run_outing_decision 반환 계약의 필요한 부분)
# 핵심 요약: 그래프가 읽는 키를 계약(명세 1.3)대로 갖춘 A 결과를 만든다. 테스트마다 바꿀 값만 덮어쓴다.
def a_state(turn, marker="A"):
    """A 전용 대화 상태처럼 보이는 딕셔너리 (marker로 A·B 섞임을 검사)"""
    return {"version": 1, "turn": turn, "owner": marker, "pending_question": None}


def a_result(turn=1, **over):
    """정상 답변 턴의 A 결과"""
    result = {"status": "ok", "detail_code": None, "message": "답변을 생성했습니다.", "route": "weather_question",
              "intent": "weather_question", "answer": f"A 답변 {turn}", "conversation_state": a_state(turn),
              "missing_fields": [], "clarification": None, "requires_user_confirmation": False, "proposed_action": None,
              "official_guidance": [], "safety_evidence": [], "llm": {"understanding": {"user_goal": ""}},
              "decision": {"answer_source": "llm", "verified": True}, "tool_trace": [], "usage": {"llm_calls": 2},
              "limitations": []}
    result.update(over)
    return result


def a_question(turn=1):
    """지역만 부족해 Python 질문을 한 턴"""
    return a_result(turn, status="needs_input", detail_code="missing_fields", route="needs_clarification",
                    answer="어느 지역인지 알려 주세요.", missing_fields=["city"],
                    clarification={"fields": ["city"], "question": "어느 지역인지 알려 주세요.", "reasons": {}},
                    decision={"answer_source": "python_question", "verified": None})


def outing_proposal(original="2026-11-25T14:00+09:00", proposed="2026-11-25T16:00+09:00"):
    """A가 만드는 유일한 제안 (공통 봉투 키 없음)"""
    return {"type": "propose_outing_time_change", "original_outing_at": original, "proposed_outing_at": proposed,
            "reason": "두 예보의 차이입니다. 한 시간 강수량 2 → 0mm", "requires_approval": True, "applied": False}


def a_proposal_result(turn=1, **proposal):
    """일정 변경 제안이 있는 A 결과 (공식 안내 한 줄 포함)"""
    return a_result(turn, route="schedule_change", intent="schedule_change", answer="오후 4시가 더 나아 보여요.",
                    requires_user_confirmation=True, proposed_action=outing_proposal(**proposal),
                    official_guidance=[{"evidence_id": "E1", "text": "하천변·산길 이동 피하기",
                                        "publisher": "경기도 자연재난과", "title": "호우", "pdf_page": 1, "url": None}])


class FakeOuting:
    """run_outing_decision 대체. 결과를 순서대로 돌려주고, 받은 인자를 사본으로 기록한다."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, request, *, device=None, index=None, conversation_state=None):
        self.calls.append({"request": request, "device": device, "index": index,
                           "conversation_state": copy.deepcopy(conversation_state)})
        item = self.results.pop(0) if self.results else a_result(len(self.calls))
        if isinstance(item, Exception):
            raise item
        return item


# %% 가상 B 결과 (명세 2.2 run_living_support 계약)
# 핵심 요약: B 코드가 아직 없으므로 명세의 필수 키를 모두 갖춘 가짜 결과를 만든다. 명세 2.5 계약 검사로 Fake 자체도 확인한다.
def b_state(turn):
    """B 전용 대화 상태"""
    return {"b_version": 1, "turn": turn, "owner": "B"}


def b_result(turn=1, **over):
    """정상 상담 턴의 B 결과"""
    result = {"status": "ok", "detail_code": None, "message": "", "route": "living_support", "answer": f"B 답변 {turn}",
              "conversation_state": b_state(turn), "missing_fields": [], "clarification": None,
              "requires_user_confirmation": False, "proposed_action": None, "usage": {"llm_calls": 1},
              "tool_trace": [], "limitations": [], "living_evidence": []}
    result.update(over)
    return result


def b_question(turn=1):
    """물품 수량을 묻는 B 턴"""
    return b_result(turn, status="needs_input", detail_code="missing_items", route="needs_clarification",
                    answer="우유와 계란을 각각 얼마나 사면 될까요?", missing_fields=["quantity"],
                    clarification={"fields": ["quantity"], "question": "우유와 계란을 각각 얼마나 사면 될까요?",
                                   "reasons": {}})


def caregiver_action(**over):
    """B의 request_caregiver_help 공통 봉투 (명세 2.4 예시)"""
    action = {"type": "request_caregiver_help", "action_id": uuid.uuid4().hex,
              "summary": "보호자께 장보기 도움(우유 1개, 계란 1판)을 요청합니다.",
              "requires_approval": True, "applied": False, "created_at": "2026-11-25T15:00+09:00",
              "source": "living_support",
              "payload": {"recipient_role": "caregiver", "category": "groceries",
                          "items": [{"name": "우유", "quantity": 1, "unit": "개"},
                                    {"name": "계란", "quantity": 1, "unit": "판"}],
                          "note": "", "urgency": "normal",
                          "message": "어르신께서 장보기에 어려움을 겪고 계십니다. 우유 1개와 계란 1판을 구매하는 데 도움을 부탁드립니다."}}
    action.update(over)
    return action


def b_proposal_result(turn=1, **over):
    """보호자 도움 요청 제안이 있는 B 결과"""
    return b_result(turn, answer="보호자께 도움을 요청할까요?", requires_user_confirmation=True,
                    proposed_action=caregiver_action(**over))


def check_living_result(result: dict) -> None:
    """명세 2.5 계약 검사 코드 (그대로 옮김). 가짜 B 결과가 계약을 지키는지 확인한다."""
    required = {"status", "detail_code", "message", "route", "answer", "conversation_state", "missing_fields",
                "clarification", "requires_user_confirmation", "proposed_action", "usage", "tool_trace", "limitations",
                "living_evidence"}
    assert required <= set(result), required - set(result)
    assert result["status"] in {"ok", "needs_input", "llm_unavailable", "llm_invalid_output", "tool_error"}
    assert result["route"] in {"living_support", "needs_clarification", "error"}
    assert isinstance(result["answer"], str) and result["answer"].strip()
    assert isinstance(result["conversation_state"], dict)
    assert (result["status"] == "needs_input") == (result["route"] == "needs_clarification")
    assert bool(result["missing_fields"]) == (result["clarification"] is not None)
    if result["clarification"]:
        assert result["clarification"]["fields"] == result["missing_fields"]
    for evidence in result["living_evidence"]:
        assert isinstance(evidence["content"], str) and evidence["content"].strip()
        assert {"title", "publisher", "url", "pdf_page"} <= set(evidence)
        assert evidence["pdf_page"] is None or isinstance(evidence["pdf_page"], int)
    action = result["proposed_action"]
    assert result["requires_user_confirmation"] == (action is not None)
    if action:
        assert action["type"] == "request_caregiver_help" and action["action_id"]
        assert action["requires_approval"] is True and action["applied"] is False
        assert result["status"] == "ok" and "approval" not in action
        payload = action["payload"]
        assert payload["items"] and isinstance(payload["message"], str) and 0 < len(payload["message"]) <= 300
        for item in payload["items"]:
            assert item["name"] in payload["message"]
    json.dumps(result, ensure_ascii=False, allow_nan=False)


class FakeLiving:
    """run_living_support 대체 (명세 2.2 호출 모양)"""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, request, *, device=None, index=None, conversation_state=None):
        self.calls.append({"request": copy.deepcopy(request), "device": device, "index": index,
                           "conversation_state": copy.deepcopy(conversation_state)})
        return self.results.pop(0) if self.results else b_result(len(self.calls))


# %% 가상 실행 함수 (5단계 apply_outing_change, B execute_living_action 대체)
# 핵심 요약: 승인된 봉투를 받아 기록만 한다. mode로 성공·실패·예외·잘못된 반환을 흉내 낸다. action_id 멱등성도 흉내 낸다.
class FakeExecutor:
    """mode: ok, raise, error, not_applied, wrong_id, not_dict"""

    def __init__(self, mode="ok"):
        self.mode = mode
        self.calls = []
        self.done = set()

    def __call__(self, action):
        self.calls.append(copy.deepcopy(action))
        if self.mode == "raise":
            raise RuntimeError("가상 실행 오류")
        if self.mode == "not_dict":
            return "완료"
        action_id = action["action_id"]
        status = "already_executed" if action_id in self.done else "ok"
        output = {"status": status, "message": "가상 기록", "action_id": action_id, "applied": True, "mock": True,
                  "record": {"action_id": action_id}, "answer": "모의 실행으로 기록했어요. (실제로 보내지 않았어요)"}
        if self.mode == "error":
            output.update(status="error", applied=False, record=None)
        elif self.mode == "not_applied":
            output.update(applied=False)
        elif self.mode == "wrong_id":
            output.update(action_id="다른id")
        else:
            self.done.add(action_id)
        return output


# %% 테스트 도우미
# 핵심 요약: 가짜 함수로 그래프를 만들고, 앱 함수(send_message·press_button)로 한 사용자의 대화를 진행한다.
def make(a=None, b=None, a_exec=None, b_exec=None, **kwargs):
    """가짜 함수를 주입한 그래프. 넘기지 않은 B·실행 함수는 운영 기본값(None)이다."""
    return sg.build_graph(a_runner=a if a is not None else FakeOuting(), b_runner=b,
                          a_executor=a_exec, b_executor=b_exec, **kwargs)


def values(graph, thread):
    """저장된 State 값"""
    return graph.get_state(sg.run_config(thread)).values


def events(graph, thread):
    """action_log의 event 이름 목록"""
    return [e["event"] for e in values(graph, thread).get("action_log", [])]


class GraphCase(unittest.TestCase):
    """새 thread_id를 하나 준비한다."""

    def setUp(self):
        self.thread = sg.new_thread_id()

    def assert_no_success_claim(self, text):
        """실행이 끝난 것처럼 말하지 않는지 확인"""
        for word in ("요청했", "기록했", "바꿨", "변경했", "처리했"):
            self.assertNotIn(word, text)


# %% G1. 그래프 구조
class G1_Structure(GraphCase):

    def test_G1_nodes_edges_and_checkpointer(self):
        """G1 노드 6개, 되돌아가는 approval 엣지, 기본 체크포인터 InMemorySaver"""
        graph = make()
        nodes = set(graph.get_graph().nodes) - {"__start__", "__end__"}
        self.assertEqual(nodes, {"intake", "a_node", "b_node", "approval", "execute", "close"})
        mermaid = graph.get_graph().draw_mermaid()
        for edge in ("__start__ --> intake", "approval -.-> execute", "approval -.-> close", "approval -.-> approval",
                     "a_node -.-> b_node", "execute --> __end__"):
            self.assertIn(edge, mermaid)
        self.assertIsInstance(graph.checkpointer, InMemorySaver)
        self.assertEqual(sg.run_config("t"), {"configurable": {"thread_id": "t"}, "recursion_limit": 12})


# %% G2. A기능 기본 흐름
class G2_OutingBasics(GraphCase):

    def test_G2a_normal_answer_ends_without_approval(self):
        """G2a 일반 답변은 바로 끝나고 승인 카드가 없다. A 상태가 저장된다."""
        fake = FakeOuting(a_result(1))
        graph = make(fake)
        view = sg.send_message(graph, self.thread, "내일 인천 날씨 어때?")
        self.assertEqual((view["answer"], view["pending"], view["warning"]), ("A 답변 1", None, False))
        self.assertEqual(values(graph, self.thread)["a_conversation_state"], a_state(1))
        self.assertEqual(fake.calls[0]["request"], "내일 인천 날씨 어때?")

    def test_G2b_clarification_state_goes_to_next_turn(self):
        """G2b 추가 질문 턴의 A 상태가 다음 턴 호출에 그대로 전달된다."""
        fake = FakeOuting(a_question(1), a_result(2))
        graph = make(fake)
        first = sg.send_message(graph, self.thread, "내일 점심에 나가.")
        self.assertEqual((first["answer"], first["pending"]), ("어느 지역인지 알려 주세요.", None))
        sg.send_message(graph, self.thread, "인천이야")
        self.assertIsNone(fake.calls[0]["conversation_state"])
        self.assertEqual(fake.calls[1]["conversation_state"], a_state(1))

    def test_G2c_python_fallback_keeps_answer_and_state(self):
        """G2c 답변 단계가 실패해 Python 대체 답변이 나와도 answer와 A 상태를 보존하고 다음 턴에 넘긴다."""
        fallback = a_result(2, status="llm_invalid_output", detail_code="answer_verification", route="error",
                            answer="확인한 예보만으로 안내해 드려요.",
                            decision={"answer_source": "python_fallback", "verified": False})
        fake = FakeOuting(a_result(1), fallback, a_result(3))
        graph = make(fake)
        sg.send_message(graph, self.thread, "첫 질문")
        view = sg.send_message(graph, self.thread, "둘째 질문")
        self.assertEqual((view["answer"], view["warning"], view["debug"]["answer_source"]),
                         ("확인한 예보만으로 안내해 드려요.", True, "python_fallback"))
        self.assertEqual(values(graph, self.thread)["a_conversation_state"], a_state(2))
        sg.send_message(graph, self.thread, "셋째 질문")
        self.assertEqual(fake.calls[2]["conversation_state"], a_state(2))

    def test_G2d_runner_exception_keeps_previous_state(self):
        """G2d A 함수가 예외를 내면 안전한 안내를 보여 주고, 이전 A 상태를 그대로 둔다."""
        fake = FakeOuting(a_result(1), RuntimeError("갑작스러운 오류"), a_result(2))
        graph = make(fake)
        sg.send_message(graph, self.thread, "첫 질문")
        view = sg.send_message(graph, self.thread, "둘째 질문")
        self.assertEqual((view["answer"], view["pending"]), (sg.SAFE_ANSWER, None))
        self.assertEqual(values(graph, self.thread)["a_conversation_state"], a_state(1))
        sg.send_message(graph, self.thread, "셋째 질문")
        self.assertEqual(fake.calls[2]["conversation_state"], a_state(1))


# %% G3. 모순된 결과에서는 승인 카드를 만들지 않음
class G3_InconsistentResults(GraphCase):

    def test_G3_no_approval_from_error_or_inconsistent_results(self):
        """G3 오류 경로의 제안, 승인 표시와 제안 불일치, status·route 불일치 → 승인 카드 없음, 기록 남김"""
        cases = {
            "error_with_proposal": a_result(1, status="llm_unavailable", route="error",
                                            requires_user_confirmation=True, proposed_action=outing_proposal()),
            "flag_without_action": a_result(1, requires_user_confirmation=True),
            "action_without_flag": a_result(1, proposed_action=outing_proposal()),
            "status_route_mismatch": a_result(1, status="needs_input", route="schedule_change",
                                              requires_user_confirmation=True, proposed_action=outing_proposal()),
            "already_applied": a_result(1, route="schedule_change", requires_user_confirmation=True,
                                        proposed_action=dict(outing_proposal(), applied=True)),
        }
        for name, result in cases.items():
            with self.subTest(name):
                graph, thread = make(FakeOuting(result), a_exec=FakeExecutor()), sg.new_thread_id()
                view = sg.send_message(graph, thread, "질문")
                self.assertIsNone(view["pending"])
                self.assertIsNone(sg.waiting_action(graph, thread))
                self.assertEqual(view["answer"], "A 답변 1")
                self.assertIn("contract_violation", events(graph, thread))


# %% G4. A 일정 변경 제안의 승인·거절
class G4_OutingApproval(GraphCase):

    def test_G4a_proposal_interrupts_with_envelope(self):
        """G4a 제안 → approval에서 멈춤. 봉투 키가 채워지고 approval 키는 없으며, A 원본 제안은 바뀌지 않는다."""
        result = a_proposal_result(1)
        original = copy.deepcopy(result["proposed_action"])
        fake = FakeOuting(result)
        graph = make(fake)
        view = sg.send_message(graph, self.thread, "내일 오후 2시에 인천 가는데 비 오면 시간 바꿀까?")
        self.assertEqual(graph.get_state(sg.run_config(self.thread)).next, ("approval",))
        pending = values(graph, self.thread)["pending_action"]
        self.assertEqual(set(pending), {"type", "action_id", "summary", "requires_approval", "applied", "created_at",
                                        "source", "payload"})
        self.assertRegex(pending["action_id"], r"^[0-9a-f]{32}$")
        self.assertEqual((pending["source"], pending["summary"]),
                         ("outing_decision", "외출 시각을 16:00로 바꾸는 것을 제안합니다."))
        self.assertEqual(pending["payload"], {k: original[k] for k in ("original_outing_at", "proposed_outing_at", "reason")})
        self.assertEqual(result["proposed_action"], original)
        self.assertEqual(values(graph, self.thread)["last_result"]["proposed_action"], original)
        self.assertEqual(view["pending"]["action_id"], pending["action_id"])
        self.assertIn("14:00 → 16:00", view["pending"]["preview"])
        self.assertEqual(view["answer"], "오후 4시가 더 나아 보여요.")

    def test_G4b_approve_runs_executor_once_with_same_action_id(self):
        """G4b 무시된 재개가 있어도 action_id는 그대로, A 함수는 1회. 승인 → 실행 함수 1회(승인 정보 포함)"""
        fake, executor = FakeOuting(a_proposal_result(1)), FakeExecutor()
        graph = make(fake, a_exec=executor)
        action_id = sg.send_message(graph, self.thread, "질문")["pending"]["action_id"]
        graph.invoke(Command(resume="응"), sg.run_config(self.thread))
        self.assertEqual(values(graph, self.thread)["pending_action"]["action_id"], action_id)
        view = sg.press_button(graph, self.thread, "approve", action_id)
        self.assertEqual((len(fake.calls), len(executor.calls)), (1, 1))
        self.assertEqual(executor.calls[0]["action_id"], action_id)
        self.assertEqual(executor.calls[0]["approval"]["status"], "approved")
        self.assertEqual(view["answer"], "모의 실행으로 기록했어요. (실제로 보내지 않았어요)")
        self.assertIsNone(view["pending"])
        state = values(graph, self.thread)
        self.assertEqual((state["executed_action_ids"], state["pending_action"]), ([action_id], None))
        self.assertIn("executed", events(graph, self.thread))

    def test_G4c_reject_runs_nothing_and_keeps_schedule(self):
        """G4c 거절 → 실행 0회. 기존 시각 유지와 이미 검색한 공식 안내를 다시 보여 준다."""
        executor = FakeExecutor()
        graph = make(FakeOuting(a_proposal_result(1)), a_exec=executor)
        action_id = sg.send_message(graph, self.thread, "질문")["pending"]["action_id"]
        view = sg.press_button(graph, self.thread, "reject", action_id)
        self.assertEqual(executor.calls, [])
        self.assertIn(sg.REJECTED, view["answer"])
        self.assertIn(sg.KEEP_OUTING, view["answer"])
        self.assertIn("하천변·산길 이동 피하기", view["answer"])
        self.assertEqual(values(graph, self.thread)["executed_action_ids"], [])
        self.assertIn("rejected", events(graph, self.thread))

    def test_G4d_no_executor_is_not_success(self):
        """G4d 5단계 실행 함수가 없으면 승인해도 성공으로 표시하지 않는다."""
        graph = make(FakeOuting(a_proposal_result(1)))
        action_id = sg.send_message(graph, self.thread, "질문")["pending"]["action_id"]
        view = sg.press_button(graph, self.thread, "approve", action_id)
        self.assertEqual(view["answer"], sg.NOT_CONNECTED)
        state = values(graph, self.thread)
        self.assertEqual((state["execution_result"]["status"], state["execution_result"]["applied"]), ("not_connected", False))
        self.assertEqual(state["executed_action_ids"], [])


# %% G5. 재개 값 방어
class G5_ResumeDefense(GraphCase):

    def test_G5a_free_text_resume_keeps_waiting(self):
        """G5a "응" 같은 자유 문장·형식이 틀린 값으로 재개하면 실행하지 않고 승인 대기를 유지한다."""
        executor = FakeExecutor()
        graph = make(FakeOuting(a_proposal_result(1)), a_exec=executor)
        action_id = sg.send_message(graph, self.thread, "질문")["pending"]["action_id"]
        # (참고) Command(resume=None)은 LangGraph 1.2.24 내부에서 UnboundLocalError가 나므로 넣지 않는다. 버튼은 None을 보내지 않는다.
        for response in ("응", {"decision": "yes", "action_id": action_id}, "", {"decision": "approve"}):
            graph.invoke(Command(resume=response), sg.run_config(self.thread))
            self.assertIsNotNone(sg.waiting_action(graph, self.thread))
        self.assertEqual(executor.calls, [])
        self.assertEqual(events(graph, self.thread).count("ignored_response"), 4)
        self.assertEqual(sg.press_button(graph, self.thread, "응", action_id)["answer"], sg.INVALID_BUTTON)
        sg.press_button(graph, self.thread, "approve", action_id)
        self.assertEqual(len(executor.calls), 1)

    def test_G5b_double_press_runs_once(self):
        """G5b 승인 버튼을 두 번 눌러도 실행은 1회. 두 번째는 '승인할 제안이 없다'고 안내한다."""
        executor = FakeExecutor()
        graph = make(FakeOuting(a_proposal_result(1)), a_exec=executor)
        action_id = sg.send_message(graph, self.thread, "질문")["pending"]["action_id"]
        sg.press_button(graph, self.thread, "approve", action_id)
        second = sg.press_button(graph, self.thread, "approve", action_id)
        self.assertEqual((len(executor.calls), second["answer"]), (1, sg.NO_PENDING))

    def test_G5c_button_without_waiting_runs_nothing(self):
        """G5c 승인 대기가 아닐 때 버튼을 누르면 그래프를 재개하지 않는다."""
        executor = FakeExecutor()
        graph = make(FakeOuting(a_result(1)), a_exec=executor)
        sg.send_message(graph, self.thread, "질문")
        view = sg.press_button(graph, self.thread, "approve", "아무id")
        self.assertEqual((view["answer"], executor.calls), (sg.NO_PENDING, []))
        fresh = sg.new_thread_id()
        self.assertEqual(sg.press_button(graph, fresh, "approve", "아무id")["answer"], sg.NO_PENDING)


# %% G6. 승인 대기 중 새 질문
class G6_NewMessageCancels(GraphCase):

    def test_G6a_new_message_cancels_previous_proposal(self):
        """G6a 승인 대기 중 새 발화 → 이전 제안은 실행 없이 취소·기록, 새 발화는 정상 처리"""
        fake, executor = FakeOuting(a_proposal_result(1), a_result(2)), FakeExecutor()
        graph = make(fake, a_exec=executor)
        sg.send_message(graph, self.thread, "질문")
        view = sg.send_message(graph, self.thread, "그냥 오늘 날씨 알려 줘")
        self.assertEqual((view["answer"], view["pending"], view["notice"]), ("A 답변 2", None, sg.CANCELLED))
        self.assertEqual((executor.calls, len(fake.calls)), ([], 2))
        self.assertIn("cancelled", events(graph, self.thread))
        self.assertEqual(fake.calls[1]["conversation_state"], a_state(1))

    def test_G6b_direct_invoke_while_waiting_never_executes(self):
        """G6b 앱 함수를 거치지 않고 대기 중인 thread에 새 입력을 넣어도 intake가 제안을 지우고 실행하지 않는다."""
        executor = FakeExecutor()
        graph = make(FakeOuting(a_proposal_result(1), a_result(2)), a_exec=executor)
        sg.send_message(graph, self.thread, "질문")
        graph.invoke({"user_message": "다른 질문", "control": None}, sg.run_config(self.thread))
        self.assertIsNone(sg.waiting_action(graph, self.thread))
        self.assertEqual((executor.calls, values(graph, self.thread)["pending_action"]), ([], None))
        self.assertIn("cancelled_by_new_input", events(graph, self.thread))


# %% G7. B기능 연결
class G7_LivingSupport(GraphCase):

    def test_G7a_fake_b_results_follow_contract(self):
        """G7a 테스트용 가짜 B 결과가 명세 2.5 계약 검사를 통과한다. (Fake가 틀리면 아래 테스트도 의미가 없음)"""
        for result in (b_result(), b_question(), b_proposal_result()):
            check_living_result(result)

    def test_G7b_handoff_from_a_passes_context(self):
        """G7b A가 생활 지원으로 이해하면 B로 넘어가고, B에는 A 상태가 아니라 context만 전달된다."""
        handoff = a_result(1, route="living_support", intent="living_support",
                           llm={"understanding": {"user_goal": "장보기 도움"}})
        living = FakeLiving(b_result(1))
        graph = make(FakeOuting(handoff), b=living)
        view = sg.send_message(graph, self.thread, "장보기가 힘들어")
        self.assertEqual(view["answer"], "B 답변 1")
        self.assertEqual(living.calls[0]["request"], {"user_message": "장보기가 힘들어",
                                                      "context": {"source": "outing_decision", "user_goal": "장보기 도움"}})
        self.assertIsNone(living.calls[0]["conversation_state"])

    def test_G7c_b_clarification_routes_next_turn_to_b(self):
        """G7c B가 추가 질문 중이면 다음 짧은 대답은 A를 거치지 않고 B로 가며, B 상태가 이어진다."""
        outing = FakeOuting(a_result(1, route="living_support", intent="living_support"))
        living = FakeLiving(b_question(1), b_result(2))
        graph = make(outing, b=living)
        first = sg.send_message(graph, self.thread, "장보기가 힘들어")
        self.assertEqual(first["active_feature"], "B")
        second = sg.send_message(graph, self.thread, "우유랑 계란")
        self.assertEqual((len(outing.calls), len(living.calls), second["answer"]), (1, 2, "B 답변 2"))
        self.assertEqual(living.calls[1]["conversation_state"], b_state(1))
        self.assertIsNone(second["active_feature"])

    def test_G7d_b_approval_runs_b_executor_once(self):
        """G7d B 제안 승인 → B 실행 함수 1회, 승인 정보와 payload.message 그대로. A 실행 함수는 부르지 않음"""
        result = b_proposal_result(1)
        a_exec, b_exec = FakeExecutor(), FakeExecutor()
        graph = make(FakeOuting(a_result(1, route="living_support", intent="living_support")),
                     b=FakeLiving(result), a_exec=a_exec, b_exec=b_exec)
        view = sg.send_message(graph, self.thread, "우유 하나, 계란 한 판 부탁해")
        self.assertEqual(view["pending"]["preview"], result["proposed_action"]["payload"]["message"])
        sg.press_button(graph, self.thread, "approve", view["pending"]["action_id"])
        self.assertEqual((len(b_exec.calls), a_exec.calls), (1, []))
        sent = b_exec.calls[0]
        self.assertEqual(sent["approval"]["status"], "approved")
        self.assertEqual(sent["payload"], result["proposed_action"]["payload"])
        self.assertNotIn("approval", result["proposed_action"])

    def test_G7e_b_made_approval_is_blocked(self):
        """G7e B가 approval 키를 스스로 만든 제안은 승인 카드로 올리지 않는다."""
        bad = b_proposal_result(1)
        bad["proposed_action"]["approval"] = {"status": "approved", "at": "2026-11-25T15:00+09:00"}
        graph = make(FakeOuting(a_result(1, route="living_support", intent="living_support")),
                     b=FakeLiving(bad), b_exec=FakeExecutor())
        view = sg.send_message(graph, self.thread, "우유 하나")
        self.assertIsNone(view["pending"])
        self.assertIn("approval_created_by_feature", values(graph, self.thread)["action_log"][-1]["problems"])

    def test_G7f_b_not_connected_is_honest(self):
        """G7f 운영 기본값(B 미연결): 준비 중 안내만 하고, 승인 카드·B 상담 상태·성공 문장이 없다."""
        graph = make(FakeOuting(a_result(1, route="living_support", intent="living_support")))
        view = sg.send_message(graph, self.thread, "장보기가 힘들어")
        self.assertEqual((view["answer"], view["pending"], view["active_feature"]), (sg.B_UNAVAILABLE, None, None))
        self.assertEqual(view["debug"]["status"], "feature_unavailable")
        self.assert_no_success_claim(view["answer"])


# %% G8. 상태·thread·색인 분리
class G8_Isolation(GraphCase):

    def test_G8a_a_and_b_states_never_cross(self):
        """G8a A 함수는 A 상태만, B 함수는 B 상태만 받는다."""
        outing = FakeOuting(a_question(1), a_result(2, route="living_support", intent="living_support"), a_result(3))
        living = FakeLiving(b_question(1), b_result(2))
        graph = make(outing, b=living)
        for text in ("내일 점심에 나가.", "장보기도 힘들어", "우유 두 개", "인천 날씨는?"):
            sg.send_message(graph, self.thread, text)
        for call in outing.calls:
            self.assertIn((call["conversation_state"] or {}).get("owner"), (None, "A"))
        for call in living.calls:
            self.assertIn((call["conversation_state"] or {}).get("owner"), (None, "B"))
        self.assertEqual(outing.calls[2]["conversation_state"], a_state(2))
        state = values(graph, self.thread)
        self.assertEqual((state["a_conversation_state"]["owner"], state["b_conversation_state"]["owner"]), ("A", "B"))

    def test_G8b_threads_are_separate(self):
        """G8b thread_id가 다르면 대화가 섞이지 않는다."""
        outing = FakeOuting(a_question(1), a_result(1))
        graph = make(outing)
        sg.send_message(graph, self.thread, "내일 점심에 나가.")
        sg.send_message(graph, sg.new_thread_id(), "다른 사람 질문")
        self.assertIsNone(outing.calls[1]["conversation_state"])

    def test_G8c_indexes_and_devices_go_to_their_own_feature(self):
        """G8c A 색인·장치는 A에만, B 색인·장치는 B에만 전달된다."""
        a_index, b_index = object(), object()
        outing = FakeOuting(a_result(1, route="living_support", intent="living_support"))
        living = FakeLiving(b_result(1))
        graph = make(outing, b=living, a_device="xpu", a_index=a_index, b_device="cpu", b_index=b_index)
        sg.send_message(graph, self.thread, "장보기가 힘들어")
        self.assertEqual((outing.calls[0]["device"], living.calls[0]["device"]), ("xpu", "cpu"))
        self.assertIs(outing.calls[0]["index"], a_index)
        self.assertIs(living.calls[0]["index"], b_index)


# %% G9. 입력 검사·반복 상한·직렬화
class G9_Safety(GraphCase):

    def test_G9a_empty_and_too_long_messages(self):
        """G9a 빈 입력·600자 초과는 A를 부르지 않는다."""
        outing = FakeOuting()
        graph = make(outing)
        self.assertEqual(sg.send_message(graph, self.thread, "   ")["answer"], sg.EMPTY_MESSAGE)
        self.assertEqual(sg.send_message(graph, self.thread, "가" * 601)["answer"], sg.TOO_LONG_MESSAGE)
        self.assertEqual(outing.calls, [])
        sg.send_message(graph, self.thread, "가" * 600)
        self.assertEqual(len(outing.calls), 1)

    def test_G9b_recursion_limit_is_handled(self):
        """G9b 반복 상한을 넘으면 예외 대신 안내 문장을 돌려준다."""
        graph = make(FakeOuting(a_proposal_result(1)), a_exec=FakeExecutor())
        with mock.patch.object(sg, "RECURSION_LIMIT", 1):
            view = sg.send_message(graph, self.thread, "질문")
        self.assertIn("처리 단계가 너무 많아", view["answer"])

    def test_G9c_state_is_json_serializable(self):
        """G9c 승인·실행까지 거친 State 전체가 JSON으로 저장 가능하다."""
        graph = make(FakeOuting(a_result(1, route="living_support", intent="living_support")),
                     b=FakeLiving(b_proposal_result(1)), b_exec=FakeExecutor())
        action_id = sg.send_message(graph, self.thread, "우유 하나")["pending"]["action_id"]
        sg.press_button(graph, self.thread, "approve", action_id)
        json.dumps(values(graph, self.thread), ensure_ascii=False, allow_nan=False)
        json.dumps(sg.turn_view(graph, self.thread), ensure_ascii=False, allow_nan=False)


# %% G10. 오래된 버튼은 현재 제안을 취소하지 않음
class G10_StaleButton(GraphCase):

    def test_G10_stale_button_keeps_current_proposal(self):
        """G10 첫 제안 → 새 질문으로 둘째 제안 → 첫 제안 버튼은 무시되고 둘째 제안은 그대로 대기, 이후 정상 승인"""
        outing = FakeOuting(a_proposal_result(1), a_proposal_result(2, proposed="2026-11-25T17:00+09:00"))
        executor = FakeExecutor()
        graph = make(outing, a_exec=executor)
        old_id = sg.send_message(graph, self.thread, "첫 질문")["pending"]["action_id"]
        new_id = sg.send_message(graph, self.thread, "둘째 질문")["pending"]["action_id"]
        self.assertNotEqual(old_id, new_id)

        # 1. 앱 함수 경로: 그래프를 재개하지 않고 현재 카드를 다시 보여 준다.
        view = sg.press_button(graph, self.thread, "approve", old_id)
        self.assertEqual((view["answer"], view["pending"]["action_id"]), (sg.STALE_BUTTON, new_id))
        view = sg.press_button(graph, self.thread, "reject", old_id)
        self.assertEqual(view["pending"]["action_id"], new_id)

        # 2. Command 직접 경로: approval 노드가 무시하고 다시 대기한다. (되돌아가는 엣지)
        graph.invoke(Command(resume={"decision": "reject", "action_id": old_id}), sg.run_config(self.thread))
        waiting = sg.waiting_action(graph, self.thread)
        assert waiting is not None, "승인 대기가 유지되어야 합니다."
        self.assertEqual(waiting["action_id"], new_id)
        self.assertEqual(executor.calls, [])

        # 3. 현재 버튼은 정상 동작한다.
        sg.press_button(graph, self.thread, "approve", new_id)
        self.assertEqual([c["action_id"] for c in executor.calls], [new_id])
        self.assertEqual(executor.calls[0]["payload"]["proposed_outing_at"], "2026-11-25T17:00+09:00")


# %% G11. source·type 불일치 차단
class G11_SourceTypeMismatch(GraphCase):

    def test_G11a_wrong_type_or_source_from_feature_is_blocked(self):
        """G11a A가 B 종류를, B가 A 종류나 다른 source를 내면 승인 카드를 만들지 않는다."""
        b_like = caregiver_action()
        cases = {
            "A_returns_B_type": (a_result(1, route="schedule_change", requires_user_confirmation=True,
                                          proposed_action=b_like), None),
            "B_returns_A_type": (a_result(1, route="living_support", intent="living_support"),
                                 b_result(1, requires_user_confirmation=True,
                                          proposed_action=dict(outing_proposal(), action_id="x", summary="s",
                                                               source="living_support", payload={"message": "m"}))),
            "B_wrong_source": (a_result(1, route="living_support", intent="living_support"),
                               b_proposal_result(1, source="outing_decision")),
            "unknown_type": (a_result(1, route="schedule_change", requires_user_confirmation=True,
                                      proposed_action=dict(outing_proposal(), type="send_money")), None),
        }
        for name, (a_res, b_res) in cases.items():
            with self.subTest(name):
                a_exec, b_exec = FakeExecutor(), FakeExecutor()
                graph, thread = make(FakeOuting(a_res), b=FakeLiving(b_res) if b_res else None,
                                     a_exec=a_exec, b_exec=b_exec), sg.new_thread_id()
                view = sg.send_message(graph, thread, "질문")
                self.assertIsNone(view["pending"])
                self.assertIn("contract_violation", events(graph, thread))
                self.assertEqual((a_exec.calls, b_exec.calls), ([], []))

    def test_G11b_execute_recheck_blocks_tampered_envelope(self):
        """G11b 실행 직전 재검사: 승인 없음·id 불일치·종류/출처 불일치·이미 승인 키가 있는 봉투는 실행하지 않는다."""
        good = sg.wrap_outing_proposal(outing_proposal(), sg.now_iso())
        approve = {"decision": "approve", "action_id": good["action_id"], "at": sg.now_iso()}
        self.assertIsNone(sg.check_before_execute(good, approve))
        cases = {
            "no_pending_action": (None, approve),
            "not_approved": (good, dict(approve, decision="reject")),
            "stale_id": (good, dict(approve, action_id="old")),
            "source_type_mismatch": (dict(good, source="living_support"), approve),
            "unknown_type": (dict(good, type="send_money"), approve),
            "invalid_envelope": (dict(good, approval={"status": "approved"}), approve),
            "already_applied": (dict(good, applied=True), approve),
        }
        for name, (pending, decision) in cases.items():
            with self.subTest(name):
                self.assertIsNotNone(sg.check_before_execute(pending, decision))


# %% G12. 실행 실패는 성공으로 표시하지 않음
class G12_ExecutionFailure(GraphCase):

    def test_G12_failed_execution_is_not_success(self):
        """G12 실행 함수 예외·status 오류·applied False·다른 action_id·dict 아님 → 실패 안내, executed_action_ids 변화 없음"""
        for mode in ("raise", "error", "not_applied", "wrong_id", "not_dict"):
            with self.subTest(mode):
                executor, thread = FakeExecutor(mode), sg.new_thread_id()
                graph = make(FakeOuting(a_proposal_result(1)), a_exec=executor)
                action_id = sg.send_message(graph, thread, "질문")["pending"]["action_id"]
                view = sg.press_button(graph, thread, "approve", action_id)
                self.assertEqual(len(executor.calls), 1)
                self.assertEqual(view["answer"], sg.EXECUTION_FAILED)
                self.assertEqual(values(graph, thread)["executed_action_ids"], [])
                self.assertIn("execution_failed", events(graph, thread))
                self.assertIsNone(view["pending"])


# %% G13. B 상담 끝내기
class G13_EndConsultation(GraphCase):

    def start_b(self):
        """B가 추가 질문 중인 상태를 만든다."""
        self.outing = FakeOuting(a_question(1), a_result(2, route="living_support", intent="living_support"), a_result(3))
        self.living = FakeLiving(b_question(1), b_question(2))
        self.graph = make(self.outing, b=self.living)
        sg.send_message(self.graph, self.thread, "내일 점심에 나가.")
        self.assertEqual(sg.send_message(self.graph, self.thread, "장보기가 힘들어")["active_feature"], "B")

    def assert_ended(self, view):
        """B 상태만 비우고 A 상태는 그대로, 다음 질문은 A로 간다."""
        self.assertEqual((view["answer"], view["active_feature"]), (sg.CONSULTATION_ENDED, None))
        state = values(self.graph, self.thread)
        self.assertEqual((state["b_conversation_state"], state["a_conversation_state"]), (None, a_state(2)))
        sg.send_message(self.graph, self.thread, "인천 날씨는?")
        self.assertEqual((len(self.outing.calls), len(self.living.calls)), (3, 1))
        self.assertEqual(self.outing.calls[2]["conversation_state"], a_state(2))

    def test_G13a_end_button(self):
        """G13a '상담 끝내기' 버튼"""
        self.start_b()
        self.assert_ended(sg.end_consultation(self.graph, self.thread))

    def test_G13b_exact_end_word(self):
        """G13b B 상담 중 '그만'만 말하면 상담을 끝낸다. (문장부호는 무시)"""
        self.start_b()
        self.assert_ended(sg.send_message(self.graph, self.thread, "그만."))

    def test_G13c_sentence_with_end_word_goes_to_b(self):
        """G13c '그만 우유도 필요해'처럼 정확히 일치하지 않으면 B의 대답으로 전달한다."""
        self.start_b()
        sg.send_message(self.graph, self.thread, "그만 우유도 필요해")
        self.assertEqual(len(self.living.calls), 2)
        self.assertEqual(self.living.calls[1]["request"]["user_message"], "그만 우유도 필요해")

    def test_G13d_end_word_without_b_goes_to_a(self):
        """G13d B 상담 중이 아니면 '그만'도 평소처럼 A로 보낸다. 버튼은 진행 중인 상담이 없다고 안내한다."""
        outing = FakeOuting()
        graph = make(outing)
        sg.send_message(graph, self.thread, "그만")
        self.assertEqual(len(outing.calls), 1)
        self.assertEqual(sg.end_consultation(graph, self.thread)["answer"], sg.NO_CONSULTATION)


# %% G14. 잘못된 자료형 입력 방어 (코드 검토에서 TypeError를 확인한 경우의 회귀 테스트)
# 핵심 요약: type·decision·status 등에 리스트·딕셔너리처럼 해시할 수 없는 값이 와도 예외 없이 거절하거나 무시한다.
BAD_VALUES = (["request_caregiver_help"], {"type": "x"}, 3, None, ["approve"])


class G14_BadTypes(GraphCase):

    def test_G14a_check_feature_result_rejects_bad_types(self):
        """G14a 결과·제안의 키가 잘못된 자료형이면 예외 없이 문제 목록을 돌려준다."""
        self.assertEqual(sg.check_feature_result(["목록"], "A"), ["result_not_dict"])
        for bad in BAD_VALUES:
            with self.subTest(bad=bad):
                a_bad = a_proposal_result(1)
                a_bad["proposed_action"]["type"] = bad
                self.assertTrue(sg.check_feature_result(a_bad, "A"))
                b_bad = b_proposal_result(1)
                b_bad["proposed_action"]["type"] = bad
                self.assertTrue(sg.check_feature_result(b_bad, "B"))
                for key in ("status", "route", "requires_user_confirmation"):
                    self.assertTrue(sg.check_feature_result(dict(a_proposal_result(1), **{key: bad}), "A"))
                for key in ("action_id", "summary", "payload", "source"):
                    b_bad = b_proposal_result(1)
                    b_bad["proposed_action"][key] = bad
                    self.assertTrue(sg.check_feature_result(b_bad, "B"))
                self.assertTrue(sg.check_feature_result(dict(a_result(1), proposed_action=bad,
                                                             requires_user_confirmation=True), "A"))

    def test_G14b_read_decision_ignores_bad_types(self):
        """G14b 재개 값의 decision·action_id·현재 제안이 잘못된 자료형이면 무시(ignored)한다."""
        pending = {"action_id": "a" * 32}
        for bad in BAD_VALUES:
            with self.subTest(bad=bad):
                self.assertEqual(sg.read_decision({"decision": bad, "action_id": "a" * 32}, pending)["decision"], "ignored")
                self.assertEqual(sg.read_decision({"decision": "approve", "action_id": bad}, pending)["decision"], "ignored")
                self.assertEqual(sg.read_decision({"decision": "approve", "action_id": bad}, {"action_id": bad})["decision"],
                                 "ignored")
                self.assertEqual(sg.read_decision(bad, pending)["decision"], "ignored")
        self.assertEqual(sg.read_decision({"decision": "approve", "action_id": "a" * 32}, ["목록"])["decision"], "ignored")
        self.assertEqual(sg.read_decision({"decision": "approve", "action_id": "a" * 32}, pending)["decision"], "approve")

    def test_G14c_is_execution_success_rejects_bad_types(self):
        """G14c 실행 함수의 status·applied·action_id가 잘못된 자료형이면 성공이 아니다."""
        good = {"status": "ok", "applied": True, "action_id": "id1"}
        self.assertTrue(sg.is_execution_success(good, "id1"))
        for bad in BAD_VALUES:
            with self.subTest(bad=bad):
                self.assertFalse(sg.is_execution_success(dict(good, status=bad), "id1"))
                self.assertFalse(sg.is_execution_success(dict(good, applied=bad), "id1"))
                self.assertFalse(sg.is_execution_success(dict(good, action_id=bad), "id1"))
                self.assertFalse(sg.is_execution_success(bad, "id1"))

    def test_G14d_check_before_execute_rejects_bad_types(self):
        """G14d 실행 직전 재검사에 잘못된 자료형의 제안·결정·type·action_id가 와도 예외 없이 실행을 막는다."""
        good = sg.wrap_outing_proposal(outing_proposal(), sg.now_iso())
        approve = {"decision": "approve", "action_id": good["action_id"], "at": sg.now_iso()}
        for bad in BAD_VALUES:
            with self.subTest(bad=bad):
                self.assertIsNotNone(sg.check_before_execute(bad, approve))
                self.assertIsNotNone(sg.check_before_execute(good, bad))
                self.assertIsNotNone(sg.check_before_execute(dict(good, type=bad), approve))
                self.assertIsNotNone(sg.check_before_execute(dict(good, source=bad), approve))
                self.assertIsNotNone(sg.check_before_execute(dict(good, action_id=bad), dict(approve, action_id=bad)))
                self.assertIsNotNone(sg.check_before_execute(good, dict(approve, decision=bad)))

    def test_G14e_view_and_card_tolerate_bad_types(self):
        """G14e 화면 변환·승인 카드가 목록·딕셔너리 자리에 잘못된 값이 와도 예외 없이 빈 값으로 처리한다."""
        for bad in (3, "문자열", {"text": "x"}, None):
            with self.subTest(bad=bad):
                self.assertEqual(sg.to_common_view({"official_guidance": bad}, "A")["evidence"], [])
                self.assertEqual(sg.to_common_view({"living_evidence": bad}, "B")["evidence"], [])
                card = sg.approval_card({"type": "request_caregiver_help", "payload": bad})
                self.assertEqual(card["preview"], "")
        self.assertEqual(sg.log_entry("e", ["목록"])["event"], "e")

    def test_G14f_graph_survives_bad_types_end_to_end(self):
        """G14f 그래프 전체: 잘못된 type의 제안은 승인 카드 없이 답변만, 리스트 decision 재개는 무시,
        리스트 status를 돌려주는 실행 함수는 실패로 처리한다. 어느 경우도 예외가 나지 않는다."""
        # 1. A가 type이 리스트인 제안을 돌려줌
        bad = a_proposal_result(1)
        bad["proposed_action"]["type"] = ["propose_outing_time_change"]
        graph = make(FakeOuting(bad), a_exec=FakeExecutor())
        view = sg.send_message(graph, self.thread, "질문")
        self.assertEqual((view["pending"], view["answer"]), (None, "오후 4시가 더 나아 보여요."))
        self.assertIn("contract_violation", events(graph, self.thread))

        # 2. 리스트 decision으로 재개 → 무시하고 계속 대기
        executor, thread = FakeExecutor(), sg.new_thread_id()
        graph = make(FakeOuting(a_proposal_result(1)), a_exec=executor)
        action_id = sg.send_message(graph, thread, "질문")["pending"]["action_id"]
        graph.invoke(Command(resume={"decision": ["approve"], "action_id": action_id}), sg.run_config(thread))
        graph.invoke(Command(resume={"decision": "approve", "action_id": [action_id]}), sg.run_config(thread))
        self.assertIsNotNone(sg.waiting_action(graph, thread))
        self.assertEqual(executor.calls, [])

        # 3. 실행 함수가 status를 리스트로 돌려줌 → 실패 안내, 성공 기록 없음
        class ListStatusExecutor(FakeExecutor):
            def __call__(self, action):
                output = super().__call__(action)
                assert isinstance(output, dict)
                output["status"] = ["ok"]
                return output

        executor, thread = ListStatusExecutor(), sg.new_thread_id()
        graph = make(FakeOuting(a_proposal_result(1)), a_exec=executor)
        action_id = sg.send_message(graph, thread, "질문")["pending"]["action_id"]
        view = sg.press_button(graph, thread, "approve", action_id)
        self.assertEqual((view["answer"], values(graph, thread)["executed_action_ids"]), (sg.EXECUTION_FAILED, []))


# %% I층. 실제 A 코드 + 가상 예보·가상 LLM
# 핵심 요약: 3단계 테스트의 Harness로 Ollama·날씨·RAG만 가상으로 바꾸고, 실제 run_outing_decision()을 그래프에 연결한다.
S08_MESSAGE = "내일 오후 2시에 인천 가는데 비 오면 시간 바꿀까?"
S08_U = tod.u(intent="schedule_change", needs_weather=True, city_text="인천", date_text="내일", time_text="오후 2시")
RAIN_AT_14 = {14: {"precipitation_mm": 2.0, "precipitation_probability_pct": 80}}


class I_RealOutingCode(GraphCase):

    def harness_graph(self, harness, executor=None):
        """실제 A 공개 함수를 연결한 그래프 (색인은 3단계 테스트의 가상 색인)"""
        return sg.build_graph(a_runner=od.run_outing_decision, a_device="xpu", a_index=tod.FAKE_INDEX,
                              a_executor=executor)

    def test_I1_virtual_forecast_schedule_change_approve_and_reject(self):
        """I1 명시적인 가상 예보(14시 비 → 16시 맑음)로 실제 A가 제안 → 승인 시 실행 1회, 거절 시 0회"""
        for decision, expected_calls in (("approve", 1), ("reject", 0)):
            with self.subTest(decision), tod.Harness(llm=tod.FakeLLM({S08_MESSAGE: S08_U}),
                                                     weather=tod.FakeWeather(by_hour=RAIN_AT_14)) as h:
                executor, thread = FakeExecutor(), sg.new_thread_id()
                graph = self.harness_graph(h, executor)
                view = sg.send_message(graph, thread, S08_MESSAGE)
                last = values(graph, thread)["last_result"]
                self.assertEqual((last["route"], last["requires_user_confirmation"]), ("schedule_change", True))
                pending = values(graph, thread)["pending_action"]
                self.assertEqual((pending["payload"]["original_outing_at"], pending["payload"]["proposed_outing_at"]),
                                 ("2026-11-25T14:00+09:00", "2026-11-25T16:00+09:00"))
                self.assertIn("[확인한 예보]", view["answer"])
                after = sg.press_button(graph, thread, decision, view["pending"]["action_id"])
                self.assertEqual(len(executor.calls), expected_calls)
                self.assertIsNone(after["pending"])
                if decision == "reject":
                    self.assertIn(sg.KEEP_OUTING, after["answer"])

    def test_I2_multi_turn_short_answer_through_graph(self):
        """I2 '내일 저녁에 비 와?' → '인천이야': 그래프에 저장한 A 상태로 짧은 대답이 이어지고 예보를 조회한다."""
        first_q = "내일 저녁에 비 와?"
        llm = tod.FakeLLM({first_q: tod.u(**tod.WQ, date_text="내일", time_text="저녁"),
                           "인천이야": tod.u(intent="weather_question", answers_previous_question=True, city_text="인천")})
        with tod.Harness(llm=llm) as h:
            graph = self.harness_graph(h)
            first = sg.send_message(graph, self.thread, first_q)
            self.assertEqual(first["debug"]["route"], "needs_clarification")
            self.assertEqual(h.weather.calls, [])
            second = sg.send_message(graph, self.thread, "인천이야")
            self.assertEqual((second["debug"]["route"], second["debug"]["status"]), ("weather_question", "ok"))
            self.assertTrue(h.weather.calls)
            self.assertEqual(values(graph, self.thread)["a_conversation_state"]["turn"], 2)

    def test_I3_python_fallback_keeps_answer_and_state(self):
        """I3 답변 단계 Ollama 오류 → Python 대체 답변. answer와 A 상태(예보 캐시 포함)가 저장되고 다음 턴이 재사용한다."""
        message = tod.TOMORROW_EVENING_Q
        llm = tod.FailingLLM({message: tod.TOMORROW_EVENING_U})
        with tod.Harness(llm=llm) as h:
            graph = self.harness_graph(h)
            view = sg.send_message(graph, self.thread, message)
            self.assertEqual(view["debug"]["answer_source"], "python_fallback")
            self.assertTrue(view["answer"].strip())
            self.assertIsNone(view["pending"])
            state = values(graph, self.thread)["a_conversation_state"]
            self.assertEqual(state["turn"], 1)
            self.assertIsNotNone(state["weather_cache"])
            calls_before = len(h.weather.calls)
            sg.send_message(graph, self.thread, message)
            self.assertEqual(len(h.weather.calls), calls_before)          # 30분 캐시 재사용
            self.assertEqual(values(graph, self.thread)["a_conversation_state"]["turn"], 2)


# %% L층. 실제 qwen2.5:3b·실제 Open-Meteo (--live-ollama 지정 시에만)
# 핵심 요약: 실제 모델 답변은 매번 다를 수 있으므로 문장 내용이 아니라 그래프 계약(답변·상태 보존, 실제 조회, 가짜 성공 없음)만 확인한다.
class L_LiveIntegration(GraphCase):

    def setUp(self):
        super().setUp()
        if not OPTIONS.live_ollama:
            raise unittest.SkipTest("--live-ollama 를 지정하면 실행합니다. (ollama serve와 qwen2.5:3b, 인터넷 필요)")
        report = od.ollama_status(od.ollama_settings())
        if not (report["server_running"] and report["model_installed"]):
            raise unittest.SkipTest(f"Ollama 서버 또는 모델이 준비되지 않았습니다: {report}")
        self.executor = FakeExecutor()
        self.graph = sg.build_graph(a_device=OPTIONS.device, a_executor=self.executor)

    def check_turn(self, view):
        """모든 실제 턴의 공통 계약: 답변 있음, 예외 없음, 승인 없이 실행 없음"""
        self.assertTrue(view["answer"].strip())
        self.assertNotEqual(view["answer"], sg.SAFE_ANSWER)
        self.assertEqual(self.executor.calls, [])
        print(f"\n  [{view['debug']['route']}/{view['debug']['status']}/{view['debug']['answer_source']}] "
              f"{view['answer'][:160]}")

    def test_L1_real_multi_turn_with_open_meteo(self):
        """L1 실제 Qwen + 실제 Open-Meteo 다중 턴. Python 대체 답변이 나와도 답변·상태가 보존되면 통과"""
        first = sg.send_message(self.graph, self.thread, "내일 점심에 시장 가려는데 비 와?")
        self.check_turn(first)
        turn1 = values(self.graph, self.thread)["a_conversation_state"]["turn"]
        second = sg.send_message(self.graph, self.thread, "인천이야")
        self.check_turn(second)
        self.assertGreater(values(self.graph, self.thread)["a_conversation_state"]["turn"], turn1)
        fetched = [t for t in second["debug"]["tool_trace"] if t.get("tool") == "get_outing_weather"]
        self.assertTrue(fetched, "둘째 턴에서 실제 날씨를 조회하지 않았습니다.")
        self.assertTrue(any(t.get("status") in ("ok", "partial", "reused_from_conversation_state") for t in fetched))

    def test_L2_real_qwen_with_virtual_forecast(self):
        """L2 실제 Qwen + 가상 예보(14시 비). 제안이 나오면 승인 흐름을 실행하고, 안 나와도 실패로 보지 않는다."""
        with mock.patch.object(od, "get_outing_weather", tod.FakeWeather(by_hour=RAIN_AT_14)):
            view = sg.send_message(self.graph, self.thread, S08_MESSAGE)
            self.check_turn(view)
            if view["pending"] is None:
                print("  제안 없음: 실제 모델이 일정 변경을 제안하지 않았습니다. (실패 아님)")
                return
            sg.press_button(self.graph, self.thread, "approve", view["pending"]["action_id"])
            self.assertEqual(len(self.executor.calls), 1)
            self.assertEqual(self.executor.calls[0]["approval"]["status"], "approved")

    def test_L3_real_weather_schedule_question(self):
        """L3 실제 날씨로 일정 변경 질문. 바꿀 필요가 없으면 제안이 없는 것이 정상. 제안이 있으면 거절해 실행 0회 확인"""
        view = sg.send_message(self.graph, self.thread, "내일 오후 2시에 인천 시장 가는데 비 오면 시간 바꿀까?")
        self.check_turn(view)
        if view["pending"]:
            after = sg.press_button(self.graph, self.thread, "reject", view["pending"]["action_id"])
            self.assertIn(sg.KEEP_OUTING, after["answer"])
        self.assertEqual(self.executor.calls, [])


# %% R층. 1·2·3단계 코드 보존
class R_Regression(unittest.TestCase):

    def test_R1_protected_files_match_base_commit(self):
        """R1 1·2·3단계 파일의 SHA-256이 기준 커밋과 같다. (줄바꿈 CRLF/LF 차이는 같게 본다)"""
        for name in PROTECTED_FILES:
            with self.subTest(name):
                try:
                    base = subprocess.run(["git", "show", f"{BASE_COMMIT}:{name}"], cwd=ROOT, capture_output=True,
                                          check=True).stdout
                except (OSError, subprocess.CalledProcessError) as exc:
                    raise unittest.SkipTest(f"git으로 기준 파일을 읽지 못했습니다: {exc}")
                current = (ROOT / name).read_bytes()
                normalize = lambda data: hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()  # noqa: E731
                self.assertEqual(normalize(current), normalize(base))


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
    if OPTIONS.device not in (None, "xpu", "cuda", "cpu"):
        raise SystemExit("--device 는 xpu, cuda, cpu 중 하나입니다.")
    unittest.main(argv=[sys.argv[0]] + rest)


if __name__ == "__main__":
    main()
