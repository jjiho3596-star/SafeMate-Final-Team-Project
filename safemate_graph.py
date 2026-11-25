"""SafeMate 4단계: A·B기능 공통 LangGraph

핵심 요약
- 사용자 한 마디를 A기능(외출·날씨, run_outing_decision) 또는 B기능(생활 지원, 주입된 함수)으로 보낸다.
- A·B의 대화 상태는 State의 서로 다른 칸에 따로 저장하고, 서로에게 넘기지 않는다.
- 승인이 필요한 제안은 approval 노드의 interrupt()에서 멈추고, Gradio 버튼 값(Command(resume=...))으로만 이어 간다.
- 실제 실행(부작용)은 execute 노드에만 있다. 승인 전에는 아무것도 실행하지 않는다.
- B기능과 5단계 일정 저장 함수는 아직 없으므로 build_graph()에 함수를 주입하는 자리만 둔다.
  주입되지 않았으면 "아직 준비 중"이라고 안내하고, 성공한 것처럼 보이지 않는다.

수업 4장 체크리스트 대응
- State = TypedDict + reducer(action_log, executed_action_ids)
- Node는 바뀐 값만 반환
- checkpointer(InMemorySaver) + thread_id
- interrupt 뒤에 부작용 (approval 노드는 읽기만, execute 노드가 실행)
- loop에는 종료 상한 (recursion_limit)
- 흐름은 코드(분기 함수)가 정하고, 판단(의도 이해)은 A·B 상담 함수 안의 LLM이 맡는다.
"""

# %% 공통 라이브러리와 3단계 공개 함수 불러오기
# 핵심 요약: LangGraph 기본 부품과 A기능 공개 함수만 가져온다. 1·2·3단계 코드는 고치지 않고 호출만 한다.
from __future__ import annotations

import copy
import json
import operator
import sys
import uuid
from datetime import datetime
from typing import Annotated, Any, Callable, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from a_outing import KST
from outing_decision import run_outing_decision


# %% 프로젝트 설정값
# 핵심 요약: 그래프 반복 상한, 입력 길이, 허용하는 승인 제안 종류와 화면 문장을 한곳에 모은다.
RECURSION_LIMIT = 12            # 한 번의 invoke에서 지나갈 수 있는 노드 수 상한 (정상 경로는 최대 4)
MAX_MESSAGE_LENGTH = 600        # A·B 공통 입력 계약 (명세 1.3·2.2)
END_WORDS = {"그만", "취소", "처음으로"}   # B 상담 중 이 말만 단독으로 하면 상담을 끝낸다
BUTTON_DECISIONS = {"approve", "reject"}  # 화면 버튼이 보낼 수 있는 값
RESUME_DECISIONS = {"approve", "reject", "cancel"}  # cancel은 새 질문이 왔을 때 앱 함수만 보낸다
SUCCESS_STATUSES = {"ok", "already_executed"}

# 승인 제안 종류별로 누가 만들었어야 하는지(source)와 어느 기능의 제안인지(feature)를 정한다.
# 상담 노드와 execute 노드가 같은 표로 두 번 검사한다.
ALLOWED_ACTIONS = {
    "propose_outing_time_change": {"source": "outing_decision", "feature": "A"},
    "request_caregiver_help": {"source": "living_support", "feature": "B"},
}

EMPTY_MESSAGE = "궁금한 내용을 한 문장으로 말씀해 주세요."
TOO_LONG_MESSAGE = f"말씀이 너무 길어요. {MAX_MESSAGE_LENGTH}자 이내로 나눠서 말씀해 주세요."
SAFE_ANSWER = "지금은 답변을 만들 수 없어요. 잠시 후 다시 말씀해 주세요."
B_UNAVAILABLE = "생활 지원(장보기 도움 요청) 기능은 아직 준비 중이에요. 지금은 요청이 실행되지 않습니다."
CONSULTATION_ENDED = "생활 지원 상담을 마쳤어요. 새로 궁금한 것을 말씀해 주세요."
NO_CONSULTATION = "지금은 진행 중인 생활 지원 상담이 없어요. 궁금한 것을 말씀해 주세요."
REJECTED = "알겠어요, 그대로 둘게요."
KEEP_OUTING = "기존 외출 시각을 유지합니다."
CANCELLED = "이전 제안은 실행하지 않고 취소했어요."
NOT_CONNECTED = "이 요청을 처리하는 기능이 아직 연결되지 않아 바뀐 것이 없어요."
EXECUTION_FAILED = "요청을 처리하지 못했어요. 성공으로 기록하지 않았습니다."
EXECUTION_BLOCKED = "승인 정보가 맞지 않아 실행하지 않았어요."
ALREADY_DONE = "이미 처리한 요청이에요. 다시 실행하지 않았어요."
DONE_DEFAULT = "승인하신 요청을 처리했어요."
NO_PENDING = "지금은 승인할 제안이 없어요."
STALE_BUTTON = "이전 제안의 버튼이에요. 지금 보이는 제안에 답해 주세요."
INVALID_BUTTON = "승인 또는 거절 버튼으로 답해 주세요."


# %% 공통 State
# 핵심 요약: 노드들이 함께 보는 데이터. A·B 대화 상태는 칸을 나눠 섞이지 않게 하고, 기록 두 칸은 reducer로 이어 붙인다.
# State에는 JSON으로 저장할 수 있는 값만 넣는다. 색인·함수 같은 객체는 build_graph()의 클로저로 노드에 전달한다.
# InMemorySaver는 프로그램이 실행되는 동안만 이 State를 보관한다. (프로그램 종료 후 영구 저장은 5단계)
class SafeMateState(TypedDict, total=False):
    # 이번 턴 입력
    user_message: str
    control: Literal["end_consultation"] | None   # 화면의 '상담 끝내기' 버튼
    # 기능별 대화 상태 (A는 A 칸만, B는 B 칸만 읽고 쓴다)
    a_conversation_state: dict | None
    b_conversation_state: dict | None
    active_feature: Literal["B"] | None           # B가 추가 질문 중이면 "B"
    handoff_context: dict | None                  # A→B로 넘기는 정보 (대화 상태가 아님)
    # 이번 턴 결과
    feature: Literal["A", "B", "system"] | None
    last_result: dict | None                      # A·B 원본 결과
    view: dict | None                             # 화면 공통 값 (to_common_view)
    answer: str                                   # 화면에 보여 줄 최종 문장
    # 승인 흐름
    pending_action: dict | None                   # 승인을 기다리는 공통 봉투 하나
    approval_decision: dict | None                # 버튼 결과 {"decision", "action_id", "at"}
    execution_result: dict | None                 # 실행 함수 반환값
    # 누적 기록 (reducer: 새 항목을 기존 목록 뒤에 이어 붙인다)
    action_log: Annotated[list[dict], operator.add]
    executed_action_ids: Annotated[list[str], operator.add]


# 매 턴 intake가 비우는 칸. 대화 상태·active_feature·기록은 thread에 계속 남긴다.
TURN_RESET: dict[str, Any] = {"control": None, "handoff_context": None, "feature": None, "last_result": None, "view": None,
              "answer": "", "approval_decision": None, "execution_result": None}


# %% 현재 한국 시각
def now_iso() -> str:
    """승인·기록 시각에 쓰는 KST ISO 문자열 (분 단위)"""
    return datetime.now(KST).isoformat(timespec="minutes")


# %% JSON 저장 가능한 값으로 바꾸는 함수
# 핵심 요약: 주입된 함수가 계약과 달리 이상한 객체를 돌려줘도 체크포인터 저장이 깨지지 않게 한다.
def json_safe(value: Any) -> Any:
    """JSON으로 바꿨다가 다시 읽은 사본을 돌려준다. 바꿀 수 없는 값은 문자열로 남긴다."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


# %% A 제안을 공통 승인 봉투로 감싸는 함수 (명세 2.4)
# 핵심 요약: A의 일정 변경 제안에는 action_id·summary 등이 없으므로 4단계가 보완한다. A 원본 딕셔너리는 고치지 않는다.
# a_node에서 한 번만 실행한다. approval 노드(interrupt 앞)에서 만들면 재개할 때마다 action_id가 바뀌어 중복 방지가 깨진다.
def wrap_outing_proposal(action: dict[str, Any], now: str) -> dict[str, Any]:
    """A의 propose_outing_time_change를 공통 봉투(type·action_id·summary·payload 등)로 만든다."""
    # 1. A가 만든 값은 그대로 payload로 옮긴다.
    payload = {key: action[key] for key in ("original_outing_at", "proposed_outing_at", "reason")}
    # 2. 공통 키를 채운다. action_id는 여기서 한 번만 만든다.
    return {"type": action["type"], "action_id": uuid.uuid4().hex,
            "summary": f"외출 시각을 {payload['proposed_outing_at'][11:16]}로 바꾸는 것을 제안합니다.",
            "requires_approval": True, "applied": False, "created_at": now,
            "source": "outing_decision", "payload": payload}


# %% 값 확인 도우미
# 핵심 요약: 주입된 함수가 계약과 다른 자료형(리스트 등)을 돌려줘도 TypeError 없이 '허용하지 않음'으로 판단하게 한다.
def is_text(value: Any) -> bool:
    """비어 있지 않은 문자열인지"""
    return isinstance(value, str) and bool(value.strip())


def allowed_rule(action_type: Any) -> dict[str, str] | None:
    """허용 표에서 제안 종류의 규칙을 찾는다. 문자열이 아니면(리스트 등 해시할 수 없는 값 포함) None"""
    return ALLOWED_ACTIONS.get(action_type) if isinstance(action_type, str) else None


# %% A·B 결과 계약 검사 함수
# 핵심 요약: 결과의 status·route·승인 요청·제안이 서로 맞는지, 제안의 종류와 출처가 그 기능의 것인지 확인한다.
# 문제 목록이 비어 있을 때만 승인 카드를 만든다. (오류 결과나 모순된 결과에서 승인을 만들지 않기 위함)
def check_feature_result(result: Any, feature: str) -> list[str]:
    """문제 이름 목록을 돌려준다. 빈 목록이면 계약을 지킨 결과다."""
    if not isinstance(result, dict):
        return ["result_not_dict"]
    problems: list[str] = []

    # 1. 화면 문장과 대화 상태
    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        problems.append("answer_empty")
    if not isinstance(result.get("conversation_state"), dict):
        problems.append("conversation_state_not_dict")

    # 2. status·route는 문자열이어야 하고, 추가 질문 상태와 경로가 같이 움직여야 한다.
    for key in ("status", "route"):
        if not isinstance(result.get(key), str):
            problems.append(f"{key}_not_text")
    if (result.get("status") == "needs_input") != (result.get("route") == "needs_clarification"):
        problems.append("status_route_mismatch")

    # 3. 승인 요청 표시와 제안이 같이 있는지
    asks = result.get("requires_user_confirmation")
    action = result.get("proposed_action")
    if not isinstance(asks, bool):
        problems.append("requires_user_confirmation_not_bool")
    elif asks != (action is not None):
        problems.append("confirmation_action_mismatch")
    if action is None:
        return problems
    if not isinstance(action, dict):
        return problems + ["proposed_action_not_dict"]

    # 4. 제안 자체의 공통 규칙: 오류·추가 질문 턴에는 제안이 없어야 하고, 승인 전 상태여야 한다.
    if result.get("route") in ("error", "needs_clarification"):
        problems.append(f"proposal_on_{result.get('route')}")
    if action.get("requires_approval") is not True or action.get("applied") is not False:
        problems.append("proposal_not_pending")
    if "approval" in action:
        problems.append("approval_created_by_feature")   # 승인은 공통 그래프만 붙인다

    # 5. 제안 종류가 이 기능의 것인지, 출처가 맞는지 (type이 문자열이 아니면 허용 표에 없는 것으로 본다)
    rule = allowed_rule(action.get("type"))
    if rule is None:
        return problems + ["unknown_action_type"]
    if rule["feature"] != feature:
        problems.append("action_type_feature_mismatch")
    if feature == "A":
        # A 원본에는 source가 없다. 래퍼가 outing_decision을 넣는다. (있다면 같아야 함)
        if action.get("source", rule["source"]) != rule["source"]:
            problems.append("source_mismatch")
        if not all(isinstance(action.get(k), str) and action.get(k)
                   for k in ("original_outing_at", "proposed_outing_at", "reason")):
            problems.append("outing_proposal_fields")
    else:
        # B는 공통 봉투를 직접 만든다. (명세 2.2 일관성 규칙 2, 2.4)
        if action.get("source") != rule["source"]:
            problems.append("source_mismatch")
        if result.get("status") != "ok" or result.get("route") != "living_support":
            problems.append("proposal_status_route")
        for key in ("action_id", "summary"):
            if not is_text(action.get(key)):
                problems.append(f"{key}_missing")
        payload = action.get("payload")
        if not isinstance(payload, dict) or not is_text(payload.get("message")):
            problems.append("payload_message_missing")
    return problems


# %% 화면 공통 값 변환 함수 (명세 3.1)
# 핵심 요약: A의 official_guidance(키 text)와 B의 living_evidence(키 content)를 같은 모양으로 맞추고, 경고 여부를 정한다.
def to_common_view(result: dict[str, Any], feature: str) -> dict[str, Any]:
    """A·B 결과에서 화면이 함께 쓰는 값만 꺼낸다. 원본 결과는 고치지 않는다."""
    # 1. 근거를 같은 모양으로 맞춘다. (목록이 아니면 근거 없음으로 본다)
    raw = result.get("official_guidance" if feature == "A" else "living_evidence")
    items = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
    if feature == "A":
        evidence = [{"content": g.get("text"), "title": g.get("title"), "publisher": g.get("publisher"),
                     "url": g.get("url"), "pdf_page": g.get("pdf_page")} for g in items]
    else:
        evidence = items
    # 2. 분기·화면에 쓰는 공통 값만 돌려준다.
    return {"feature": feature, "route": result.get("route"), "status": result.get("status"),
            "warning": result.get("status") not in ("ok", "needs_input"), "evidence": evidence}


# %% 승인 카드 함수
# 핵심 요약: 승인·거절 버튼 위에 보여 줄 내용. B는 보호자에게 갈 문장 미리보기, A는 원래·변경 시각과 예보 차이를 보여 준다.
def approval_card(action: dict[str, Any]) -> dict[str, Any]:
    """interrupt()에 넘기고 화면에도 쓰는 승인 카드"""
    raw = action.get("payload")
    payload: dict[str, Any] = raw if isinstance(raw, dict) else {}
    if action.get("type") == "request_caregiver_help":
        preview = payload.get("message", "")
    else:
        preview = (f"{str(payload.get('original_outing_at', ''))[11:16]} → {str(payload.get('proposed_outing_at', ''))[11:16]}"
                   f" / {payload.get('reason', '')}")
    return {"action_id": action.get("action_id"), "type": action.get("type"), "source": action.get("source"),
            "summary": action.get("summary", ""), "preview": preview}


# %% 버튼 결과 해석 함수
# 핵심 요약: 재개 값이 정해진 형식이고 현재 제안의 action_id와 같을 때만 그 결정을 쓴다. 나머지는 모두 '무시'다.
# '무시'는 제안을 취소하지 않는다. 현재 승인 대기를 그대로 유지한다. (오래된 버튼, "응" 같은 자유 문장)
def read_decision(response: Any, pending: Any) -> dict[str, Any]:
    """{"decision": approve|reject|cancel|ignored, "action_id", "at", "reason"}"""
    # 1. 현재 제안의 action_id (제안이 dict가 아니거나 id가 문자열이 아니면 없음)
    current = pending.get("action_id") if isinstance(pending, dict) else None
    current = current if is_text(current) else None
    # 2. 재개 값 형식: dict이고 decision이 정해진 문자열이어야 한다. (리스트 등은 in 검사 전에 거른다)
    decision = response.get("decision") if isinstance(response, dict) else None
    if not isinstance(response, dict) or not isinstance(decision, str) or decision not in RESUME_DECISIONS:
        return {"decision": "ignored", "action_id": current, "at": now_iso(), "reason": "invalid_response"}
    # 3. 지금 기다리는 제안의 버튼이어야 한다.
    if current is None or response.get("action_id") != current:
        return {"decision": "ignored", "action_id": current, "at": now_iso(), "reason": "stale_action_id"}
    return {"decision": decision, "action_id": current, "at": now_iso(), "reason": None}


# %% 실행 직전 재검사 함수
# 핵심 요약: 승인 노드를 지나왔더라도 실행 직전에 한 번 더 확인한다. 하나라도 어기면 실행하지 않는다.
def check_before_execute(pending: Any, decision: Any) -> str | None:
    """문제 이름을 돌려준다. None이면 실행해도 된다."""
    # 1. 승인 대기 중인 제안과 그 제안에 대한 승인 버튼이 있어야 한다. (dict가 아니면 승인 없음으로 본다)
    if not isinstance(pending, dict):
        return "no_pending_action"
    action_id = pending.get("action_id")
    if not isinstance(decision, dict) or decision.get("decision") != "approve" or not is_text(action_id) \
            or decision.get("action_id") != action_id:
        return "not_approved"
    # 2. 제안 종류와 출처가 허용 표와 맞아야 한다.
    rule = allowed_rule(pending.get("type"))
    if rule is None or pending.get("source") != rule["source"]:
        return "source_type_mismatch"
    # 3. 아직 실행하지 않은 제안이어야 하고, 승인 정보는 그래프만 붙인다.
    if pending.get("requires_approval") is not True or pending.get("applied") is not False or "approval" in pending:
        return "invalid_envelope"
    return None


# %% 실행 성공 판정 함수
# 핵심 요약: 실행 함수가 "성공"을 분명히 돌려준 경우에만 성공으로 본다. 애매하면 실패로 처리한다.
def is_execution_success(output: Any, action_id: str) -> bool:
    """status가 ok·already_executed이고, applied가 True이며, action_id가 같아야 성공이다."""
    if not isinstance(output, dict):
        return False
    status = output.get("status")      # 리스트 등은 in 검사 전에 거른다
    return (isinstance(status, str) and status in SUCCESS_STATUSES
            and output.get("applied") is True and output.get("action_id") == action_id)


# %% 기록 한 줄 함수
def log_entry(event: str, action: Any = None, **extra: Any) -> dict[str, Any]:
    """action_log에 붙일 한 줄. 개인정보 없이 종류·id·시각만 남긴다."""
    entry: dict[str, Any] = {"event": event, "at": now_iso()}
    if isinstance(action, dict):
        entry.update(action_id=action.get("action_id"), type=action.get("type"), source=action.get("source"))
    entry.update(extra)
    return entry


# %% 그래프 만들기
# 핵심 요약: 상담·실행 함수와 색인을 주입받아 노드를 만들고, 조건 분기로 연결한 뒤 체크포인터와 함께 compile한다.
# B 함수와 실행 함수는 아직 없으므로 기본값은 None이다. None이면 해당 노드가 '준비 중'이라고 정직하게 안내한다.
def build_graph(*, a_runner: Callable = run_outing_decision, a_device: str | None = None, a_index: Any = None,
                b_runner: Callable | None = None, b_device: str | None = None, b_index: Any = None,
                a_executor: Callable | None = None, b_executor: Callable | None = None, checkpointer: Any = None):
    """컴파일된 SafeMate 그래프를 돌려준다. 같은 그래프 객체를 모든 사용자(thread_id)가 함께 쓴다."""
    executors = {"A": a_executor, "B": b_executor}

    # %% 1) intake 노드
    # 핵심 요약: 이번 턴 칸을 비우고, 입력 검사와 '상담 끝내기'를 처리한다. 상담 함수는 부르지 않는다.
    def intake(state: SafeMateState) -> dict:
        """입력이 바로 끝낼 턴이면 feature="system"과 answer를 정한다."""
        updates: dict[str, Any] = dict(TURN_RESET)
        message = state.get("user_message")
        text = message.strip() if isinstance(message, str) else ""

        # 1. 승인 대기가 남은 채 새 입력이 들어오면(앱 함수를 거치지 않은 경우) 실행하지 않고 취소한다.
        leftover = state.get("pending_action")
        if leftover:
            updates["pending_action"] = None
            updates["action_log"] = [log_entry("cancelled_by_new_input", leftover)]

        # 2. 상담 끝내기: 버튼이거나, B 상담 중에 '그만'·'취소'·'처음으로'만 말한 경우. A 상태는 건드리지 않는다.
        if state.get("control") == "end_consultation" or (
                state.get("active_feature") == "B" and text.rstrip(".!?~ ") in END_WORDS):
            was_active = state.get("active_feature") == "B"
            updates.update(feature="system", active_feature=None, b_conversation_state=None,
                           answer=CONSULTATION_ENDED if was_active else NO_CONSULTATION)
            updates["action_log"] = updates.get("action_log", []) + [log_entry("consultation_ended", active=was_active)]
            return updates

        # 3. 입력 검사: 빈 문장·너무 긴 문장은 상담 함수를 부르지 않는다.
        if not text:
            updates.update(feature="system", answer=EMPTY_MESSAGE)
        elif len(text) > MAX_MESSAGE_LENGTH:
            updates.update(feature="system", answer=TOO_LONG_MESSAGE)
        return updates

    # %% 2) a_node: A기능(외출·날씨) 상담
    # 핵심 요약: run_outing_decision()을 한 번 부르고, 돌려받은 A 상태를 그대로 저장한다. 제안이 계약을 지키면 봉투로 감싼다.
    def a_node(state: SafeMateState) -> dict:
        """A 결과를 State 갱신분으로 바꾼다. 승인 전이므로 일정은 바꾸지 않는다."""
        previous = state.get("a_conversation_state")
        message = state.get("user_message", "")    # intake가 이미 검사한 문장

        # 1. A 공개 함수 호출. A는 예외를 던지지 않도록 만들어졌지만, 혹시 나면 이전 상태를 지키고 오류로 안내한다.
        try:
            raw = a_runner(message, device=a_device, index=a_index, conversation_state=previous)
        except Exception as exc:  # noqa: BLE001 - 어떤 오류든 화면이 멈추지 않게 한다
            failed = {"status": "runner_exception", "route": "error", "message": f"{type(exc).__name__}: {exc}"[:300]}
            return {"feature": "A", "last_result": failed, "view": to_common_view(failed, "A"), "answer": SAFE_ANSWER,
                    "pending_action": None, "action_log": [log_entry("feature_exception", feature="A")]}
        problems = check_feature_result(raw, "A")
        result: dict[str, Any] = json_safe(raw) if isinstance(raw, dict) else {"status": "invalid_result", "route": "error"}

        # 2. A가 돌려준 대화 상태는 오류 턴이어도 그대로 저장한다. (어느 상태를 줄지는 A가 이미 정함)
        updates: dict[str, Any] = {"feature": "A", "last_result": result, "view": to_common_view(result, "A"),
                                   "pending_action": None}
        if isinstance(result.get("conversation_state"), dict):
            updates["a_conversation_state"] = result["conversation_state"]
        answer = result.get("answer")
        updates["answer"] = answer if isinstance(answer, str) and answer.strip() else SAFE_ANSWER

        # 3. 계약 위반이면 기록만 하고 승인 카드는 만들지 않는다. 지키면 제안을 공통 봉투로 감싼다. (action_id 1회 생성)
        if problems:
            updates["action_log"] = [log_entry("contract_violation", feature="A", problems=problems)]
        else:
            proposal = result.get("proposed_action")
            if isinstance(proposal, dict):
                updates["pending_action"] = wrap_outing_proposal(proposal, now_iso())

        # 4. 생활 지원 요청이면 B로 넘길 context만 만든다. (A 대화 상태는 넘기지 않음)
        if result.get("route") == "living_support":
            llm = result.get("llm")
            understanding = llm.get("understanding") if isinstance(llm, dict) else None
            goal = understanding.get("user_goal") if isinstance(understanding, dict) else None
            updates["handoff_context"] = {"source": "outing_decision", "user_goal": goal if isinstance(goal, str) else ""}
        return updates

    # %% 3) b_node: B기능(생활 지원) 상담
    # 핵심 요약: 주입된 run_living_support를 부른다. 없으면 '준비 중' 안내만 하고 승인 카드·성공 문장을 만들지 않는다.
    def b_node(state: SafeMateState) -> dict:
        """B 결과를 State 갱신분으로 바꾼다. B 제안 봉투는 B가 만든 것을 검사만 하고 그대로 쓴다."""
        # 1. B 미연결: 가짜 결과를 만들지 않는다.
        if b_runner is None:
            missing = {"status": "feature_unavailable", "route": "error", "message": "B기능 함수가 연결되지 않았습니다."}
            return {"feature": "B", "last_result": missing, "view": to_common_view(missing, "B"), "answer": B_UNAVAILABLE,
                    "active_feature": None, "pending_action": None}

        # 2. B 공개 함수 호출 (명세 2.2). B 상태와 B 색인만 넘긴다.
        request = {"user_message": state.get("user_message", ""), "context": state.get("handoff_context")}
        try:
            raw = b_runner(request, device=b_device, index=b_index,
                              conversation_state=state.get("b_conversation_state"))
        except Exception as exc:  # noqa: BLE001
            failed = {"status": "runner_exception", "route": "error", "message": f"{type(exc).__name__}: {exc}"[:300]}
            return {"feature": "B", "last_result": failed, "view": to_common_view(failed, "B"), "answer": SAFE_ANSWER,
                    "active_feature": None, "pending_action": None,
                    "action_log": [log_entry("feature_exception", feature="B")]}
        problems = check_feature_result(raw, "B")
        result: dict[str, Any] = json_safe(raw) if isinstance(raw, dict) else {"status": "invalid_result", "route": "error"}

        # 3. B 상태 저장과 '추가 질문 중' 표시. 다음 턴의 짧은 대답은 A를 거치지 않고 B로 간다.
        asking = result.get("route") == "needs_clarification" and result.get("status") == "needs_input"
        updates: dict[str, Any] = {"feature": "B", "last_result": result, "view": to_common_view(result, "B"),
                                   "active_feature": "B" if asking else None, "pending_action": None}
        if isinstance(result.get("conversation_state"), dict):
            updates["b_conversation_state"] = result["conversation_state"]
        answer = result.get("answer")
        updates["answer"] = answer if isinstance(answer, str) and answer.strip() else SAFE_ANSWER

        # 4. 계약을 지킨 제안만 승인 대기로 올린다.
        if problems:
            updates["action_log"] = [log_entry("contract_violation", feature="B", problems=problems)]
        else:
            proposal = result.get("proposed_action")
            if isinstance(proposal, dict):
                updates["pending_action"] = copy.deepcopy(proposal)
        return updates

    # %% 4) approval 노드: 사용자 승인 대기
    # 핵심 요약: 맨 앞에서 interrupt()로 멈춘다. 재개하면 이 노드가 처음부터 다시 실행되므로, 앞에는 읽기만 둔다.
    # 실행은 하지 않고 버튼 결과를 정리해 State에 남기기만 한다.
    def approval(state: SafeMateState) -> dict:
        """버튼 결과를 approval_decision으로 돌려준다. 무시한 입력은 기록만 남긴다."""
        pending = state.get("pending_action")
        response = interrupt(approval_card(pending or {}))
        decision = read_decision(response, pending)
        updates: dict[str, Any] = {"approval_decision": decision}
        if decision["decision"] == "ignored":
            updates["action_log"] = [log_entry("ignored_response", pending, reason=decision["reason"])]
        return updates

    # %% 5) execute 노드: 승인 후 실행 (부작용은 여기에만)
    # 핵심 요약: 실행 직전에 다시 검사하고, 승인 정보를 붙인 사본으로 기능별 실행 함수를 부른다. 성공이 분명할 때만 성공으로 표시한다.
    def execute(state: SafeMateState) -> dict:
        """execution_result·answer·기록을 돌려주고 승인 대기를 끝낸다."""
        pending = state.get("pending_action")
        decision = state.get("approval_decision")

        # 1. 실행 직전 재검사
        problem = check_before_execute(pending, decision)
        if problem:
            return {"pending_action": None, "answer": EXECUTION_BLOCKED,
                    "execution_result": {"status": "blocked", "reason": problem, "applied": False},
                    "action_log": [log_entry("execution_blocked", pending, reason=problem)]}
        assert pending is not None and decision is not None
        action_id = pending["action_id"]

        # 2. 같은 제안을 이 대화에서 이미 실행했으면 다시 부르지 않는다. (그래프 단계 중복 방지)
        if action_id in (state.get("executed_action_ids") or []):
            return {"pending_action": None, "answer": ALREADY_DONE,
                    "execution_result": {"status": "already_executed_in_graph", "action_id": action_id, "applied": True},
                    "action_log": [log_entry("duplicate_skipped", pending)]}

        # 3. 실행 함수가 없으면 아무것도 바뀌지 않았다고 안내한다.
        executor = executors[ALLOWED_ACTIONS[pending["type"]]["feature"]]
        if executor is None:
            return {"pending_action": None, "answer": NOT_CONNECTED,
                    "execution_result": {"status": "not_connected", "action_id": action_id, "applied": False},
                    "action_log": [log_entry("not_connected", pending)]}

        # 4. 승인 정보는 그래프만 붙인다. 원본 제안은 고치지 않고 사본에 붙인다. (명세 2.3)
        approved = copy.deepcopy(pending)
        approved["approval"] = {"status": "approved", "at": decision["at"]}
        try:
            output = executor(approved)
        except Exception as exc:  # noqa: BLE001
            output = {"status": "error", "message": f"{type(exc).__name__}: {exc}"[:300], "applied": False}
        output = json_safe(output) if isinstance(output, dict) else {"status": "invalid_return", "applied": False}

        # 5. 성공 판정: 애매하면 실패. 실패는 executed_action_ids에 넣지 않는다.
        if is_execution_success(output, action_id):
            answer = output.get("answer")
            return {"pending_action": None, "execution_result": output,
                    "answer": answer if isinstance(answer, str) and answer.strip() else DONE_DEFAULT,
                    "executed_action_ids": [action_id],
                    "action_log": [log_entry("executed", pending, status=output.get("status"))]}
        return {"pending_action": None, "execution_result": output, "answer": EXECUTION_FAILED,
                "action_log": [log_entry("execution_failed", pending, status=output.get("status"))]}

    # %% 6) close 노드: 거절·취소
    # 핵심 요약: 아무것도 실행하지 않고 승인 대기를 끝낸다. A 제안을 거절하면 기존 시각 유지와 앞서 안내한 공식 수칙을 다시 보여 준다.
    def close(state: SafeMateState) -> dict:
        """거절이면 안내 문장, 새 질문 때문에 취소면 기록만 남긴다."""
        pending = state.get("pending_action")
        kind = (state.get("approval_decision") or {}).get("decision")
        if not pending:
            return {"pending_action": None}

        # 1. 새 질문으로 취소된 경우: 화면 문장은 다음 질문의 답이 덮어쓴다.
        if kind == "cancel":
            return {"pending_action": None, "answer": CANCELLED, "action_log": [log_entry("cancelled", pending)]}

        # 2. 거절: 기존 일정을 유지하고, A 제안이면 이미 검색한 공식 안전 안내 한 줄을 다시 보여 준다. (새로 지어내지 않음)
        answer = REJECTED
        if pending.get("type") == "propose_outing_time_change":
            answer += " " + KEEP_OUTING
            guidance = [g for g in ((state.get("last_result") or {}).get("official_guidance") or [])
                        if isinstance(g, dict) and g.get("text")]
            if guidance:
                answer += f"\n[공식 안전 안내] {guidance[0]['text']} ({guidance[0].get('publisher') or '출처 확인'})"
        return {"pending_action": None, "answer": answer, "action_log": [log_entry("rejected", pending)]}

    # %% 분기 함수 (LLM 없이 State만 보고 다음 노드를 고른다)
    def after_intake(state: SafeMateState) -> str:
        """바로 끝낼 턴이면 END, B가 질문 중이면 b_node, 그 밖은 a_node"""
        if state.get("feature") == "system":
            return END
        return "b_node" if state.get("active_feature") == "B" else "a_node"

    def after_feature(state: SafeMateState) -> str:
        """승인할 제안이 있으면 approval, A가 생활 지원으로 이해했으면 b_node, 그 밖은 END"""
        if state.get("pending_action"):
            return "approval"
        if state.get("feature") == "A" and (state.get("last_result") or {}).get("route") == "living_support":
            return "b_node"
        return END

    def after_approval(state: SafeMateState) -> str:
        """승인 → execute, 거절·취소 → close, 무시 → approval(다시 대기)"""
        decision = (state.get("approval_decision") or {}).get("decision")
        if decision == "approve":
            return "execute"
        if decision == "ignored" and state.get("pending_action"):
            return "approval"
        return "close"

    # %% 배선: 노드 추가 → 시작 엣지 → 조건 엣지 → 끝 엣지 → compile
    builder = StateGraph(SafeMateState)
    for name, node in (("intake", intake), ("a_node", a_node), ("b_node", b_node),
                       ("approval", approval), ("execute", execute), ("close", close)):
        builder.add_node(name, node)
    builder.add_edge(START, "intake")
    builder.add_conditional_edges("intake", after_intake, ["a_node", "b_node", END])
    builder.add_conditional_edges("a_node", after_feature, ["approval", "b_node", END])
    builder.add_conditional_edges("b_node", after_feature, ["approval", END])
    builder.add_conditional_edges("approval", after_approval, ["execute", "close", "approval"])
    builder.add_edge("execute", END)
    builder.add_edge("close", END)
    return builder.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


# %% 앱 함수: Gradio(6단계)가 그래프 내부를 몰라도 쓰도록 하는 얇은 함수
# 핵심 요약: thread_id 하나가 한 사람의 대화방이다. 버튼 입력은 여기서 1차로 검사하고, 통과한 것만 Command(resume=...)로 넘긴다.
def new_thread_id() -> str:
    """세션마다 새 대화방 이름 (무작위 값, 개인정보 아님)"""
    return uuid.uuid4().hex


def run_config(thread_id: str) -> RunnableConfig:
    """thread_id와 반복 상한을 담은 실행 설정"""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


def waiting_action(graph, thread_id: str) -> dict | None:
    """approval 노드에서 멈춰 있으면 승인 대기 중인 제안을, 아니면 None을 돌려준다."""
    snapshot = graph.get_state(run_config(thread_id))
    if snapshot.next == ("approval",):
        return snapshot.values.get("pending_action") or {}
    return None


def turn_view(graph, thread_id: str, notice: str | None = None) -> dict:
    """화면이 그릴 값: 답변, 승인 카드(대기 중일 때만), B 상담 중 여부, 근거, 경고, 개발자용 정보"""
    values = graph.get_state(run_config(thread_id)).values
    pending = waiting_action(graph, thread_id)
    view = values.get("view") or {}
    last = values.get("last_result") or {}
    return {"answer": notice or values.get("answer") or "",
            "pending": approval_card(pending) if pending else None,
            "active_feature": values.get("active_feature"),
            "evidence": view.get("evidence", []), "warning": bool(view.get("warning")),
            "notice": notice,
            "debug": {"feature": values.get("feature"), "route": last.get("route"), "status": last.get("status"),
                      "detail_code": last.get("detail_code"),
                      "answer_source": (last.get("decision") or {}).get("answer_source"),
                      "tool_trace": last.get("tool_trace", []), "execution_result": values.get("execution_result"),
                      "action_log": (values.get("action_log") or [])[-5:]}}


def _invoke(graph, thread_id: str, payload) -> str | None:
    """그래프를 실행한다. 반복 상한·예상하지 못한 오류는 화면 문장으로 바꾼다."""
    try:
        graph.invoke(payload, run_config(thread_id))
    except GraphRecursionError:
        return "처리 단계가 너무 많아 멈췄어요. 다시 말씀해 주세요."
    except Exception as exc:  # noqa: BLE001
        print(f"[safemate_graph] 그래프 실행 오류: {type(exc).__name__}: {exc}", file=sys.stderr)
        return SAFE_ANSWER
    return None


def _cancel_waiting(graph, thread_id: str) -> bool:
    """승인 대기 중이면 그 제안을 cancel로 끝낸다. 취소했으면 True"""
    pending = waiting_action(graph, thread_id)
    if pending is None:
        return False
    _invoke(graph, thread_id, Command(resume={"decision": "cancel", "action_id": pending.get("action_id")}))
    return True


def send_message(graph, thread_id: str, text: str) -> dict:
    """사용자 발화 처리. 승인 대기 중이면 이전 제안을 실행하지 않고 취소한 뒤 새 발화를 처리한다."""
    # 1. 승인 대기 중의 새 질문은 이전 제안의 취소로 본다. (명세 3.2: 제안은 다음 한 턴에만 유효)
    cancelled = _cancel_waiting(graph, thread_id)
    # 2. 새 발화를 처음부터 처리한다.
    error = _invoke(graph, thread_id, {"user_message": text, "control": None})
    view = turn_view(graph, thread_id, notice=error)
    if cancelled and not error:
        view["notice"] = CANCELLED
    return view


def press_button(graph, thread_id: str, decision: str, action_id: str) -> dict:
    """승인·거절 버튼 처리. 승인 대기가 아니거나 오래된 버튼이면 그래프를 재개하지 않는다."""
    # 1. 승인 대기 중이 아니면 실행 금지
    pending = waiting_action(graph, thread_id)
    if pending is None:
        return turn_view(graph, thread_id, notice=NO_PENDING)
    # 2. 버튼 값 형식 확인
    if decision not in BUTTON_DECISIONS:
        return turn_view(graph, thread_id, notice=INVALID_BUTTON)
    # 3. 오래된 버튼은 무시하고 현재 승인 대기를 그대로 둔다.
    if action_id != pending.get("action_id"):
        return turn_view(graph, thread_id, notice=STALE_BUTTON)
    # 4. 통과한 버튼만 Command(resume=...)로 재개한다.
    error = _invoke(graph, thread_id, Command(resume={"decision": decision, "action_id": action_id}))
    return turn_view(graph, thread_id, notice=error)


def end_consultation(graph, thread_id: str) -> dict:
    """'상담 끝내기' 버튼. B 상담 상태만 비우고 A 상태는 그대로 둔다."""
    _cancel_waiting(graph, thread_id)
    error = _invoke(graph, thread_id, {"user_message": "", "control": "end_consultation"})
    return turn_view(graph, thread_id, notice=error)


# %% 독립 실행: 그래프 구조(Mermaid) 출력
# 핵심 요약: 발표 자료용 그래프 그림을 확인한다. 상담 함수는 실행하지 않는다.
if __name__ == "__main__":
    print(build_graph().get_graph().draw_mermaid())
