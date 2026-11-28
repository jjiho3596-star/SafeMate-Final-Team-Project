"""SafeMate 5단계: 일정 상담 에이전트

핵심 요약
- 어르신의 일정 관련 말을 LLM(qwen2.5:3b)이 한 번 읽고, '무슨 작업인지·어떤 일정인지·사용자가 말한 표현'만 JSON으로 정리한다.
- 날짜·시각 계산, 대상 일정 확인, 충돌·빈 시간, 예보 비교, 승인 봉투 만들기는 schedule_service(Python)가 맡는다.
- 실외 일정의 시각을 정할 때만 1단계 예보를 조회하고, 위험 신호가 있을 때만 2단계 공식 안전수칙을 검색한다.
- 반환값은 4단계 그래프가 읽는 A기능 결과와 같은 모양의 딕셔너리다.
  승인용 제안(proposed_action)은 만들기만 한다. 저장은 그래프가 사용자 승인을 확인한 뒤 execute 노드에서만 한다.

역할 분담
- LLM: 작업 종류(operation), 요청 방식(request_mode), 사용자 표현 옮겨 적기, (시간 선택을 맡긴 경우) 후보 중 선택과 이유 코드
- Python: 표현이 발화에 실제로 있는지 확인, 날짜·시각 계산, 일정 존재 확인, 충돌·보호, 예보 수치 비교, 안내 문장, 제안 봉투
"""

# %% 공통 라이브러리와 기존 단계 함수 불러오기
# 핵심 요약: LLM 호출·예보·RAG는 3단계 함수를, 일정 계산·제안·저장 검사는 5-1·5-2단계 함수를 그대로 쓴다.
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

import outing_decision as od
import schedule_service as sv
import schedule_store as ss
from a_outing import AMBIGUOUS_CITY_NAMES, FORECAST_DAYS, KST


# %% 일정 상담 설정값
# 핵심 요약: 일정 이해 출력의 허용값, 한 턴의 조회 상한, 화면 문장을 한곳에 모은다.
STATE_VERSION = 1
HISTORY_LIMIT = 6              # LLM에 넘기는 직전 대화 줄 수
CONTEXT_SCHEDULE_LIMIT = 20    # LLM에 넘기는 저장된 일정 수 (id만 고를 수 있게 함)
HINT_LIMIT = 8                 # 3단계 이해 단계에 넘기는 일정 요약 수 (3단계 SCHEDULE_HINT_LIMIT와 같음)
MAX_WEATHER_CALLS = 4          # 원래 시각 1 + 후보 최대 3
LIST_LIMIT = 5                 # 후보 목록으로 보여 줄 일정 수
TEXT_LIMIT = 120

OPERATIONS = ("query", "create", "update", "cancel", "complete", "other")
CHANGE_OPERATIONS = ("create", "update", "cancel", "complete")
QUERY_SCOPES = ("date", "target", "next", "remaining", "week", "history", "summary", "none")
REQUEST_MODES = ("request", "negated", "conditional", "question", "correction")
SHIFT_BASES = ("schedule", "now", "unknown")
PREFERENCES = ("avoid_rain", "avoid_heat", "avoid_cold", "avoid_wind")
PREFERENCE_TOPICS = {"avoid_rain": "호우", "avoid_heat": "폭염", "avoid_cold": "한파", "avoid_wind": "강풍"}
PREFERENCE_WORDS = {"avoid_rain": "비", "avoid_heat": "더위", "avoid_cold": "추위", "avoid_wind": "강한 바람"}
CATEGORY_GUESSES = ss.CATEGORIES + ("unknown",)
CORRECTS_FIELDS = ("date", "time", "target", "none")
TEXT_FIELDS = ("target_text", "date_text", "time_text", "new_date_text", "new_time_text", "shift_text",
               "duration_text", "title", "city_text", "reason_text")
FLAG_FIELDS = ("delegate_choice", "other_day_ok", "check_weather", "multiple_requests")
# 승인 계획(6절)·인수인계 9절에서 확정한 일정 이해 스키마의 키 순서
SCHEMA_KEYS = ("operation", "query_scope", "answers_previous_question", "target_ids", *TEXT_FIELDS, "shift_base",
               *FLAG_FIELDS, "preferences", "category_guess", "request_mode", "exclude_previous_target", "corrects_field")
CHOICE_REASONS = ("less_rain", "lower_wind", "milder_temp", "no_conflict", "closer_to_original")
ROUTES = ("schedule", "needs_clarification", "error", "handoff")
CONTEXT_MODES = ("continue", "handoff", "recheck")

ASK_SUFFIX = "승인 버튼을 누르시면 저장하고, 거절하시면 그대로 둘게요."
HANDOFF_TEXT = "일정 관리가 아닌 말씀으로 이해했어요."
NO_GUARANTEE = "(예보는 조회한 1시간 구간의 값이며, 안전을 보장한다는 뜻은 아니에요.)"
B_NOT_CONNECTED = "보호자 도움 요청 기록은 생활 지원 기능이 아직 연결되지 않아 확인할 수 없어요."


# %% 일정 상담 대화 상태
# 핵심 요약: A·B 상태와 따로 쓰는 S(일정) 전용 상태. 진행 중인 작업 초안(draft)과 직전 대화만 담고, 일정 자체는 담지 않는다.
# 일정은 매 턴 저장소에서 다시 읽는다. (그래프 State에 일정을 캐시하면 저장 뒤 오래된 시각을 말할 수 있기 때문)
def new_schedule_state() -> dict[str, Any]:
    """빈 일정 대화 상태"""
    return {"version": STATE_VERSION, "turn": 0, "history": [], "draft": None}


def new_draft(operation: str) -> dict[str, Any]:
    """진행 중인 일정 작업 초안. 사용자가 말한 값과 Python이 계산한 값만 담는다."""
    return {"operation": operation, "conditional": False, "awaiting": None, "question": "", "candidates": [],
            "target_id": None, "excluded_ids": [], "query": {"text": "", "date": None, "time": None, "period": None},
            "date": None, "time": None, "shift_text": "", "shift_base": None, "minutes": None, "title": "",
            "city": None, "category": None, "reason": "", "delegate": False, "preferences": [], "other_day_ok": False,
            "check_weather": False, "pending": {},
            # (5-6 D34·D42·D43 보완) 확인 질문 상태: 물어본 날씨 조건, 명시적으로 철회한 조건, 질문 횟수, 위임 확인을 거절했는지
            "asked": [], "waived": [], "confirm_tries": 0, "delegate_declined": False}


def _clean_draft(raw: Any) -> dict[str, Any] | None:
    """형식이 맞는 초안만 남긴다. 알 수 없는 키·잘못된 자료형은 기본값으로 둔다. (그래프 State에서 받은 값을 그대로 믿지 않음)"""
    if not isinstance(raw, dict) or raw.get("operation") not in CHANGE_OPERATIONS + ("query",):
        return None
    draft = new_draft(raw["operation"])
    for key, default in draft.items():
        value = raw.get(key, default)
        if key in ("candidates", "excluded_ids", "preferences"):
            draft[key] = [v for v in value if isinstance(v, str)] if isinstance(value, list) else []
        elif key in ("asked", "waived"):
            draft[key] = [v for v in value if v in PREFERENCES] if isinstance(value, list) else []
        elif key == "confirm_tries":
            draft[key] = value if type(value) is int and 0 <= value <= 9 else 0
        elif key == "query":
            query = value if isinstance(value, dict) else {}
            draft[key] = {k: query.get(k) if isinstance(query.get(k), str) else d for k, d in default.items()}
        elif key == "pending":
            # 대상을 묻는 동안 보관한 새 시각 표현('5시로')이나 A기능이 제안한 시각. 문자열만 남긴다.
            draft[key] = {k: v for k, v in value.items() if isinstance(k, str) and isinstance(v, str)} \
                if isinstance(value, dict) else {}
        elif isinstance(default, bool):
            draft[key] = value if isinstance(value, bool) else default
        elif key == "minutes":
            draft[key] = value if type(value) is int else None
        elif isinstance(value, str) or value is None:
            draft[key] = value if isinstance(value, str) or default is None else default
    return draft


def load_schedule_state(state: Any) -> dict[str, Any]:
    """그래프가 넘긴 S 상태를 검사한다. 형식이 다르면 새 상태로 시작한다. (A·B 상태를 실수로 넘겨도 섞이지 않음)"""
    if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
        return new_schedule_state()
    history = [h for h in state.get("history", []) if isinstance(h, dict) and isinstance(h.get("text"), str)
               and h.get("role") in ("user", "assistant")] if isinstance(state.get("history"), list) else []
    turn = state.get("turn")
    return {"version": STATE_VERSION, "turn": turn if type(turn) is int and turn >= 0 else 0,
            "history": history[-HISTORY_LIMIT:], "draft": _clean_draft(state.get("draft"))}


def has_draft(state: Any) -> bool:
    """진행 중인 일정 작업이 있는지 (그래프가 active_feature="S"를 정할 때 씀)"""
    return isinstance(state, dict) and _clean_draft(state.get("draft")) is not None


def clear_draft(state: Any) -> dict[str, Any]:
    """저장 성공·거절·상담 끝내기 뒤에 초안만 비운다. 직전 대화는 남긴다."""
    cleaned = load_schedule_state(state)
    cleaned["draft"] = None
    return cleaned


# %% 저장된 일정 요약 함수
# 핵심 요약: 3단계 이해 단계(일정 의도 분류 보조)와 일정 이해 LLM에 넘길 일정 목록을 만든다. 개인정보는 없다.
def schedule_hint(doc: dict[str, Any], now: datetime) -> list[dict[str, str]]:
    """3단계 run_outing_decision_with_schedule(schedule_hint=...)용: 지금 이후 끝나는 예정 일정의 제목·날짜·시각 (최대 8개)"""
    items = [s for s in doc["schedules"] if s["status"] == "scheduled" and sv.interval(s)[1] > now]
    items.sort(key=lambda s: s["start_at"])
    return [{"title": s["title"], "date": sv.interval(s)[0].date().isoformat(), "time": f"{sv.interval(s)[0]:%H:%M}"}
            for s in items[:HINT_LIMIT]]


def hint_from_store(store: ss.ScheduleStore, now: datetime) -> list[dict[str, str]] | None:
    """저장소를 읽어 요약을 만든다. 저장소를 읽지 못하면 None (3단계는 요약 없이 기존처럼 동작)"""
    try:
        return schedule_hint(store.load(), now)
    except ss.ScheduleStoreError:
        return None


def context_schedules(doc: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
    """일정 이해 LLM에 넘길 일정: 어제부터 31일 뒤까지 걸친 일정 (최대 20개). id는 고르기에만 쓴다."""
    first, last = now.date() - timedelta(days=1), now.date() + timedelta(days=31)
    return sv.schedules_between(doc, first, last)[:CONTEXT_SCHEDULE_LIMIT]


# %% 일정 이해 LLM 프롬프트와 출력 규격
# 핵심 요약: LLM은 답변 문장을 쓰지 않고, 날짜·시각을 계산하지도 않는다. 사용자가 말한 표현을 그대로 옮겨 적기만 한다.
SCHEDULE_PROMPT = """당신은 SafeMate의 '일정 이해 단계'입니다. 혼자 생활하는 어르신의 한국어 발화를 읽고
일정 요청의 뜻을 JSON으로만 정리합니다. 답변 문장을 쓰지 말고, 날짜·시각을 계산하지 마세요.

규칙
1. operation: query(일정 조회·기억 확인·다음 일정·남은 일정·변경 기록·하루 정리), create(새 일정 등록, "기억해 줘"),
   update(시간·날짜 변경), cancel(취소, "안 갈래"), complete(다녀왔음·끝냈음 기록),
   other(일정과 관계없는 말: 날씨만 묻기, 생활 지원("우유가 떨어졌어"), 안전 수칙, 일상 대화)
2. request_mode: request(해 달라는 요청), negated(하지 말라는 말: "바꾸지 마", "바꿀 필요 없어", "원래 시간에 갈래"),
   conditional(조건부: "비가 오면 바꿀까?"), question(물어보기만 함: "바꿀 수 있어?"), correction(앞서 말한 것을 고침: "~ 말고 ~", "~ 아니고 ~")
3. query_scope: date(그날 일정), target(특정 일정의 시각), next(다음 일정), remaining(오늘 남은 일정), week(이번 주),
   history(바꾼·등록한 기록, 원래 시각), summary(오늘 한 일 정리), none(조회가 아님)
4. 텍스트 칸에는 '사용자_발화'에 실제로 있는 표현만 그대로 옮겨 적고, 없으면 반드시 ""로 둡니다. 숫자로 바꾸지 마세요.
   '등록된_일정'이나 '직전_대화'의 글자를 텍스트 칸에 옮겨 적지 마세요. 대부분의 칸은 ""입니다.
   target_text(어떤 일정인지), date_text·time_text(찾는 일정 또는 새 일정의 날짜·시각), new_date_text·new_time_text(바꿀 날짜·시각),
   shift_text("한 시간 늦게", "30분 앞당겨"), duration_text(걸리는 시간), title(새 일정 이름), city_text(지역), reason_text(이유)
5. target_ids: '등록된_일정'의 id 중 사용자가 가리킨 일정. 확실하지 않으면 []로 둡니다.
6. shift_base: shift_text가 원래 일정 시각 기준이면 schedule, "지금부터"면 now, 알 수 없으면 unknown
7. delegate_choice: "알아서", "아무 때나", "괜찮은 시간으로"처럼 시간 선택을 맡기면 true. 저장 승인이라는 뜻은 아닙니다.
8. preferences: 비를 피하려면 avoid_rain, 더위 avoid_heat, 추위 avoid_cold, 바람 avoid_wind. check_weather: 날씨를 확인해 달라는 뜻이면 true
9. '진행_중인_상담'이 있고 이번 말이 그 질문의 대답이면 answers_previous_question=true로 두고 operation은 진행 중인 작업과 같게 둡니다.
10. exclude_previous_target: "그거 말고 다른 일정"처럼 앞서 고른 일정이 아니라고 하면 true. corrects_field: 고친 항목(date, time, target), 없으면 none
11. multiple_requests: 한 문장에 일정 작업을 두 개 이상 요청하면 true. category_guess: 새 일정이 바깥 외출이면 outing, 기관 방문·약속이면 appointment, 집안일이면 home

예시
- "내일 시장 가는 시간 좀 바꿔 줘" → operation=update, request_mode=request, target_text="시장 가는", date_text="내일"
- "오후 네 시쯤" (몇 시로 바꿀지 물은 뒤) → operation=update, answers_previous_question=true, new_time_text="오후 네 시쯤"
- "내일 일정 바꾸지 마" → operation=update, request_mode=negated, date_text="내일"
- "비가 오면 시간을 바꿀까?" → operation=update, request_mode=conditional, preferences=["avoid_rain"], check_weather=true
- "저녁 일곱 시 아니고 여덟 시야" → operation=update, request_mode=correction, corrects_field=time, new_time_text="여덟 시"
- "아까 말한 일정 말고 다른 일정이야" → request_mode=correction, corrects_field=target, exclude_previous_target=true
- "비 안 오는 시간으로 알아서 바꿔 줘" → operation=update, delegate_choice=true, preferences=["avoid_rain"]
- "내가 내일 뭐 하기로 했지?" → operation=query, query_scope=date, date_text="내일"
- "내일 아침 여덟 시에 산책할 거야. 기억해 줘." → operation=create, title="산책", date_text="내일", time_text="아침 여덟 시", category_guess=outing
- "오늘 시장 다녀왔어" → operation=complete, target_text="시장", date_text="오늘"
- "오늘 날씨 어때?", "우유가 떨어졌어" → operation=other"""

CHOICE_PROMPT = """당신은 SafeMate의 '시간 선택 단계'입니다. Python이 계산한 후보 시각 중에서 하나를 고르고 이유 코드를 고릅니다.
후보에 없는 시각을 만들지 마세요. 예보_상태가 unknown인 후보를 날씨가 좋다는 근거로 쓰지 마세요.
사용자가 피하고 싶은 날씨가 있는 후보는 고르지 마세요. 알맞은 후보가 없으면 choice="none"입니다.
이유 코드: less_rain(원래 시각보다 비 예보가 적음), lower_wind(바람이 약함), milder_temp(기온이 덜 덥거나 덜 추움),
no_conflict(다른 일정과 겹치지 않음), closer_to_original(원래 시각과 가장 가까움)"""


def schedule_schema(ids: list[str]) -> dict[str, Any]:
    """일정 이해 JSON 스키마. target_ids는 이번 호출에 넘긴 일정 id만 고를 수 있다."""
    text: dict[str, Any] = {"type": "string"}
    flag: dict[str, Any] = {"type": "boolean"}
    id_items: dict[str, Any] = {"type": "string", "enum": list(ids)} if ids else {"type": "string"}
    properties: dict[str, Any] = {
        "operation": {"type": "string", "enum": list(OPERATIONS)},
        "query_scope": {"type": "string", "enum": list(QUERY_SCOPES)},
        "answers_previous_question": flag,
        "target_ids": {"type": "array", "items": id_items, **({} if ids else {"maxItems": 0})},
        **{key: text for key in TEXT_FIELDS},
        "shift_base": {"type": "string", "enum": list(SHIFT_BASES)},
        **{key: flag for key in FLAG_FIELDS},
        "preferences": {"type": "array", "items": {"type": "string", "enum": list(PREFERENCES)}},
        "category_guess": {"type": "string", "enum": list(CATEGORY_GUESSES)},
        "request_mode": {"type": "string", "enum": list(REQUEST_MODES)},
        "exclude_previous_target": flag,
        "corrects_field": {"type": "string", "enum": list(CORRECTS_FIELDS)},
    }
    return {"type": "object", "properties": properties, "required": list(SCHEMA_KEYS), "additionalProperties": False}


# %% 일정 이해 결과 검사 함수
# 핵심 요약: 키·자료형·허용값을 확인한다. 넘기지 않은 일정 id는 버린다. (존재하는 id라도 '사용자가 골랐다'는 뜻으로 쓰지 않음)
def validate_schedule_understanding(data: Any, ids: list[str]) -> tuple[dict[str, Any], list[str]]:
    """(검사한 이해 결과, 버린 id 목록). 형식이 틀리면 ValueError (호출하는 쪽이 한 번 다시 요청)"""
    # 1. 키가 규격과 정확히 같아야 한다.
    if not isinstance(data, dict) or set(data) != set(SCHEMA_KEYS):
        raise ValueError("키가 규격과 다릅니다")
    enums = {"operation": OPERATIONS, "query_scope": QUERY_SCOPES, "shift_base": SHIFT_BASES,
             "category_guess": CATEGORY_GUESSES, "request_mode": REQUEST_MODES, "corrects_field": CORRECTS_FIELDS}
    for key, allowed in enums.items():
        if data[key] not in allowed:
            raise ValueError(f"{key} 값이 허용 목록에 없습니다")
    # 2. 참·거짓 칸과 문자열 칸의 자료형
    for key in ("answers_previous_question", "exclude_previous_target", *FLAG_FIELDS):
        if type(data[key]) is not bool:
            raise ValueError(f"{key}는 true/false여야 합니다")
    for key in TEXT_FIELDS:
        if not isinstance(data[key], str):
            raise ValueError(f"{key}는 문자열이어야 합니다")
    # 3. 목록 칸: 허용값만 남긴다. 넘기지 않은 일정 id는 버리고 기록한다.
    if not isinstance(data["target_ids"], list) or not isinstance(data["preferences"], list):
        raise ValueError("target_ids·preferences는 목록이어야 합니다")
    cleaned = dict(data)
    cleaned.update({key: data[key].strip()[:TEXT_LIMIT] for key in TEXT_FIELDS})
    dropped = [i for i in data["target_ids"] if not (isinstance(i, str) and i in ids)]
    cleaned["target_ids"] = [i for i in data["target_ids"] if isinstance(i, str) and i in ids]
    cleaned["preferences"] = sorted({p for p in data["preferences"] if p in PREFERENCES})
    return cleaned, [str(i) for i in dropped]


# %% 사용자 발화 근거 확인 함수
# 핵심 요약: LLM이 옮겨 적은 표현이 이번 발화(또는 직전 사용자 발화)에 실제로 있는지 확인하고, 없으면 버린다.
# 3단계 ground_slots와 같은 방식이다. (예: 사용자는 '시장'만 말했는데 모델이 '오후 5시'를 지어내면 버림)
def _squeeze(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


SYSTEM_LABEL_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")


def ground_texts(understanding: dict[str, Any], message: str,
                 history: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """(근거가 확인된 이해 결과, 버린 표현 목록)

    (실측 보완) 근거는 이번 발화에서만 찾는다. 직전 발화까지 인정하면 앞 턴의 '내일'이 새 날짜로 섞여 들어갔다.
    진행 중인 상담의 값은 초안(draft)이 이미 들고 있으므로 직전 발화가 필요 없다.
    (실측 보완) qwen2.5:3b는 '등록된_일정'의 표기('내일(11월 26일 목요일)', '[예정]')를 함께 옮겨 적는다.
    괄호·대괄호 안의 시스템 표기는 떼고 비교하고, 대상 표현은 사용자가 실제로 말한 낱말만 남긴다('시장 장보기' → '시장')."""
    del history     # 시그니처 유지용: 직전 대화는 근거로 쓰지 않는다
    source = _squeeze(message)
    grounded, discarded = dict(understanding), []
    for key in TEXT_FIELDS:
        value = understanding[key]
        if not value:
            continue
        cleaned = SYSTEM_LABEL_RE.sub("", value).strip()
        kept = cleaned if cleaned and _squeeze(cleaned) in source else ""
        if not kept and key == "city_text":
            # (검토 보완) 모델이 지역 이름을 늘려 쓰면('경기도 광주' → '광주광역시') 그 행정구역을 확정하지 않는다.
            # 사용자가 말한 핵심 이름('광주')만 남기고, 모호한 이름이면 1단계 AMBIGUOUS_CITY_NAMES 규칙으로 다시 묻는다(handle_create).
            core = re.sub(r"(특별자치시|특별자치도|특별시|광역시|시|군|구|도)$", "", _squeeze(cleaned))
            kept = core if core and core in source else ""
        if not kept and key == "title":
            # 새 일정 이름은 조사만 바뀌어 옮겨질 수 있다('친구 만나기로 했어' → '친구 만나기'). 낱말 앞 두 글자가 모두 발화에 있어야 한다.
            words = sv.name_words(cleaned)
            kept = cleaned if words and all(word[:2] in source for word in words) else ""
        if not kept and key == "target_text":
            # 대상은 사용자가 말한 낱말만 남긴다. 저장된 제목을 옮겨 적었어도 사용자가 말한 부분('시장')은 근거가 된다.
            kept = " ".join(word for word in sv.name_words(cleaned) if word in source)
        if not kept and key == "shift_text" and shift_paraphrase_ok(cleaned, message, understanding["shift_base"]):
            # (5-6 D39 보완) '늦출게'를 '한 시간 늦게'로 바꿔 옮긴 경우: 발화 자체를 계산한 값·방향과 같고 기준도 모순되지 않을 때만 인정한다.
            kept = cleaned
            discarded.append({"field": key, "value": value, "kept": kept, "paraphrase_verified": True})
            grounded[key] = kept
            continue
        if kept != value:
            grounded[key] = kept
            discarded.append({"field": key, "value": value} if not kept else {"field": key, "value": value, "kept": kept})
    # 사용자가 말한 날씨 선호도 확인한다. (말하지 않은 '비'를 근거로 일정을 옮기지 않기 위함)
    words = {"avoid_rain": r"비|우산|강수|장마|소나기", "avoid_heat": r"더|덥|폭염|뜨거", "avoid_cold": r"추|춥|한파|쌀쌀|눈",
             "avoid_wind": r"바람|강풍"}
    preferences = [p for p in understanding["preferences"] if re.search(words[p], message)]
    for p in sorted(set(understanding["preferences"]) - set(preferences)):
        discarded.append({"field": "preferences", "value": p})
    grounded["preferences"] = preferences
    return grounded, discarded


SHIFT_CHECK_BASE = datetime(2026, 1, 5, 12, 0, tzinfo=KST)     # 값·방향 비교에만 쓰는 고정 기준 (계산 결과를 저장하지 않음)


def shift_paraphrase_ok(value: str, message: str, base: str) -> bool:
    """(5-6 D39 보완) 모델이 바꿔 옮긴 상대 시각이 발화와 같은 뜻인지: resolve_shift로 계산한 분 값·방향이 같고,
    발화에 여러 시간 표현이 섞이지 않았으며(계산 불가·모호하면 불인정), 기준('지금' 유무)이 모순되지 않을 때만 True"""
    said, model = sv.resolve_shift(message, SHIFT_CHECK_BASE), sv.resolve_shift(value, SHIFT_CHECK_BASE)
    if not (said and model and "minutes" in said and "minutes" in model) or said["minutes"] != model["minutes"]:
        return False
    now_said = "지금" in message
    return not ((base == "now" and not now_said) or (base == "schedule" and now_said))


# %% (5-6 D34 보완) 날씨 조건 일관성 확인 함수
# 핵심 요약: 사용자가 말한 날씨 조건(비·더위·추위·바람)을 모델이 이해 결과에서 빠뜨렸는지 원문과 비교한다. 키워드로 의도를 분류하지 않고,
# 'LLM 출력과 원문의 불일치'만 찾는다. 빠뜨렸으면 카드를 만들지 않고 그 조건의 말로 되묻는다(조건을 임의로 채우지 않음).
# 낱말 앞뒤 경계로 '준비·비행기·비빔밥·비밀·더 늦게·추석'의 글자에는 반응하지 않는다.
CONDITION_WORDS = {"avoid_rain": re.compile(r"(?<![가-힣])비(?=[가는를도오와만요랑\s,.?!~]|$)|우산|강수|장마|소나기"),
                   "avoid_heat": re.compile(r"폭염|더위|더운|덥|무더"),
                   "avoid_cold": re.compile(r"한파|추위|추운|춥|쌀쌀|영하|(?<![가-힣])눈(?=[이은을도오와\s,.?!]|$)"),
                   "avoid_wind": re.compile(r"바람|강풍")}
CONDITION_ASK = {"avoid_rain": "비가 오지 않을 때", "avoid_heat": "덥지 않을 때", "avoid_cold": "춥지 않을 때",
                 "avoid_wind": "바람이 강하지 않을 때"}
# 확인 대답의 최소 불일치 검사용 표지 (자연어 전체를 분류하지 않음)
AVOID_MARKS = re.compile(r"피하|피해|싫|안\s*돼|안\s*되|않았으면|말았으면|오면\s*안")
NEGATION_START = re.compile(r"^\s*(아니|아뇨|싫어|하지\s*마)")
# (5-6 D45·D46 보완) 철회 근거는 명시적 무관 표현만 인정한다. '와도·불어도' 같은 양보 표현만으로는 철회하지 않는다(뒤 범위의 판정을 따름).
# '비 없이'·'비 안 오는'·'오지 않'·'~지 마'처럼 조건을 지키려는(회피·부정·금지) 표현은 철회를 막는다. '상관없이·조건 없이'의 '없이'는 제외한다.
INDIFFERENT_MARKS = re.compile(r"상관\s*없|관계\s*없|괜찮|조건\s*(?:없이|은?\s*빼)")
CONCESSIVE_MARKS = re.compile(r"와도|불어도|더워도|추워도|오더라도")
WAIVER_BLOCK_MARKS = re.compile(r"안\s*(?:오|불|덥|춥|더운|추운)|지\s*않|지\s*마|지\s*말|말아|"
                                r"(?<!상관)(?<!상관 )(?<!관계)(?<!관계 )(?<!조건)(?<!조건 )없(?:는|을|이|으면|게)")
QUESTION_END = re.compile(r"\?|(?:까|까요|냐|니|나요|는지)\s*[.!~]*\s*$")
WEATHER_GENERIC = re.compile(r"날씨|조건")
LINK_ONLY = re.compile(r"[\s,·]*(?:[은는이가도]\s*)?(?:이?랑|하고|와|과|이?나|그리고)?[\s,·]*")
# (5-6 D47 보완) 확인 질문의 yes를 인정하지 않는 명확한 불확실·부정 표지 (의미 판단은 확인 단계 LLM이 한다)
UNCERTAIN_MARKS = re.compile(r"글쎄|모르겠|모르지|잘\s*모르|아직\s*모르|애매|생각해\s*볼|고민")
NEGATION_ANY = re.compile(r"하지\s*마|하지\s*말|말아|그만")
CONFIRM_KINDS = ("weather_condition", "delegate_confirm", "cancel_confirm")
CONFIRM_LIMIT = 2              # 최초 질문 + 다시 묻기 1회 (프로젝트 설정)
CONFIRM_ANSWER_LIMIT = 40      # 확인 질문의 대답으로 볼 발화 길이 상한 (프로젝트 설정)


# %% (5-6 F1·F2·F3 보완) 작업 종류·지역 표현과 사용자 원문의 불일치 확인
# 핵심 요약: GPU 종합 검증에서 ① '비 오면 장보기 취소해 줘'를 모델이 변경(update)으로 분류해 위임 확인 → '응' → 이동 카드가 나왔고(F1),
# ② '공원 걷기'의 '공원'을 지역으로 써서 그 '지역' 예보로 조건을 확인했으며(F2), ③ 되물은 지역의 대답을 확정하지 못했는데도
# 지역 없이 카드를 만들었다(F3). 키워드로 의도를 분류하지 않고, LLM 출력과 원문이 명확히 어긋날 때만 묻거나 보류한다.
# - 취소 요청 꼴('취소해·안 갈래')만 본다. '취소돼서'(이유)·'취소하지 마'(부정)·변경 표현이 함께 있는 말은 변경 요청 그대로 둔다.
# - 지역 이름은 광역시·도 이름만 좁게 본다('세 시'·'친구'·'포도'·'서울대병원'에는 반응하지 않음).
CANCEL_REQUEST_MARKS = re.compile(r"취소\s*(?:해|할|하고|하자|부탁)|안\s*갈(?:래|게|\s*거)|안\s*가기로")
CANCEL_BLOCK_MARKS = re.compile(r"취소\s*(?:하지|는\s*말|말고|안\s*해|안\s*할)")
COMPLETE_MARKS = re.compile(r"완료|끝냈|다녀왔|마쳤")
UPDATE_MARKS = re.compile(r"바꿔|바꾸|바꿀|옮겨|옮기|미뤄|미루|당겨|당기|늦춰|늦추|변경")
PROVINCE_NAMES = "경기도|강원도|강원특별자치도|충청북도|충청남도|전라북도|전북특별자치도|전라남도|경상북도|경상남도|제주도|제주특별자치도|제주"
REGION_WORD = re.compile(rf"(?<![가-힣])({od.METRO_NAMES}|{PROVINCE_NAMES})(?=에서|에|은|는|이|가|으로|로|\s|[,.?!]|$)")
CITY_REQUIRED = "city_required"     # 초안 pending: 확정해야 할 지역 표현 (사용자가 말한 지역·모호한 지역)
CITY_TRIES = "city_tries"           # 초안 pending: 지역 질문 횟수 (문자열, 최초 질문 + 다시 묻기 1회)


def said_region(message: str) -> str | None:
    """발화에 낱말로 나온 광역시·도 이름. 없으면 None (작은 시·군 이름은 오탐을 막으려고 보지 않는다)"""
    match = REGION_WORD.search(message or "")
    return match.group(1) if match else None


def region_marked(city: str) -> bool:
    """정상 지역 표지가 있는지: 광역시 이름, 시·군·구·도 접미사(세 글자 이상), 도 이름, 1단계 모호 지역표. 두 단어면 하나라도"""
    for word in re.split(r"[,\s]+", (city or "").strip()):
        if word and (re.fullmatch(od.METRO_NAMES, word) or (len(word) >= 3 and od.CITY_UNIT.search(word))
                     or od.PROVINCE_ANSWER.match(word) or word.lower() in AMBIGUOUS_CITY_NAMES):
            return True
    return False


def drop_name_city(grounded: dict[str, Any], doc: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """(F2) city_text가 이번 발화의 대상·이름이나 등록된 일정 제목 안의 낱말이고 정상 지역 표지가 없으면 지역으로 쓰지 않는다.
    (근거가 확인된 이해 결과, 버린 표현 목록) 정상 지역('인천'·'경기도 광주'·'광주광역시')은 이름 안에 있어도 지우지 않는다."""
    city = grounded.get("city_text") or ""
    if not city or region_marked(city):
        return grounded, []
    key = _squeeze(city)
    names = [grounded.get("target_text") or "", grounded.get("title") or ""] + [str(s.get("title") or "") for s in doc["schedules"]]
    if not any(key in _squeeze(name) for name in names if name):
        return grounded, []
    return dict(grounded, city_text=""), [{"field": "city_text", "value": city, "reason": "schedule_name"}]


def recover_city_text(message: str, *, model_saw_city: bool) -> str | None:
    """(F3) 지역을 물은 뒤의 대답에서 사용자 원문 그대로의 지역 표현을 꺼낸다(3단계 recover_city_answer와 같은 모양 규칙 재사용).
    문장 전체가 지역 이름 모양일 때만: 두 단어는 도 이름·시군구 접미사가 있을 때, 한 단어는 정상 지역 표지가 있거나 모델이 지역을
    알아챈 경우(근거 확인에서 버려진 지역 값이 있음). '글쎄'·'몰라'·'아니'·'그래도' 같은 말은 지역으로 보지 않는다."""
    text = od.CITY_ANSWER_ENDING.sub("", re.sub(r"[\s.!~]+$", "", (message or "").strip())).strip()
    words = [w for w in re.split(r"[,\s]+", text) if w]
    if not words or len(words) > 2 or len(text) > 20 or any(not re.fullmatch(r"[가-힣]{2,10}", w) for w in words):
        return None
    if UNCERTAIN_MARKS.search(text) or NEGATION_START.search(text) or re.match(r"몰라|그래|그러|아니|글쎄", text):
        return None
    if len(words) == 2 and not any(od.PROVINCE_ANSWER.match(w) or od.CITY_UNIT.search(w) for w in words):
        return None
    if len(words) == 1 and not (region_marked(text) or model_saw_city):
        return None
    return text


def operation_conflict(t: Turn) -> str | None:
    """(F1) 모델은 변경(update)이라 했는데 발화가 취소 요청이거나 날씨가 붙은 완료 표현이면 'cancel'·'complete'. 아니면 None.
    바꿀 시각·날짜·계산되는 상대 시각이 있거나 변경 표현('바꿔·옮겨')이 함께 있으면 변경 요청으로 둔다(다른 일정 취소가 이유인 경우 포함)."""
    u, text = t.u, t.message
    # 대상의 날짜·시각을 새 칸에 한 번 더 옮겨 적은 값(실측: date_text·new_date_text 모두 '내일')은 바꿀 값으로 보지 않는다(D38과 같은 중복 기재).
    new_date = u["new_date_text"] and _squeeze(u["new_date_text"]) != _squeeze(u["date_text"])
    new_time = u["new_time_text"] and _squeeze(u["new_time_text"]) != _squeeze(u["time_text"])
    if UPDATE_MARKS.search(text) or new_date or new_time:
        return None
    moved = sv.resolve_shift(u["shift_text"], t.now) if u["shift_text"] else None
    if moved and "at" in moved:
        return None
    if CANCEL_REQUEST_MARKS.search(text) and not (CANCEL_BLOCK_MARKS.search(text) or NEGATION_ANY.search(text)):
        return "cancel"
    if COMPLETE_MARKS.search(text) and said_conditions(text):
        return "complete"
    return None


def said_conditions(message: str) -> list[str]:
    """발화에 나온 날씨 조건 낱말의 조건 이름 목록"""
    return [p for p in PREFERENCES if CONDITION_WORDS[p].search(message or "")]


def explicit_waiver(text: str, conditions: list[str]) -> list[str]:
    """명시적 철회로 확인되는 조건 목록. (5-6 D45·D46 보완) 조건 낱말 위치로 발화를 나눠 조건마다 따로 본다.
    한 조건의 범위 = 그 낱말부터 다음 조건 낱말('날씨'·'조건' 포함) 앞까지. 범위에 회피·부정 표지가 있으면 그 조건은 철회하지 않고,
    명시적 무관 표지('상관없어'·'괜찮아')가 있으면 철회 후보다. 양보('와도')·접속('랑')만 있는 범위는 바로 뒤 범위의 판정을 따른다.
    질문형 발화는 철회하지 않는다. '날씨'·'조건' 범위는 자기 범위가 없는 조건에만 적용한다. 범위를 알 수 없으면 빈 목록(다시 묻기)"""
    if not text or QUESTION_END.search(text.strip()):
        return []
    # 1. 조건 낱말 위치와 범위별 판정 (avoid | waive | none)
    marks = sorted([(m.start(), m.end(), p) for p in PREFERENCES for m in CONDITION_WORDS[p].finditer(text)]
                   + [(m.start(), m.end(), None) for m in WEATHER_GENERIC.finditer(text)], key=lambda x: x[0])
    verdicts: list[list[Any]] = []
    for i, (start, end, p) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        span, rest = text[start:stop], text[end:stop]
        # (5-6 D54 보완) 같은 범위에 불확실('모르겠'·'글쎄')·금지('가지 마') 표현이 있으면 무관 표지가 있어도 철회하지 않는다.
        if AVOID_MARKS.search(span) or WAIVER_BLOCK_MARKS.search(span) or UNCERTAIN_MARKS.search(span):
            verdict = "avoid"
        elif INDIFFERENT_MARKS.search(span):
            verdict = "waive"
        else:
            # 양보·접속 표현뿐인 범위('비 와도 ', '비랑 ')만 뒤 범위를 따른다. 근거가 없는 범위는 철회하지 않는다.
            residue = CONCESSIVE_MARKS.sub("", rest).strip(" ,·")
            linked = bool(LINK_ONLY.fullmatch(rest)) or (bool(CONCESSIVE_MARKS.search(rest)) and len(residue) <= 2)
            verdict = "link" if linked else "none"
        verdicts.append([p, verdict])
    for i in range(len(verdicts) - 1, -1, -1):
        if verdicts[i][1] == "link":
            verdicts[i][1] = verdicts[i + 1][1] if i + 1 < len(verdicts) else "none"
    # 2. 조건마다 결정: 자기 범위가 있으면 그것만, 없으면 '날씨·조건' 범위
    generic = [v for q, v in verdicts if q is None]
    waived = []
    for p in conditions:
        own = [v for q, v in verdicts if q == p] or generic
        if "waive" in own and "avoid" not in own:
            waived.append(p)
    return waived


def merge_conditions(draft: dict[str, Any], u: dict[str, Any]) -> None:
    """(5-6 D44 보완) 초안의 확정 조건에 이번 턴의 근거 확인된 조건을 더한다(바꿔치기하지 않음). 철회는 확인 질문에서만 한다."""
    new = set(u["preferences"])
    draft["waived"] = sorted(set(draft["waived"]) - new)
    draft["preferences"] = sorted((set(draft["preferences"]) | new) - set(draft["waived"]))


def weather_question(conditions: list[str], kind: str) -> str:
    verb = "등록" if kind == "create" else "변경"
    names = "·".join(PREFERENCE_WORDS[p] for p in conditions)
    asks = ", ".join(CONDITION_ASK[p] for p in conditions)
    return (f"말씀에 날씨 이야기({names})가 있어요. {asks}만 {verb}할까요? "
            f"날씨와 관계없이 {verb}하시려면 '날씨 상관없이'라고 말씀해 주세요.")


# %% (5-6 D34·D42·D43 보완) 확인 질문 대답 이해 LLM 프롬프트와 출력 규격
# 핵심 요약: Python이 물어본 확인 질문(날씨 조건·위임·취소)에 대한 대답의 뜻은 LLM이 고른다. 기존 일정 이해 프롬프트는 바꾸지 않는다.
# Python은 answer 하나만 믿지 않고, 근거 표현이 발화에 있는지와 명확한 불일치(부정인데 yes, 회피인데 철회)만 확인한다.
CONFIRM_ANSWERS = ("yes", "no_condition", "no", "cancel", "other")
CONFIRM_PROMPT = """당신은 SafeMate의 '확인 대답 이해 단계'입니다. Python이 어르신께 물어본 '확인_질문'과 '사용자_대답'을 읽고
대답의 뜻을 JSON으로만 고릅니다. 답변 문장을 쓰지 마세요.
answer: yes(질문 내용에 동의: "네", "응", "그렇게 해 줘"), no_condition(날씨 조건을 빼 달라는 뜻: "날씨 상관없이", "~와도 괜찮아"),
no(아니라고만 했거나 무엇을 원하는지 분명하지 않은 부정), cancel(요청 자체를 그만두라는 뜻: "등록하지 마", "그만할래"),
other(질문과 관계없는 말이거나 판단할 수 없음)
evidence_text: 판단 근거가 된 표현을 '사용자_대답'에서 글자 그대로 옮겨 적습니다. 없으면 ""."""


def confirm_schema() -> dict[str, Any]:
    """확인 대답 이해 JSON 스키마"""
    return {"type": "object", "properties": {"answer": {"type": "string", "enum": list(CONFIRM_ANSWERS)},
                                             "evidence_text": {"type": "string"}},
            "required": ["answer", "evidence_text"], "additionalProperties": False}


def validate_confirm(data: Any) -> tuple[str, str]:
    """(answer, evidence_text). 형식이 틀리면 ValueError (호출하는 쪽이 한 번 다시 요청)"""
    if not isinstance(data, dict) or set(data) != {"answer", "evidence_text"}:
        raise ValueError("키가 규격과 다릅니다")
    if data["answer"] not in CONFIRM_ANSWERS:
        raise ValueError("answer 값이 허용 목록에 없습니다")
    if not isinstance(data["evidence_text"], str):
        raise ValueError("evidence_text는 문자열이어야 합니다")
    return data["answer"], data["evidence_text"].strip()[:TEXT_LIMIT]


# %% 결과 딕셔너리와 한 턴의 작업 묶음
# 핵심 요약: 그래프가 읽는 키를 항상 같은 모양으로 만든다. 한 턴에 쓰는 값은 Turn 하나로 묶어 함수에 넘긴다.
def empty_schedule_result() -> dict[str, Any]:
    """A 결과 계약과 같은 이름의 키 + 일정 전용 키(schedules, weather)"""
    return {"status": None, "detail_code": None, "message": "", "route": None, "answer": "", "conversation_state": None,
            "missing_fields": [], "clarification": None, "requires_user_confirmation": False, "proposed_action": None,
            "usage": {"llm_calls": 0, "weather_calls": 0, "rag_searches": 0}, "tool_trace": [], "limitations": [],
            "official_guidance": [], "safety_evidence": [], "rag_status": {}, "embedding_device": None,
            "weather": [], "schedules": [], "decision": None,
            "llm": {"model": None, "calls": [], "understanding": None, "discarded": [], "rules_applied": [], "choice": None}}


@dataclass
class Turn:
    """한 번의 요청을 처리하는 동안 함께 쓰는 값"""
    message: str
    u: dict[str, Any]
    state: dict[str, Any]
    doc: dict[str, Any]
    now: datetime
    result: dict[str, Any]
    settings: dict[str, Any]
    device: str | None
    index: Any
    b_records: list[dict[str, Any]] | None
    draft: dict[str, Any] | None = None
    notices: list[str] = field(default_factory=list)
    answer_source: str = "python_template"
    # (5-6 D21 보완) 이번 턴에 조회한 정시 구간 예보 {(지역, 정시 구간 시작): 결과}. 같은 구간을 다시 조회하지 않는다(턴이 끝나면 버림).
    forecasts: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    def rule(self, name: str) -> None:
        """Python이 적용한 보정·판단 규칙을 기록한다. (모델 판단과 프로그램 보정을 구분해 보고하기 위함)"""
        self.result["llm"]["rules_applied"].append(name)


# %% 화면 문장 도우미
# 핵심 요약: 일정과 시각을 '내일(11월 26일 목요일) 오후 2시'처럼 읽기 쉬운 말로 바꾼다. 문장은 모두 Python 틀이다.
def when_text(moment: datetime, now: datetime) -> str:
    return f"{sv.korean_day(moment.date(), now)} {sv.korean_time(moment)}"


def item_by_id(doc: dict[str, Any], schedule_id: Any) -> dict[str, Any] | None:
    return next((s for s in doc["schedules"] if s["schedule_id"] == schedule_id), None)


def numbered(items: list[dict[str, Any]], now: datetime, suggested: list[str] | None = None) -> str:
    """'1) 내일(…) 오후 2시~오후 3시 시장 장보기 [예정]' 목록. LLM이 추천한 일정은 (추천)만 붙인다."""
    marks = set(suggested or [])
    return "\n".join(f"{n}) {sv.describe(item, now)}" + (" (추천)" if item["schedule_id"] in marks else "")
                     for n, item in enumerate(items, 1))


def bullet(items: list[dict[str, Any]], now: datetime, *, with_day: bool = True) -> str:
    return "\n".join(f"• {sv.describe(item, now, with_day=with_day)}" for item in items)


# %% 결과 마무리 함수 (답변·질문·제안·넘김)
# 핵심 요약: status·route·질문·제안이 그래프 계약(needs_input ⇔ needs_clarification, 제안 ⇔ requires_user_confirmation)과 맞게 채운다.
def reply(t: Turn, answer: str, kind: str = "schedule_answer", *, keep_draft: bool = False) -> None:
    """조회 답변·안내. 승인 제안 없음. keep_draft가 아니면 진행 중인 작업을 끝낸다."""
    if not keep_draft:
        t.state["draft"] = None
    t.result.update(status="ok", route="schedule", answer=answer)
    t.result["decision"] = {"kind": kind}


def ask(t: Turn, draft: dict[str, Any], awaiting: str, question: str, *, candidates: list[str] | None = None,
        reason: str = "") -> None:
    """부족한 정보 하나만 묻는다. 다음 턴의 짧은 대답을 잇기 위해 초안에 물어본 항목과 후보를 남긴다."""
    draft.update(awaiting=awaiting, question=question)
    if candidates is not None:
        draft["candidates"] = candidates
    t.state["draft"] = draft
    t.result.update(status="needs_input", route="needs_clarification", answer=question, missing_fields=[awaiting],
                    clarification={"fields": [awaiting], "question": question, "reasons": {awaiting: reason} if reason else {}})
    t.result["decision"] = {"kind": "schedule_question"}


def propose(t: Turn, draft: dict[str, Any], action: dict[str, Any], lines: list[str]) -> None:
    """승인용 제안을 돌려준다. 저장하지 않는다. 다음 턴의 정정('아니 모레 오전으로')을 받기 위해 초안은 남긴다."""
    draft.update(awaiting=None, question="", candidates=[])
    t.state["draft"] = draft
    t.result.update(status="ok", route="schedule", answer="\n".join(lines + [ASK_SUFFIX]),
                    requires_user_confirmation=True, proposed_action=action)
    t.result["decision"] = {"kind": "schedule_proposal"}


def hand_off(t: Turn) -> None:
    """일정 관리가 아닌 말: 그래프가 A기능(→ 필요하면 B기능)으로 넘긴다. 진행 중인 일정 초안은 지우지 않는다."""
    t.result.update(status="ok", route="handoff", answer=HANDOFF_TEXT)
    t.result["decision"] = {"kind": "handoff"}


# %% 예보 조회 함수
# 핵심 요약: 실외 일정의 한 시각에 대해 3단계 read_weather()로 1시간 구간 예보를 새로 조회하고, signal·no_signal·unknown으로 나눈다.
# 캐시를 쓰지 않는다. 매번 조회한 시각(fetched_at)을 남긴다. 지역이 없거나 예보 기간 밖이면 조회하지 않고 unknown이다.
def fetch_forecast(t: Turn, city: str | None, at: datetime) -> dict[str, Any]:
    """{"status", "signals", "reason", "weather", "at", "fetched_at"}"""
    def unknown(reason: str) -> dict[str, Any]:
        return {"status": "unknown", "signals": {}, "reason": reason, "weather": None, "at": ss.iso(at), "fetched_at": None}

    usage = t.result["usage"]
    if not city:
        return unknown("일정에 지역이 없어 예보를 확인하지 않았어요.")
    if at < t.now or at.date() >= t.now.date() + timedelta(days=FORECAST_DAYS):
        return unknown(f"예보 조회 기간(오늘부터 {FORECAST_DAYS}일) 밖이라 날씨는 확인하지 않았어요.")
    if usage["weather_calls"] >= MAX_WEATHER_CALLS:
        return unknown("이번 요청의 예보 조회 횟수 상한에 도달했어요.")
    usage["weather_calls"] += 1
    reply_ = od.read_weather(city, at, t.result["tool_trace"])
    status = sv.forecast_status(reply_)
    checked = {**status, "weather": reply_.get("weather") if isinstance(reply_, dict) else None, "at": ss.iso(at),
               "fetched_at": ss.iso(od.current_time())}
    t.result["weather"].append(checked)
    return checked


# %% (5-6 D21 보완) 일정 전체 시간 구간 예보 확인 함수
# 핵심 요약: 날씨 조건이 있는 일정은 시작 시각의 1시간 구간만이 아니라 [시작, 종료)와 겹치는 모든 정시 예보 구간을 확인한다.
# (14:00~15:00은 14시, 14:30~15:30은 14·15시, 자정을 넘기면 다음 날 0시도) 같은 지역·정시 구간은 한 턴 안에서 다시 조회하지 않고
# 호출 상한(4회)은 그대로 둔다. 한 구간이라도 확인하지 못하면(상한·예보 기간 밖·누락·오류) 전체를 unknown으로 보고 조건 충족으로 쓰지 않는다.
SPAN_MAX_KEYS = ("temperature_c", "apparent_temperature_c", "wind_speed_mps", "precipitation_mm", "precipitation_probability_pct")


def span_hours(start: datetime, end: datetime) -> list[datetime]:
    """[start, end)와 겹치는 정시 구간마다 조회할 시각. 첫 구간은 시작 시각 그대로(기존 조회와 같음), 다음 구간부터는 정시"""
    hours = [start]
    hour = start.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    while hour < end:
        hours.append(hour)
        hour += timedelta(hours=1)
    return hours


def fetch_span(t: Turn, city: str | None, start: datetime, end: datetime) -> dict[str, Any]:
    """구간별 예보를 합친 결과. {"status","signals","reason","weather","at","fetched_at","parts"}
    weather는 후보 비교·이유 검증용으로 구간별 값의 최댓값(한파 비교용 최저 기온은 coldest)이다. 표시는 parts의 원래 값을 쓴다."""
    # 1. 구간마다 예보를 조회한다. 이번 턴에 이미 조회한 같은 지역·정시 구간은 그 결과를 다시 쓴다.
    parts = []
    for at in span_hours(start, end):
        key = (str(city), ss.iso(at.replace(minute=0, second=0, microsecond=0)))
        if key not in t.forecasts:
            t.forecasts[key] = fetch_forecast(t, city, at)
        parts.append(t.forecasts[key])
    if len(parts) == 1:
        return dict(parts[0], at=ss.iso(start), parts=parts)
    # 2. 위험 신호는 모든 구간을 합친다. 한 구간이라도 확인하지 못했으면 전체를 unknown으로 둔다.
    signals: dict[str, list[str]] = {}
    for part in parts:
        for topic, fields in part["signals"].items():
            signals.setdefault(topic, [])
            signals[topic] += [f for f in fields if f not in signals[topic]]
    unknown = [part for part in parts if part["status"] == "unknown"]
    if unknown:
        hours = ", ".join(sv.korean_time(ss.parse_iso(part["at"]) or start) for part in unknown)
        status, reason = "unknown", f"일정 시간 중 {hours} 구간의 예보를 확인하지 못했어요. ({unknown[0]['reason']})"
    else:
        status = "signal" if signals else "no_signal"
        reason = ("일정 시간 중 프로젝트 기준의 위험 신호가 있는 구간이 있습니다." if signals
                  else "일정 시간의 모든 구간에서 프로젝트 기준의 위험 신호가 보이지 않았습니다(안전 보장 아님).")
    # 3. 후보 비교·이유 검증용 값: 확인한 구간의 최댓값, 한파 비교는 최저 기온(coldest)
    known = [part["weather"] for part in parts if isinstance(part.get("weather"), dict)]
    weather = None
    if known:
        weather = dict(known[0])
        for key in SPAN_MAX_KEYS:
            values = [w[key] for w in known if isinstance(w.get(key), (int, float))]
            weather[key] = max(values) if values else None
        lows = [w["temperature_c"] for w in known if isinstance(w.get("temperature_c"), (int, float))]
        weather["coldest"] = min(lows) if lows else None
    fetched = [part["fetched_at"] for part in parts if part.get("fetched_at")]
    return {"status": status, "signals": signals, "reason": reason, "weather": weather, "at": ss.iso(start),
            "fetched_at": fetched[-1] if fetched else None, "parts": parts}


def weather_needed(u: dict[str, Any], category: Any, city: Any) -> bool:
    """실외 외출 일정이고 지역이 있을 때만 예보를 본다. 날씨와 무관한 개인 사정(늦잠 등)만 말했으면 보지 않는다. (요구 13·16)"""
    if category != "outing" or not city:
        return False
    personal = bool(u["reason_text"]) and not (u["check_weather"] or u["preferences"])
    return not personal


def weather_block(t: Turn, checks: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """[확인한 예보] 줄: 3단계 weather_fact_lines()로 원자료 값을 그대로 적고, 구간별 판단과 조회 시각을 붙인다."""
    slots, notes = [], []
    for label, check in checks:
        # (5-6 D21 보완) 일정 전체 구간을 확인한 결과는 구간마다 원래 예보 값과 판단을 따로 적는다(일부 구간만으로 전체를 말하지 않음).
        for number, part in enumerate(check.get("parts") or [check]):
            at = ss.parse_iso(check["at"] if number == 0 else part["at"]) or t.now
            if part["weather"]:
                slots.append(dict(part["weather"], fact_id=f"W{len(slots) + 1}"))
            state = {"signal": "프로젝트 기준 위험 신호 있음(" + ", ".join(part["signals"]) + ")",
                     "no_signal": "프로젝트 기준 위험 신호 없음(안전 보장 아님)"}.get(part["status"], "확인 못함 · " + part["reason"])
            notes.append(f"• {label if number == 0 else label + '(이어지는 구간)'} {sv.korean_time(at)}: {state}")
    if not checks:
        return []
    lines = ["[확인한 예보] (조회한 예보 값을 그대로 옮겼어요)"]
    if slots:
        first = ss.parse_iso(checks[0][1]["at"]) or t.now
        weather = {"city": slots[0]["city"], "slots": slots, "failed_windows": [], "date": first.date().isoformat(),
                   "date_label": sv.korean_day(first.date(), t.now)}
        lines += od.weather_fact_lines(weather)
    fetched = [c["fetched_at"] for _, c in checks if c["fetched_at"]]
    if fetched:
        notes.append(f"(예보 조회 시각: {fetched[-1][11:16]}) {NO_GUARANTEE}")
    return lines + notes


# %% 공식 안전수칙 검색 함수
# 핵심 요약: 고른 시각의 예보에 위험 신호가 있을 때만 3단계 choose_topics()·collect_evidence()로 2단계 RAG를 부른다.
# 공식 안내는 검색된 원문 문장을 글자 그대로 인용한다(3단계 quote_original). 검색에 실패하면 지어내지 않고 한계로 남긴다.
def safety_guidance(t: Turn, check: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not check or check["status"] != "signal" or not check["weather"]:
        return []
    # (5-6 D21 보완) 일정 전체 구간을 확인했으면 위험 신호가 있는 모든 구간의 원래 예보 값으로 주제를 고른다(두 번째 구간만의 강풍 포함).
    found = [p["weather"] for p in check.get("parts") or [check] if p["status"] == "signal" and p["weather"]]
    topics = od.choose_topics({"slots": [dict(w, fact_id=f"W{n}") for n, w in enumerate(found, 1)]}, "outing_consultation", [])
    try:
        od.collect_evidence(topics, t.device, t.index, t.result, t.result["usage"])
    except od.DecisionError as exc:
        t.result["limitations"].append(f"공식 안전수칙을 검색하지 못했어요. ({exc.code})")
        return []
    guidance = []
    for evidence in t.result["safety_evidence"][:2]:
        quote = od.quote_original(evidence["content"], t.message)
        if quote:
            guidance.append({"evidence_id": evidence["id"], "text": quote, "publisher": evidence["publisher"],
                             "title": evidence["title"], "pdf_page": evidence["pdf_page"], "url": evidence["url"]})
    t.result["official_guidance"] = guidance
    return guidance


def guidance_lines(guidance: list[dict[str, Any]]) -> list[str]:
    if not guidance:
        return []
    lines = ["[공식 안전 안내] (검색된 문서의 원문을 그대로 옮겼어요)"]
    for g in guidance:
        lines.append(f"• “{g['text']}”\n  (출처: {g['publisher']}, 「{g['title']}」 PDF {g['pdf_page']}면)")
    return lines


# %% 일정 이해 실행 함수
# 핵심 요약: 저장된 일정·진행 중인 상담·직전 대화를 '자료'로 넘겨 LLM을 한 번 부르고, 형식이 틀리면 한 번만 다시 요청한다.
def understand_schedule(t: Turn) -> dict[str, Any]:
    """검사와 발화 근거 확인을 마친 일정 이해 결과"""
    items = context_schedules(t.doc, t.now)
    ids = [s["schedule_id"] for s in items]
    draft = t.state["draft"]
    pending = None
    if draft:
        shown = [item_by_id(t.doc, i) for i in draft["candidates"]]
        pending = {"작업": draft["operation"], "물어본_것": draft["question"],
                   "후보": [{"번호": n, "id": s["schedule_id"], "일정": sv.describe(s, t.now)}
                          for n, s in enumerate([s for s in shown if s], 1)]}
    context = {"현재_한국_시각": f"{t.now:%Y-%m-%d %H:%M} ({'월화수목금토일'[t.now.weekday()]}요일)",
               # 일정 목록은 id 고르기에 필요한 짧은 값만 넘긴다. (긴 표기를 넘기면 3B 모델이 텍스트 칸에 그대로 옮겨 적음 — 실측)
               "등록된_일정": [{"id": s["schedule_id"], "제목": s["title"],
                            "시작": f"{sv.interval(s)[0]:%m-%d %H:%M}", "상태": s["status"]} for s in items],
               "진행_중인_상담": pending, "직전_대화": t.state["history"][-HISTORY_LIMIT:], "사용자_발화": t.message}
    messages = [{"role": "system", "content": SCHEDULE_PROMPT},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
    schema = schedule_schema(ids)

    # 1. 최대 2번 요청한다. (3단계 understand_message와 같은 방식, 온도 0)
    last_error = ""
    for _ in (1, 2):
        raw = od.call_llm(messages, schema, t.settings, t.result, t.result["usage"], stage="schedule_understanding",
                          temperature=0)
        try:
            understanding, dropped = validate_schedule_understanding(od.parse_model_json(raw), ids)
            break
        except (ValueError, TypeError) as exc:
            last_error = str(exc)
            messages += [{"role": "assistant", "content": raw[:2000]},
                         {"role": "user", "content": f"형식 오류: {last_error}. 규격의 모든 키를 한 번씩만 넣어 JSON으로 다시 답하세요."}]
    else:
        raise od.DecisionError("llm_invalid_output", "schedule_understanding",
                               f"일정 이해 결과의 형식을 확인하지 못했습니다. ({last_error})")

    # 2. 발화 근거가 없는 표현은 버린다. 모델이 처음 낸 값과 검사 뒤 값을 함께 남긴다.
    grounded, discarded = ground_texts(understanding, t.message, t.state["history"])
    # (5-6 F2 보완) 일정 이름 안의 낱말('공원 걷기'의 '공원')을 지역으로 쓰지 않는다. 지우면 날씨 조건이 있을 때 기존 흐름이 지역을 묻는다.
    grounded, name_city = drop_name_city(grounded, t.doc)
    discarded += name_city
    if name_city:
        t.rule("city_text_was_schedule_name")
    t.result["llm"]["understanding"] = grounded
    t.result["llm"]["model_understanding"] = understanding
    t.result["llm"]["discarded"] = discarded + [{"field": "target_ids", "value": i} for i in dropped]
    if any(d.get("paraphrase_verified") for d in discarded):
        t.rule("shift_paraphrase_verified")
    return grounded


def understand_confirm(t: Turn, draft: dict[str, Any]) -> tuple[str, str]:
    """(5-6 보완) 확인 질문에 대한 대답의 뜻 (answer, 근거 표현). 형식 오류가 두 번이면 other.
    근거 표현이 발화에 그대로 없으면 버린다(근거 없음). 승인 정보는 만들지 않는다."""
    context = {"확인_질문": draft["question"], "물어본_것": draft["awaiting"], "사용자_대답": t.message}
    messages = [{"role": "system", "content": CONFIRM_PROMPT}, {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
    for _ in (1, 2):
        raw = od.call_llm(messages, confirm_schema(), t.settings, t.result, t.result["usage"], stage="schedule_confirm", temperature=0)
        try:
            answer, evidence = validate_confirm(od.parse_model_json(raw))
            break
        except (ValueError, TypeError) as exc:
            messages += [{"role": "assistant", "content": raw[:2000]},
                         {"role": "user", "content": f"형식 오류: {exc}. answer와 evidence_text만 넣어 JSON으로 다시 답하세요."}]
    else:
        t.rule("confirm_answer_format_invalid")
        return "other", ""
    if evidence and _squeeze(evidence) not in _squeeze(t.message):
        evidence = ""
    t.result["llm"]["confirm"] = {"answer": answer, "evidence_text": evidence}
    return answer, evidence


def confirm_conflict(t: Turn, answer: str, evidence: str = "", kind: str = "weather_condition") -> bool:
    """모델 yes를 사용자 동의로 볼 수 없는지 (5-6 D47 보완): 근거 표현이 없음, 일정 이해가 negated, 발화·근거에 불확실 표현('글쎄'·'모르겠'),
    부정('아니'로 시작, '하지 마'·'그만'). 날씨 조건 확인에서 '비 오면 하지 마'처럼 조건 낱말과 함께 쓴 부정은 조건을 지키겠다는 뜻이라 막지 않는다.
    위임·취소 확인은 결과의 위험이 커서 회피('피하고 싶어'·'싫어')·질문형도 동의로 보지 않는다. 의미 판단 자체는 확인 단계 LLM이 한다."""
    if answer != "yes":
        return False
    text = t.message
    if not evidence or t.u["request_mode"] == "negated":
        return True
    if UNCERTAIN_MARKS.search(text) or UNCERTAIN_MARKS.search(evidence) or NEGATION_START.search(text):
        return True
    if NEGATION_ANY.search(text) and (kind != "weather_condition" or not said_conditions(text)):
        return True
    return kind != "weather_condition" and bool(AVOID_MARKS.search(text) or QUESTION_END.search(text.strip()))


# %% 대상 일정 찾기 함수
# 핵심 요약: 앞 턴에 보여 준 후보 중에서 고른 대답인지 먼저 보고, 아니면 schedule_service.find_targets()로 찾는다.
# LLM이 고른 id만으로는 확정하지 않는다. (single은 사용자 발화로 하나가 정해진 경우뿐, 여러 개면 llm_suggested는 추천 표시만)
def _parse_target_filters(t: Turn, draft: dict[str, Any], keep_previous: bool) -> dict[str, Any]:
    """찾을 일정의 이름·날짜·시각·시간대. 새로 말한 것이 없으면 직전 조건을 이어 쓴다."""
    u, query = t.u, draft["query"]
    text = u["target_text"] or (query["text"] if keep_previous else "")
    filters: dict[str, Any] = {"text": text, "date": query["date"] if keep_previous else None,
                               "time": query["time"] if keep_previous else None,
                               "period": query["period"] if keep_previous else None}
    if u["date_text"]:
        parsed = sv.parse_day(u["date_text"], t.now)
        if parsed and "ambiguous" in parsed:
            return {"ambiguous": parsed["ambiguous"]}
        filters["date"] = parsed["date"] if parsed else None
    if u["time_text"]:
        clock = sv.parse_clock(u["time_text"])
        if clock and "time" in clock:
            filters["time"], filters["period"] = clock["time"], None
        elif clock and "ambiguous" in clock:
            # (검토 보완) '두 시'처럼 오전·오후가 모호하면 시각 조건을 버리고 날짜만으로 찾지 않는다. 날짜 등 확인한 조건은 남긴다.
            return {"ambiguous": clock["ambiguous"], "awaiting": "target_time", "partial": filters}
        elif clock is None:
            period = od.parse_period_text(u["time_text"], include_soon=False)
            filters["period"] = period if period in od.PERIODS else None
    return filters


def resolve_target(t: Turn, draft: dict[str, Any], operation: str) -> dict[str, Any]:
    """{"item"}, {"ask": 질문, "candidates": [...], "awaiting"} 또는 {"refuse": 안내}"""
    u, doc, now = t.u, t.doc, t.now
    # 1. 앞 턴에 보여 준 후보 중에서 고른 대답 ('첫 번째', '내일 거', '시장')
    shown = [s for s in (item_by_id(doc, i) for i in draft["candidates"]) if s]
    if shown and draft["awaiting"] in ("target", "confirm") and not u["exclude_previous_target"]:
        chosen = sv.choose_from_candidates(shown, t.message, now)
        # 하나만 보여 주고 '이 일정 말씀이신가요?'라고 물었으면, 모델이 대답으로 이해했고 부정하지 않은 경우만 확인으로 본다.
        if chosen is None and draft["awaiting"] == "confirm" and len(shown) == 1 \
                and u["answers_previous_question"] and u["request_mode"] in ("request", "question"):
            chosen = shown[0]
            t.rule("target_confirmed_by_answer")
        if chosen is not None and (operation == "query" or chosen["status"] == "scheduled"):
            t.rule("target_chosen_from_candidates")
            return {"item": chosen}
        if not (u["target_text"] or u["date_text"] or u["time_text"]):
            return {"ask": draft["question"] or "어느 일정인지 번호로 알려 주세요.\n" + numbered(shown, now),
                    "candidates": [s["schedule_id"] for s in shown], "awaiting": draft["awaiting"]}

    # 2. '다른 일정이야': 직전에 고른 일정(또는 보여 준 추천)을 후보에서 뺀다.
    excluded = list(draft["excluded_ids"])
    if u["exclude_previous_target"]:
        excluded += [draft["target_id"]] if draft["target_id"] else [i for i in draft["candidates"][:1]]
        draft["target_id"] = None
        t.rule("previous_target_excluded")
    draft["excluded_ids"] = sorted(set(excluded))

    # 3. 이름·날짜·시각으로 찾는다. 새 대상을 말하지 않았으면 직전 조건을 이어 쓴다.
    # (5-6 D38 보완) 변경 요청에서 모델이 새 시각을 '찾을 일정의 시각'(time_text)에도 똑같이 넣었고 그 표현이 발화에 한 번만 나오면
    # 새 시각으로만 본다. (같은 표현을 두 번 말했으면 대상 시각과 새 시각을 모두 말한 것이라 대상 조건을 지우지 않음)
    if operation == "update" and u["time_text"] and _squeeze(u["time_text"]) == _squeeze(u["new_time_text"]) \
            and _squeeze(t.message).count(_squeeze(u["time_text"])) <= 1:
        t.u = u = dict(u, time_text="")
        t.rule("duplicated_time_text_used_as_new_time")
    keep = bool(draft["query"]["text"] or draft["query"]["date"]) and not u["target_text"]
    filters = _parse_target_filters(t, draft, keep)
    time_is_new = False
    if "ambiguous" in filters:
        partial = filters.get("partial")
        # 변경 요청에서 이름으로 대상을 말했고 '바꿀 시각' 칸이 비어 있으면, 모호한 시각은 바꿀 시각이다.
        # 이 경우만 이름으로 대상을 찾고, 새 시각의 오전·오후는 resolve_new_time()이 묻는다.
        if isinstance(partial, dict) and operation == "update" and sv.name_words(partial["text"]) and not u["new_time_text"]:
            filters, time_is_new = partial, True
            t.rule("ambiguous_time_text_used_as_new_time")
        else:
            if isinstance(partial, dict):
                draft["query"] = partial
            return {"ask": filters["ambiguous"], "candidates": [], "awaiting": filters.get("awaiting", "date")}
    draft["query"] = filters
    lookup = "query" if operation == "query" else operation
    found = sv.find_targets(doc, now=now, operation=lookup, target_text=filters["text"], day=filters["date"],
                            clock=filters["time"], period=filters["period"], llm_ids=t.u["target_ids"],
                            exclude_ids=draft["excluded_ids"])
    # 3-1. (변경 요청) 시각을 찾을 일정의 시각으로 읽어 아무것도 없으면, 그 시각을 '바꿀 시각'으로 보고 이름으로만 다시 찾는다.
    # 예: 모델이 '시장 가는 시간을 오후 4시로'의 '오후 4시'를 time_text에 넣은 경우
    if found["status"] == "none" and operation == "update" and filters["time"] and filters["text"] and not u["new_time_text"]:
        retry = sv.find_targets(doc, now=now, operation=lookup, target_text=filters["text"], day=filters["date"],
                                llm_ids=t.u["target_ids"], exclude_ids=draft["excluded_ids"])
        if retry["status"] == "single":
            t.rule("time_text_used_as_new_time")
            draft["query"]["time"] = None
            return {"item": retry["matches"][0], "time_is_new": True}
    # 3-2. (5-6 D38 보완) 시간대('저녁')로 찾아 아무것도 없으면, 이름만으로 찾은 일정이 하나일 때만 그 시간대를 '바꿀 시각'으로 본다.
    # (그 일정이 그 시간대 안에 있었다면 위에서 찾았으므로, 여기서 찾은 일정은 시간대 밖에 있다. 여러 개면 아래에서 묻는다)
    if found["status"] == "none" and operation == "update" and filters["period"] and filters["text"] and not u["new_time_text"]:
        retry = sv.find_targets(doc, now=now, operation=lookup, target_text=filters["text"], day=filters["date"],
                                llm_ids=t.u["target_ids"], exclude_ids=draft["excluded_ids"])
        if retry["status"] == "single":
            t.rule("period_text_used_as_new_time")
            draft["query"]["period"] = None
            return {"item": retry["matches"][0], "time_is_new": True}
    t.result["llm"]["target_lookup"] = {"status": found["status"], "basis": found["basis"],
                                        "llm_suggested": found["llm_suggested"]}

    # 4. 결과별 처리
    words = {"update": "바꿀", "cancel": "취소할", "complete": "완료로 기록할", "query": "알려 드릴"}[lookup]
    if found["status"] == "single":
        return {"item": found["matches"][0], "time_is_new": time_is_new}
    if found["status"] == "multiple":
        shown = found["matches"][:LIST_LIMIT]
        question = f"해당하는 일정이 여러 개예요. 어느 일정을 {words}까요? 번호로 알려 주세요.\n" + \
                   numbered(shown, now, found["llm_suggested"])
        return {"ask": question, "candidates": [s["schedule_id"] for s in shown], "awaiting": "target"}
    if found["status"] == "confirm":
        item = found["matches"][0]
        return {"ask": f"'{item['title']}'({when_text(sv.interval(item)[0], now)}) 일정 말씀이신가요?",
                "candidates": [item["schedule_id"]], "awaiting": "confirm"}
    if found["status"] == "none" and operation in ("cancel", "complete"):
        # 완료·취소된 일정을 가리킨 경우는 '없다' 대신 그 상태를 알려 준다.
        other = sv.find_targets(doc, now=now, operation="query", target_text=filters["text"], day=filters["date"],
                                clock=filters["time"], period=filters["period"], exclude_ids=draft["excluded_ids"])
        if other["status"] == "single" and other["matches"][0]["status"] != "scheduled":
            item = other["matches"][0]
            message = sv.completion_check(item, now)["message"] if operation == "complete" else \
                f"'{item['title']}'은(는) 이미 {sv.STATE_WORDS[item['status']]} 상태라 취소할 수 없어요."
            return {"refuse": message}
    # 5. 없음·말하지 않음: 없는 일정을 바꿨다고 하지 않고, 고를 수 있는 일정을 보여 주며 묻는다.
    pool = [s for s in doc["schedules"] if s["schedule_id"] not in draft["excluded_ids"]
            and (lookup == "query" or (s["status"] == "scheduled" and sv.interval(s)[1] > now))]
    if filters["date"]:
        pool = [s for s in pool if s in sv.schedules_on(doc, date.fromisoformat(filters["date"]))]
    pool = sorted(pool, key=lambda s: s["start_at"])[:LIST_LIMIT]
    head = "말씀하신 일정을 저장된 기록에서 찾지 못했어요." if found["status"] == "none" else f"어느 일정을 {words}까요?"
    if not pool:
        return {"refuse": head + " 저장된 일정 중에 해당하는 것이 없어요."}
    return {"ask": f"{head} 아래에서 번호로 알려 주세요.\n" + numbered(pool, now),
            "candidates": [s["schedule_id"] for s in pool], "awaiting": "target"}


# %% 새 시각 계산 함수
# 핵심 요약: 상대 시간('한 시간 늦게'), 날짜·시각 표현('모레 오전 열 시'), 정정('일곱 시 아니고 여덟 시'),
# 시간대('저녁쯤'), 선택 위임('알아서')을 구분해 새 시작 시각을 정한다. 계산은 모두 schedule_service 함수가 한다.
NEGATION_SPLIT = re.compile(r"말고|아니고|아니라|아니야|아니")


def _after_negation(text: str) -> tuple[str, str]:
    """('일곱 시 아니고 여덟 시' → ('일곱 시', '여덟 시')). 부정 표현이 없으면 ('', 원문)"""
    found = list(NEGATION_SPLIT.finditer(text or ""))
    if not found:
        return "", text
    return text[:found[-1].start()], text[found[-1].end():]


def _combine(day: date, clock: str) -> datetime:
    return datetime.combine(day, time.fromisoformat(clock), tzinfo=KST)


def resolve_new_time(t: Turn, draft: dict[str, Any], base_start: datetime | None, *,
                     time_is_new: bool = False) -> dict[str, Any]:
    """{"start", "approximate", "inherited"}, {"auto": {"day", "window", "period"}}, {"ask", "awaiting", "reason"} 또는 {}"""
    u, now = t.u, t.now
    awaiting = draft["awaiting"]
    # 1. 상대 시간: 기준(원래 일정·지금)은 LLM이 고르고, 계산은 resolve_shift(datetime)가 한다. 기준을 모르면 묻는다.
    shift = u["shift_text"] or (draft["shift_text"] if awaiting == "shift_base" else "")
    if shift:
        base_name = u["shift_base"] if u["shift_base"] != "unknown" else draft["shift_base"]
        if base_name is None and base_start is None:
            base_name = "now"
        if base_name not in ("schedule", "now"):
            draft.update(shift_text=shift)
            return {"ask": f"원래 일정 시각({sv.korean_time(base_start or now)})을 기준으로 옮길까요, 지금부터 계산할까요?",
                    "awaiting": "shift_base"}
        moved = sv.resolve_shift(shift, base_start if base_name == "schedule" and base_start else now)
        if moved and "ambiguous" in moved:
            draft.update(shift_text=shift, shift_base=base_name)
            return {"ask": moved["ambiguous"], "awaiting": "time"}
        if moved:
            t.rule(f"relative_shift_from_{base_name}")
            draft.update(shift_text="", shift_base=None)
            return {"start": moved["at"], "approximate": False, "inherited": False}

    # 2. 날짜: 새로 말한 날짜 → 직전 턴에 정한 날짜 → 원래 일정 날짜 순서로 쓴다.
    day: date | None = None
    date_text = u["new_date_text"] or (u["date_text"] if awaiting == "date" or base_start is None else "")
    if date_text:
        parsed = sv.parse_day(date_text, now, context_date=(base_start or now).date().isoformat())
        if parsed and "ambiguous" in parsed:
            return {"ask": parsed["ambiguous"], "awaiting": "date"}
        if parsed:
            day = date.fromisoformat(parsed["date"])
            draft["date"] = parsed["date"]
    if day is None and draft["date"]:
        day = date.fromisoformat(draft["date"])

    # 3. 시각: 새 시각 표현, 시각을 묻고 받은 짧은 대답, 정정 표현('~ 아니고 ~')의 뒷부분
    time_text = u["new_time_text"] or (u["time_text"] if awaiting == "time" or time_is_new or base_start is None else "")
    if not time_text and awaiting in ("time", "date") and sv.parse_clock(t.message) is not None:
        time_text = t.message
        t.rule("short_answer_time_from_message")
    inherit = None
    if u["request_mode"] == "correction" or u["corrects_field"] == "time":
        before, after = _after_negation(time_text)
        if not before:
            before, _ = _after_negation(t.message)
        time_text = after.strip() or time_text
        earlier = sv.parse_clock(before) if before else None
        inherit = draft["time"] or (earlier["time"] if earlier and "time" in earlier else None)
    if time_text:
        clock = sv.parse_clock(time_text, inherit_from=inherit)
        if clock and "ambiguous" in clock:
            return {"ask": clock["ambiguous"], "awaiting": "time", "reason": "ampm"}
        if clock and "time" in clock:
            day = day or (base_start.date() if base_start else None)
            if day is None:
                return {"ask": "어느 날인지 알려 주세요.", "awaiting": "date"}
            draft["time"] = clock["time"]
            if clock["inherited"]:
                t.rule("ampm_inherited_from_previous_time")
            return {"start": _combine(day, clock["time"]), "approximate": clock["approximate"],
                    "inherited": clock["inherited"]}
        period = od.parse_period_text(time_text, include_soon=False)
        if period in od.PERIODS:
            return {"auto": {"day": day or (base_start or now).date(), "window": od.PERIODS[period], "period": period}}

    # 4. 시간 선택을 맡겼거나(알아서), 날씨 조건으로 옮겨야 하는 경우: 자동 후보를 찾는다.
    if u["delegate_choice"] or draft["delegate"]:
        draft["delegate"] = True
        return {"auto": {"day": day or (base_start or now).date(), "window": sv.AUTO_WINDOW, "period": None}}

    # 5. 날짜만 바꾼 경우('금요일로 옮겨 줘'): 원래 시각을 유지한다. 원래와 같은 날짜면 바꿀 내용이 없으므로 몇 시인지 묻는다.
    if day is not None and base_start is not None and day != base_start.date() and (u["new_date_text"] or awaiting == "date"):
        return {"start": _combine(day, f"{base_start:%H:%M}"), "approximate": False, "inherited": False}
    return {}


# %% 자동 후보 선택 함수 (시간 선택 위임)
# 핵심 요약: Python이 빈 시간 후보(최대 3개)와 후보별 예보를 계산하고, LLM은 그중 하나와 이유 코드를 고른다.
# Python이 이유가 실제 수치와 맞는지 확인한다. 맞지 않거나 호출이 실패하면 Python 규칙으로 고른다(python_fallback).
# 사용자가 피하려는 날씨가 있으면, 예보를 확인했고(unknown 아님) 그 날씨 신호가 없는 후보만 고를 수 있다.
REASON_WORDS = {"less_rain": "원래 시각보다 비 예보가 적어요", "lower_wind": "원래 시각보다 바람이 약해요",
                "milder_temp": "원래 시각보다 기온이 덜 덥거나 덜 추워요", "no_conflict": "다른 일정과 겹치지 않아요",
                "closer_to_original": "원래 시각과 가장 가까운 빈 시간이에요"}


def _values(check: dict[str, Any]) -> dict[str, Any]:
    weather = check.get("weather") or {}
    return {key: weather.get(key) for key in od.FIELDS}


def eligible(check: dict[str, Any], preferences: list[str]) -> bool:
    """피하려는 날씨가 있으면 예보를 확인했고 그 신호가 없어야 한다. 없으면 어떤 후보든 된다."""
    if not preferences:
        return True
    topics = {PREFERENCE_TOPICS[p] for p in preferences}
    return check["status"] != "unknown" and not topics & set(check["signals"])


def reason_holds(reason: str, index: int, checks: list[dict[str, Any]], original: dict[str, Any] | None,
                 starts: list[datetime], prefer: datetime | None) -> bool:
    """LLM이 고른 이유 코드가 실제 후보·예보 값과 맞는지 Python이 확인한다."""
    chosen = checks[index]
    if reason == "no_conflict":
        return True     # 후보는 모두 충돌 검사를 통과한 빈 시간이다
    if reason == "closer_to_original":
        if prefer is None:
            return False
        gaps = [abs((s - prefer).total_seconds()) for s in starts]
        return gaps[index] == min(gaps)
    # 날씨 이유는 두 구간 모두 예보를 확인했을 때만 비교한다. (unknown을 근거로 쓰지 않음)
    if original is None or "unknown" in (original["status"], chosen["status"]):
        return False
    before, after = _values(original), _values(chosen)
    keys = {"less_rain": ("precipitation_mm", "precipitation_probability_pct"), "lower_wind": ("wind_speed_mps",),
            "milder_temp": ("temperature_c",)}.get(reason, ())
    if not keys or any(before[k] is None or after[k] is None for k in keys):
        return False
    if reason == "less_rain":
        return all(after[k] <= before[k] for k in keys) and any(after[k] < before[k] for k in keys)
    if reason == "lower_wind":
        return after["wind_speed_mps"] < before["wind_speed_mps"]
    if "폭염" in original["signals"]:
        return after["temperature_c"] < before["temperature_c"]
    if "한파" in original["signals"]:
        # (5-6 D21 보완) 여러 구간을 합친 예보는 최고값을 담으므로, 한파 비교는 가장 추운 구간의 기온(coldest)으로 한다.
        cold_before = (original.get("weather") or {}).get("coldest", before["temperature_c"])
        cold_after = (chosen.get("weather") or {}).get("coldest", after["temperature_c"])
        return cold_before is not None and cold_after is not None and cold_after > cold_before
    return False


def llm_choice(t: Turn, starts: list[datetime], checks: list[dict[str, Any]], original: dict[str, Any] | None,
               prefer: datetime | None, preferences: list[str]) -> dict[str, Any] | None:
    """{"choice": 후보 번호 또는 None, "reasons"}. 호출·형식 실패면 None (Python 규칙으로 넘어감)"""
    ids = [f"C{n}" for n in range(1, len(starts) + 1)]
    labels = {key: f"{label}({unit})" for key, (label, unit) in od.FIELDS.items()}

    def row(at: datetime, check: dict[str, Any] | None) -> dict[str, Any]:
        values = _values(check) if check else {}
        return {"시각": when_text(at, t.now), "예보_상태": check["status"] if check else "unknown",
                **{labels[k]: v for k, v in values.items()}}

    context = {"원래_시각": row(prefer, original) if prefer else None,
               "후보": [dict(row(s, c), id=i, 원래_시각과_차이_분=int(abs((s - prefer).total_seconds()) // 60) if prefer else None)
                      for i, s, c in zip(ids, starts, checks)],
               "피하고_싶은_날씨": [PREFERENCE_WORDS[p] for p in preferences], "사용자_발화": t.message}
    schema = {"type": "object", "additionalProperties": False, "required": ["choice", "reasons"],
              "properties": {"choice": {"type": "string", "enum": ids + ["none"]},
                             "reasons": {"type": "array", "items": {"type": "string", "enum": list(CHOICE_REASONS)}}}}
    messages = [{"role": "system", "content": CHOICE_PROMPT},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
    try:
        raw = od.call_llm(messages, schema, t.settings, t.result, t.result["usage"], stage="schedule_choice", temperature=0)
        data = od.parse_model_json(raw)
        if set(data) != {"choice", "reasons"} or data["choice"] not in ids + ["none"] or not isinstance(data["reasons"], list) \
                or any(r not in CHOICE_REASONS for r in data["reasons"]):
            raise ValueError("choice")
    except od.DecisionError as exc:
        t.result["llm"]["choice"] = {"status": exc.status, "code": exc.code}
        return None
    except (ValueError, TypeError):
        t.result["llm"]["choice"] = {"status": "llm_invalid_output", "code": "schedule_choice"}
        return None
    t.result["llm"]["choice"] = {"status": "ok", "raw": data}
    return {"choice": None if data["choice"] == "none" else ids.index(data["choice"]), "reasons": sorted(set(data["reasons"]))}


def keep_auto_range(draft: dict[str, Any], auto: dict[str, Any]) -> None:
    """(3차 검토 보완) 후보를 찾기 전에 지역·다른 날짜를 되물을 때, 사용자가 말한 날짜와 시간대('저녁')를 초안에 남긴다.
    시간대는 기존 초안의 pending 칸(문자열만 남기는 검사 _clean_draft 재사용)에 이름으로 둔다.
    다음 턴에 사용자가 새 날짜·시각을 말하지 않았을 때만 이어 쓴다. (정정한 내용이 우선)"""
    draft["date"] = auto["day"].isoformat()
    if auto.get("period") in od.PERIODS:
        draft["pending"] = {**draft["pending"], "period": auto["period"]}


def choose_slot(t: Turn, draft: dict[str, Any], auto: dict[str, Any], *, minutes: int, exclude_id: str | None,
                prefer: datetime | None, city: str | None, category: str | None,
                original_check: dict[str, Any] | None = None) -> dict[str, Any]:
    """{"start", "checks", "reason"}, {"ask", "awaiting"} 또는 {"none": 안내, "lines"}"""
    u = t.u
    day, window = auto["day"], auto["window"]
    if u["other_day_ok"]:
        draft["other_day_ok"] = True
    extra = sv.MAX_EXTRA_DAYS if draft["other_day_ok"] else 0
    # 1. 빈 시간 후보: 원하는 시각(같은 날이면 원래 시각, 아니면 범위의 가운데)에 가까운 순서로 최대 3개
    # 일정 자신을 빼고 찾으므로 원래 시각도 '빈 시간'으로 나온다. 원래 시각은 후보에서 뺀다. (같은 시각으로 바꾸는 제안 방지)
    anchor = prefer if prefer is not None and prefer.date() == day else _combine(day, f"{(window[0] + window[1]) // 2:02}:00")
    found = sv.find_free_slots(t.doc, day=day, minutes=minutes, now=t.now, prefer=anchor, window=window,
                               exclude_id=exclude_id, extra_days=extra, limit=sv.MAX_CANDIDATES + 1)
    starts = [s for s in found if s != prefer][:sv.MAX_CANDIDATES]
    if not starts:
        if not extra:
            keep_auto_range(draft, auto)
            return {"ask": f"{sv.korean_day(day, t.now)}에는 {minutes}분 동안 비어 있는 시간이 없어요. "
                           f"다른 날도 괜찮으시면 '다른 날도 괜찮아'라고 말씀해 주세요. (최대 {sv.MAX_EXTRA_DAYS}일 뒤까지 찾아볼게요)",
                    "awaiting": "other_day"}
        return {"none": "말씀하신 범위에서 빈 시간을 찾지 못했어요. 원하시는 날짜나 시각을 말씀해 주세요.", "lines": []}

    # 2. 시간 선택을 맡기지 않았고(시간대만 말함) 피하려는 날씨도 없으면 가장 가까운 빈 시간을 Python이 고른다. (LLM·예보 비교 없음)
    # (검토 보완) 피하려는 날씨를 말했으면 시간대만 말한 경우에도 아래 3·4의 예보 확인을 거친다.
    preferences = draft["preferences"]
    if not draft["delegate"] and not preferences:
        t.answer_source = "python_rule"
        return {"start": starts[0], "checks": [], "reason": REASON_WORDS["closer_to_original"] if prefer else "",
                "verified": False}

    # 3. 실외 일정이면 원래 시각과 후보의 예보를 새로 조회한다. 실외 일정이 아니면 날씨로 고르지 않는다.
    # (검토 보완) 사용자가 말한 날씨 조건은 지우지 않는다. 지역이 없으면 묻고, 실외 일정이 아니면 확인할 수 없다고 안내한다.
    if preferences and category == "outing" and not city:
        keep_auto_range(draft, auto)
        return {"ask": "어느 지역의 예보를 확인할까요? 지역을 알려 주시면 "
                       + "·".join(PREFERENCE_WORDS[p] for p in preferences) + " 예보를 확인해 시간을 찾을게요.",
                "awaiting": "city"}
    if preferences and category != "outing":
        words = "·".join(PREFERENCE_WORDS[p] for p in preferences)
        return {"none": f"실외 외출 일정으로 저장되어 있지 않아 {words} 예보로 시간을 고를 수 없어요. "
                        "일정은 바꾸지 않았어요. 원하시는 시각을 직접 말씀해 주세요.", "lines": []}
    outdoor = category == "outing" and bool(city) and (weather_needed(u, category, city) or bool(preferences))
    # (5-6 D21 보완) 피하려는 날씨가 있으면 원래 시각과 후보 모두 일정 전체 구간([시작, 시작+소요 시간))을 확인한다.
    # 날씨 조건 없이 맡긴 경우는 기존처럼 시작 시각 구간만 비교한다(일반 일정 동작 유지).
    def check_at(at: datetime) -> dict[str, Any]:
        return fetch_span(t, city, at, at + timedelta(minutes=minutes)) if preferences else fetch_forecast(t, city, at)
    original = original_check
    if outdoor and original is None and prefer is not None:
        original = check_at(prefer)
    checks = [check_at(s) for s in starts] if outdoor else \
        [{"status": "unknown", "signals": {}, "reason": "날씨를 비교하지 않았어요.", "weather": None, "at": ss.iso(s),
          "fetched_at": None} for s in starts]
    allowed = [n for n, c in enumerate(checks) if eligible(c, preferences)]

    # 4. LLM 선택 → Python 검증. 후보가 둘 이상이고 예보를 비교했을 때만 LLM에게 고르게 한다.
    picked, reasons = None, []
    if outdoor and len(starts) >= 2:
        answer = llm_choice(t, starts, checks, original, prefer, preferences)
        if answer is not None:
            index = answer["choice"]
            valid = (index is None and not allowed) or (
                index is not None and index in allowed and bool(answer["reasons"])
                and all(reason_holds(r, index, checks, original, starts, prefer) for r in answer["reasons"]))
            if valid:
                picked, reasons = index, answer["reasons"]
                t.answer_source = "llm_choice"
            else:
                t.rule("llm_choice_rejected_by_python_check")
        if t.answer_source != "llm_choice":
            t.answer_source = "python_fallback"
    if t.answer_source != "llm_choice":
        # Python 규칙: 조건에 맞는 후보 중 원하는 시각에 가장 가까운 것. 맞는 후보가 없으면 억지로 고르지 않는다.
        picked = allowed[0] if allowed else None
        if t.answer_source == "python_template":
            t.answer_source = "python_rule"
        if picked is not None:
            reasons = [r for r in CHOICE_REASONS if reason_holds(r, picked, checks, original, starts, prefer)]

    labelled = ([("원래 시각", original)] if original else []) + [(f"후보 {n}", c) for n, c in enumerate(checks, 1)]
    if picked is None:
        words = "·".join(PREFERENCE_WORDS[p] for p in preferences) or "조건"
        lines = [f"{words} 신호가 없다고 확인된 빈 시간을 찾지 못해 바꾸지 않았어요. 원하시는 시각을 직접 말씀해 주세요.",
                 "빈 시간 후보: " + ", ".join(when_text(s, t.now) for s in starts)] + weather_block(t, labelled)
        return {"none": "\n".join(lines), "lines": lines}
    reason = "시간 선택을 맡기셔서 고른 시각: " + ", ".join(REASON_WORDS[r] for r in reasons) if reasons else "시간 선택을 맡기셔서 고른 빈 시간"
    keep = ([("원래 시각", original)] if original else []) + [("바꿀 시각", checks[picked])]
    # verified: 피하려는 날씨가 있었고 그 조건을 예보로 확인한 후보 (offer_*가 다시 확인하지 않게 함)
    return {"start": starts[picked], "checks": keep if outdoor else [], "reason": reason, "verified": bool(preferences)}


# %% 승인 제안 만들기 함수 (변경·등록·취소·완료)
# 핵심 요약: schedule_service.propose_*()로 봉투를 만든다(action_id 1회 생성). 충돌이면 빈 시간 첫 후보로 다시 제안하고 그 사실을 알린다.
# 제안이 만들어진 뒤에만, 실외 일정이면 바꿀 시각의 예보와 (위험 신호가 있으면) 공식 수칙을 붙인다.
def _forecast_summary(checks: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """(5-6 D21 보완) 전체 구간을 확인한 결과는 구간마다 한 줄씩 남긴다. (첫 구간의 시각은 일정 시작 시각)"""
    rows = []
    for label, c in checks:
        for number, part in enumerate(c.get("parts") or [c]):
            rows.append({"label": label if number == 0 else f"{label}(이어지는 구간)", "at": c["at"] if number == 0 else part["at"],
                         "status": part["status"], "reason": part["reason"], "fetched_at": part["fetched_at"]})
    return rows


def preference_gate(t: Turn, draft: dict[str, Any], start: datetime, end: datetime, city: str | None, category: Any,
                    label: str) -> dict[str, Any]:
    """(2차 검토 보완) 사용자가 피하려는 날씨를 말했는데 시각을 직접·상대 시간·충돌 대안으로 정한 경우, 그 시각의 예보를 확인한다.
    {"checks": [(label, 예보)]}(조건 확인됨), {"ask", "awaiting"}(지역 질문) 또는 {"none": 안내}(확인 못함·조건에 맞지 않음)
    choose_slot()과 같은 기준(eligible)을 쓴다. 확인하지 못한 예보를 '비가 안 온다'로 보지 않는다.
    (5-6 D21 보완) 시작 시각만이 아니라 일정 전체 구간 [start, end)의 정시 예보를 모두 확인한다."""
    preferences = draft["preferences"]
    words = "·".join(PREFERENCE_WORDS[p] for p in preferences)
    when = when_text(start, t.now)
    if category != "outing":
        return {"none": f"실외 외출 일정으로 저장되어 있지 않아 {words} 예보를 확인할 수 없어요. 일정은 바꾸지 않았어요. "
                        "날씨와 관계없이 그 시각으로 정하시려면 다시 말씀해 주세요."}
    if not city:
        draft["pending"] = {"start": ss.iso(start)}     # 지역을 받으면 같은 시각으로 다시 확인한다
        return {"ask": f"{when}의 {words} 예보를 확인하려면 지역이 필요해요. 어느 지역인지 알려 주세요.", "awaiting": "city"}
    check = fetch_span(t, city, start, end)
    lines = weather_block(t, [(label, check)])
    if check["status"] == "unknown":
        t.rule("preference_unknown_not_treated_as_clear")
        return {"none": "\n".join([f"{when}의 예보를 확인하지 못해 {words} 조건을 판단할 수 없어 제안하지 않았어요. "
                                   f"({check['reason']}) 다른 시각을 말씀해 주세요."] + lines)}
    if not eligible(check, preferences):
        t.rule("preference_not_met_at_requested_time")
        return {"none": "\n".join([f"{when}에는 {words} 예보가 있어 제안하지 않았어요. 다른 시각을 말씀하시거나 "
                                   "'알아서 바꿔 줘'라고 말씀해 주세요."] + lines)}
    return {"checks": [(label, check)]}


def _forecast_city(item: dict[str, Any], draft: dict[str, Any]) -> str | None:
    """예보를 조회할 지역: 일정에 저장된 지역, 없으면 이번 상담에서 사용자가 말한 지역 (일정의 지역은 바꾸지 않음)"""
    return item["city"] or draft["city"]


def _attach_weather(t: Turn, action: dict[str, Any], checks: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """제안 봉투에 확인한 예보 줄·공식 안내를 넣고, 답변에 붙일 줄을 돌려준다. (승인 전 봉투라 저장 대조에 영향 없음)"""
    lines = weather_block(t, checks)
    guidance = safety_guidance(t, checks[-1][1]) if checks else []
    action["payload"].update(weather_lines=lines, official_guidance=guidance, forecast=_forecast_summary(checks))
    return lines + guidance_lines(guidance)


def waived_notice(draft: dict[str, Any], action: dict[str, Any]) -> None:
    """(5-6 D34 보완) 사용자가 명시적으로 철회한 날씨 조건이 있으면 카드에 그 사실을 적는다. (승인 전 봉투라 저장 대조에 영향 없음)"""
    if draft["waived"]:
        words = "·".join(PREFERENCE_WORDS[p] for p in draft["waived"])
        action["payload"]["notes"].append(f"말씀하신 대로 날씨 조건({words})을 빼고 제안해요. (예보로 그 조건을 확인하지 않았어요)")


def conflict_notice(message: str, options: list[datetime], *, weather_pending: bool) -> str:
    """충돌 안내. (3차 검토 보완) 날씨 조건을 아직 확인하지 않은 대체 시각은 '후보'라고만 쓰고, '제안'이라는 말은 쓰지 않는다.
    날씨 조건이 없으면 바로 승인 제안이 되므로 기존 문장('~로 제안해요')을 그대로 쓴다."""
    first = sv.korean_time(options[0])
    text = (f"{message} 그래서 비어 있는 {first}을(를) 대체 후보로 골라 예보를 확인해요." if weather_pending
            else f"{message} 그래서 비어 있는 {first}로 제안해요.")
    if options[1:]:
        text += " (다른 빈 시간: " + ", ".join(sv.korean_time(a) for a in options[1:]) + ")"
    return text


def offer_update(t: Turn, draft: dict[str, Any], item: dict[str, Any], start: datetime, *, approximate: bool = False,
                 inherited: bool = False, checks: list[tuple[str, dict[str, Any]]] | None = None, reason: str = "",
                 notice: str = "", verified: bool = False) -> None:
    """verified: choose_slot()이 날씨 조건을 이미 확인한 후보이면 True"""
    u = t.u
    old_start, old_end = sv.interval(item)
    minutes = int((old_end - old_start).total_seconds() // 60)
    reason = u["reason_text"] or draft["reason"] or reason or "사용자 요청"
    draft["reason"] = u["reason_text"] or draft["reason"]
    made = sv.propose_update(t.doc, item, start, t.now, reason=reason, approximate=approximate, inherited=inherited)
    # 1. 충돌: 같은 날 빈 시간 중 가장 가까운 시각으로 다시 제안한다. (보호 일정은 옮기지 않음)
    if not made["ok"] and made["code"] == "conflict" and made.get("alternatives"):
        options = made["alternatives"]
        # (5-6 D21 보완) 시각이 바뀌므로 앞에서 확인한 날씨 결과(verified)를 쓰지 않고 바뀐 시각을 다시 확인한다.
        verified = False
        notice = conflict_notice(made["message"], options, weather_pending=bool(draft["preferences"]))
        t.rule("conflict_alternative_offered")
        start, approximate, inherited, checks = options[0], False, False, None
        made = sv.propose_update(t.doc, item, start, t.now, reason=reason)
    if not made["ok"]:
        if made["code"] in ("same_time", "past_time", "conflict"):
            extra = " 그날은 빈 시간이 없어요." if made["code"] == "conflict" else ""
            return ask(t, draft, "time", f"{made['message']}{extra} 다른 시각을 말씀해 주세요.")
        return reply(t, made["message"], "schedule_refusal")

    # 2. (2차 검토 보완) 피하려는 날씨를 말했으면 직접·상대·충돌 대안 시각도 제안 전에 예보로 확인한다. 확인하지 못하면 제안하지 않는다.
    action = made["action"]
    city = _forecast_city(item, draft)
    if draft["preferences"] and not verified:
        # (5-6 D21 보완) 바꿀 시각은 일정 전체 구간(원래 소요 시간 그대로)을, 원래 시각도 전체 구간을 확인한다(같은 구간은 다시 조회 안 함).
        gate = preference_gate(t, draft, start, start + (old_end - old_start), city, item["category"], "바꿀 시각")
        if "ask" in gate:
            return ask(t, draft, gate["awaiting"], "\n".join(([notice] if notice else []) + [gate["ask"]]))
        if "none" in gate:
            return ask(t, draft, "time", "\n".join(([notice] if notice else []) + [gate["none"]]))
        checks = [("원래 시각", fetch_span(t, city, old_start, old_end))] + gate["checks"]
    # 3. 실외 일정: 원래 시각과 바꿀 시각의 예보 (이미 조회했으면 다시 조회하지 않음)
    if checks is None and weather_needed(u, item["category"], item["city"]):
        checks = [("원래 시각", fetch_forecast(t, item["city"], old_start)), ("바꿀 시각", fetch_forecast(t, item["city"], start))]
    extra_lines = _attach_weather(t, action, checks or [])
    if checks and city and city != item["city"]:
        extra_lines.append(f"(예보는 말씀하신 지역 '{city}' 기준이에요. 일정에 저장된 지역은 바꾸지 않아요.)")
    draft.update(target_id=item["schedule_id"], date=start.date().isoformat(), time=f"{start:%H:%M}", pending={})
    new_when = sv.korean_time(start) if start.date() == old_start.date() else when_text(start, t.now)
    waived_notice(draft, action)
    lines = ([notice] if notice else []) + [
        f"'{item['title']}' 일정을 {when_text(old_start, t.now)}에서 {new_when}(으)로 바꾸는 것을 제안해요. "
        f"(소요 시간 {minutes}분 그대로)", f"이유: {reason}"] + action["payload"]["notes"] + t.notices + extra_lines
    propose(t, draft, action, lines)


def offer_create(t: Turn, draft: dict[str, Any], start: datetime, *, approximate: bool = False,
                 checks: list[tuple[str, dict[str, Any]]] | None = None, notice: str = "", verified: bool = False) -> None:
    """verified: choose_slot()이 날씨 조건을 이미 확인한 후보이면 True"""
    u = t.u
    title, city, category = draft["title"], draft["city"], draft["category"] or "other"
    made = sv.propose_create(t.doc, title=title, start=start, minutes=draft["minutes"], now=t.now, city=city,
                             category=category, reason=u["reason_text"], approximate=approximate)
    if not made["ok"] and made["code"] == "conflict" and made.get("alternatives"):
        options = made["alternatives"]
        # (5-6 D21 보완) 시각이 바뀌므로 앞에서 확인한 날씨 결과(verified)를 쓰지 않고 바뀐 시각을 다시 확인한다.
        verified = False
        notice = conflict_notice(made["message"], options, weather_pending=bool(draft["preferences"]))
        t.rule("conflict_alternative_offered")
        start, checks = options[0], None
        made = sv.propose_create(t.doc, title=title, start=start, minutes=draft["minutes"], now=t.now, city=city,
                                 category=category, reason=u["reason_text"])
    if not made["ok"]:
        if made["code"] in ("past_time", "conflict"):
            return ask(t, draft, "time", f"{made['message']} 다른 시각을 말씀해 주세요.")
        if made["code"] == "title_needed":
            return ask(t, draft, "title", made["message"])
        if made["code"] == "bad_duration":
            return ask(t, draft, "duration", made["message"])
        return reply(t, made["message"], "schedule_refusal")
    action = made["action"]
    # (2차 검토 보완) '오후 두 시에 비 안 오면 등록'처럼 시각을 직접 정했어도 날씨 조건은 제안 전에 예보로 확인한다.
    if draft["preferences"] and not verified:
        # (5-6 D21 보완) 등록할 일정의 전체 구간(제안 봉투의 종료 시각까지)을 확인한다.
        end = ss.parse_iso(action["payload"]["after"]["end_at"]) or start + timedelta(minutes=sv.DEFAULT_MINUTES)
        gate = preference_gate(t, draft, start, end, city, category, "일정 시각")
        if "ask" in gate:
            return ask(t, draft, gate["awaiting"], "\n".join(([notice] if notice else []) + [gate["ask"]]))
        if "none" in gate:
            return ask(t, draft, "time", "\n".join(([notice] if notice else []) + [gate["none"]]))
        checks = gate["checks"]
    if checks is None and weather_needed(u, category, city):
        checks = [("일정 시각", fetch_forecast(t, city, start))]
    extra_lines = _attach_weather(t, action, checks or [])
    after = action["payload"]["after"]
    end = ss.parse_iso(after["end_at"]) or start
    minutes = int((end - start).total_seconds() // 60)
    draft.update(date=start.date().isoformat(), time=f"{start:%H:%M}")
    waived_notice(draft, action)
    lines = ([notice] if notice else []) + [
        f"{when_text(start, t.now)}에 '{title}'을(를) 새 일정으로 등록하는 것을 제안해요. (소요 시간 {minutes}분"
        + (f", 지역 {city}" if city else "") + ")"] + action["payload"]["notes"] + t.notices + extra_lines
    propose(t, draft, action, lines)


# %% 작업별 처리 함수
# 핵심 요약: 대상 확인 → 필요한 정보 확인(한 번에 하나만 묻기) → 계산 → 제안 순서로 처리한다. 저장은 하지 않는다.
def _target_for(t: Turn, draft: dict[str, Any], operation: str) -> dict[str, Any]:
    """초안에 이미 정한 대상이 있으면 그대로 쓰고(최신 저장소에서 다시 읽음), 아니면 새로 찾는다."""
    item = item_by_id(t.doc, draft["target_id"]) if draft["target_id"] else None
    if item is not None and item["status"] == "scheduled" and not t.u["exclude_previous_target"]:
        return {"item": item}
    if draft["target_id"] and item is None:
        draft["target_id"] = None
        t.result["limitations"].append("앞서 고른 일정이 저장소에 없어 다시 찾았어요.")
    return resolve_target(t, draft, operation)


CHANGE_WORDS = ("new_date_text", "new_time_text", "shift_text", "shift_base")


def ambiguous_city(t: Turn, draft: dict[str, Any], kind: str | None = None) -> bool:
    """(검토 보완) '광주'처럼 시·도 없이는 정할 수 없는 지역은 1단계 규칙(AMBIGUOUS_CITY_NAMES)을 따라 확정하지 않고 묻는다.
    (5-6 D1 보완) kind(create·update)를 주면 묻기 전에 이번 턴에 말한 날짜·시각·대상 표현을 초안에 보관한다.
    물었으면 True (등록·변경 공통)"""
    city = draft["city"]
    if not isinstance(city, str) or city.strip().lower() not in AMBIGUOUS_CITY_NAMES:
        return False
    options = AMBIGUOUS_CITY_NAMES[city.strip().lower()]
    draft["city"] = None
    if kind:
        hold_said(t, draft, kind)
    # (5-6 F3 보완) 이 지역이 확정될 때까지 지역을 뺀 카드를 만들지 않도록 표시한다(문자열만 담는 기존 pending 칸).
    draft["pending"] = {**draft["pending"], CITY_REQUIRED: city.strip(), CITY_TRIES: "1"}
    ask(t, draft, "city", f"어느 지역인지 정할 수 없어요. {options} 중 어디인지 알려 주세요.")
    return True


def city_unconfirmed(t: Turn, draft: dict[str, Any], kind: str) -> bool:
    """(5-6 F3 보완) 확인 중인 지역(사용자가 말한 지역·모호한 지역)이 아직 정해지지 않았으면 카드를 만들지 않는다. 물었거나 끝냈으면 True.
    지역 질문을 한 번 더 하고, 그래도 정해지지 않으면 제안을 보류하고 초안을 지운다(지역을 임의로 빼거나 고르지 않음)."""
    said = draft["pending"].get(CITY_REQUIRED)
    if not said:
        return False
    if draft["city"]:
        draft["pending"] = {k: v for k, v in draft["pending"].items() if k not in (CITY_REQUIRED, CITY_TRIES)}
        return False
    if draft["pending"].get(CITY_TRIES, "1") != "1":
        t.rule("city_unconfirmed_proposal_held")
        reply(t, "지역을 확인하지 못해 일정을 제안하지 않았어요. 지역을 정해서 다시 말씀해 주세요.")
        return True
    hold_said(t, draft, kind)
    draft["pending"] = {**draft["pending"], CITY_TRIES: "2"}
    options = AMBIGUOUS_CITY_NAMES.get(said.lower())
    ask(t, draft, "city", f"지역을 확인하지 못했어요. {options} 중 어디인지 지역 이름만 말씀해 주세요." if options
        else f"지역을 확인하지 못했어요. 말씀하신 '{said}'처럼 지역 이름만 다시 알려 주세요.")
    return True


def ask_said_region(t: Turn, draft: dict[str, Any]) -> bool:
    """(5-6 F3 보완) 등록에서 사용자가 지역 이름('인천에서')을 말했는데 모델이 지역을 비웠으면 카드 전에 지역을 묻는다. 물었으면 True"""
    region = said_region(t.message)
    if draft["city"] or not region or draft["pending"].get(CITY_REQUIRED):
        return False
    hold_said(t, draft, "create")
    draft["pending"] = {**draft["pending"], CITY_REQUIRED: region, CITY_TRIES: "1"}
    t.rule("said_region_not_extracted_asked")
    ask(t, draft, "city", f"말씀에 지역 이름({region})이 있어요. 일정 지역을 알려 주세요.")
    return True


# %% (5-6 D1 보완) 모호한 지역을 되묻는 동안 이번 턴의 표현 보관·복원
# 핵심 요약: 지역을 되묻고 바로 끝내면 그 턴에 말한 날짜·시각·대상 표현이 다음 턴에 남지 않았다(D1).
# 묻기 전에 사용자가 말한 표현(발화 근거 확인을 거친 문자열)을 초안 pending의 'hold:' 칸에 보관하고(문자열만 남기는 _clean_draft 그대로),
# 다음 턴에 이어 쓴다. 같은 묶음(날짜·시각·대상)을 이번 턴에 새로 말했으면 새 값이 우선한다(정정 우선, 묶음 안의 표현은 서로 배타적).
HOLD = "hold:"
HELD_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
    "create": {"date": ("date_text",), "time": ("time_text", "shift_text", "shift_base"),
               "other": ("duration_text", "reason_text")},
    "update": {"target": ("target_text", "date_text", "time_text"), "date": ("new_date_text",),
               "time": ("new_time_text", "shift_text", "shift_base"), "other": ("reason_text",)},
}
TIME_SAID = {"create": ("time_text", "shift_text"), "update": ("new_time_text", "shift_text")}


DATE_WORDS = ("date_text", "new_date_text")


def _fixed_day(text: str, now: datetime) -> str:
    """(5-6 D12 보완) '내일·모레·금요일'처럼 지금이 기준인 날짜 표현은 말한 순간(KST)의 달력 날짜(ISO)로 바꿔 보관한다.
    자정을 넘겨 대답해도 날짜가 밀리지 않는다. '그날·다음 날'처럼 앞 대화가 기준이거나 정할 수 없는 표현은 그대로 둔다."""
    parsed = sv.parse_day(text, now)
    return parsed["date"] if parsed and "date" in parsed else text


def hold_said(t: Turn, draft: dict[str, Any], kind: str) -> None:
    """지역·날짜·제목·소요 시간을 되묻기 직전에 이번 턴의 표현을 보관한다.
    '지금부터' 상대 시각은 말한 순간 기준의 절대 시각으로, 지금 기준 날짜는 말한 순간의 달력 날짜로 보관한다."""
    u = t.u
    # 1. 이전 보관 값은 지우고, 이번 턴(앞 턴에서 이어 쓴 값 포함)의 표현으로 다시 보관한다.
    pending = {k: v for k, v in draft["pending"].items() if not k.startswith(HOLD)}
    for keys in HELD_GROUPS[kind].values():
        for key in keys:
            value = u.get(key)
            if isinstance(value, str) and value.strip() and value != "unknown":
                pending[HOLD + key] = _fixed_day(value, t.now) if key in DATE_WORDS else value
    # 2. 지금 기준 상대 시각은 지역 질문을 거치는 동안 기준이 밀리지 않게 정상 계산된 시각만 보관한다. (계산 실패·모호하면 표현 그대로)
    base = u.get("shift_base")
    if u.get("shift_text") and (base == "now" or (kind == "create" and base != "schedule")):
        moved = sv.resolve_shift(u["shift_text"], t.now)
        if moved and isinstance(moved.get("at"), datetime):
            pending[HOLD + "start"] = ss.iso(moved["at"])
            pending.pop(HOLD + "shift_text", None)
            pending.pop(HOLD + "shift_base", None)
            draft["date"] = moved["at"].date().isoformat()     # 시각만 정정하면 이 날짜를 이어 쓴다
    # 3. 앞 턴에서 이어 받은 절대 시각이 아직 쓰이지 않았으면 다시 보관한다. (지역을 또 모호하게 답한 경우)
    if not any(u.get(k) for k in TIME_SAID[kind]) and isinstance(pending.get("start"), str) and HOLD + "start" not in pending:
        pending[HOLD + "start"] = pending.pop("start")
    draft["pending"] = pending
    draft["other_day_ok"] = draft["other_day_ok"] or bool(u.get("other_day_ok"))


def restore_held(t: Turn, draft: dict[str, Any], kind: str) -> dict[str, Any]:
    """지역·날짜·제목·소요 시간 질문 뒤 보관한 표현을 이어 쓴다. 이번 턴에 같은 묶음을 말했으면 보관 값을 버린다.
    날짜만 새로 말했으면 보관한 상대 시각은 버린다(새 날짜와 상대 시각을 임의로 조합하지 않고 시각을 다시 묻게 함).
    규칙 기록 이름은 직전에 물은 항목을 따른다(지역 질문이면 city_question_held_restored)."""
    held = {k[len(HOLD):]: v for k, v in draft["pending"].items() if k.startswith(HOLD)}
    if not held:
        return t.u
    draft["pending"] = {k: v for k, v in draft["pending"].items() if not k.startswith(HOLD)}
    u = dict(t.u)
    groups = HELD_GROUPS[kind]

    def said(name: str) -> bool:
        keys = TIME_SAID[kind] if name == "time" else groups[name]
        return any(isinstance(u.get(k), str) and u[k].strip() and u[k] != "unknown" for k in keys)
    # 1. 버릴 묶음: 이번 턴에 새로 말한 묶음, 날짜만 바꾼 경우의 상대 시각, 대상을 바꾼 경우의 이전 대상 표현
    relative = bool(held.get("shift_text") or held.get("start"))
    drop = {name for name in groups if name != "other" and said(name)}
    if said("date") and relative:
        drop.add("time")
    if kind == "update" and u.get("exclude_previous_target"):
        drop.add("target")
    # 2. 남은 묶음은 이번 턴의 빈 칸에 채운다. 절대 시각은 기존 pending['start'] 이어 쓰기 경로로 옮긴다.
    for name, keys in groups.items():
        if name in drop:
            continue
        for key in keys:
            if key in held and not (isinstance(u.get(key), str) and u[key].strip() and u[key] != "unknown"):
                u[key] = held[key]
    if "start" in held and "time" not in drop:
        draft["pending"] = {**draft["pending"], "start": held["start"]}
    t.u = u
    t.rule(f"{draft['awaiting'] or 'follow_up'}_question_held_restored")
    return u


# %% (5-6 D34·D42·D43 보완) 확인 질문과 대답 처리 함수
# 핵심 요약: 날씨 조건 누락·위임·취소 부정이 모호하면 카드를 만들지 않고 한 번에 하나만 묻는다(최초 + 다시 묻기 1회).
# 대답의 뜻은 확인 단계 LLM이 고르고, Python은 근거 부족과 명확한 불일치만 막는다. 확인 긍정은 저장 승인이 아니다(저장은 승인 버튼으로만).
def guard_conditions(t: Turn, draft: dict[str, Any], kind: str) -> bool:
    """발화가 말한 날씨 조건 중 이해 결과·초안에 없는 조건이 있으면 그 조건으로 묻는다. 물었으면 True.
    부정 요청(유지)과 조건이 하나도 없는 조건부 변경('어떤 날씨일 때' 질문)은 기존 경로가 막으므로 보지 않는다. check_weather는 조건이 아니다.
    (5-6 D50 보완) 조건부 변경에서 일부 조건만 추출됐으면(발화에는 비·바람) 빠진 조건을 함께 확인할지 묻는다."""
    if t.u["request_mode"] == "negated" or (draft["conditional"] and not draft["preferences"]):
        return False
    missing = [p for p in said_conditions(t.message) if p not in draft["preferences"] and p not in draft["waived"]]
    if not missing:
        return False
    # 1. 처음 요청에서 이미 명시적으로 철회했으면('비 와도 상관없어') 묻지 않고 해제로 기록한다.
    for p in explicit_waiver(t.message, missing):
        draft["waived"] = sorted(set(draft["waived"]) | {p})
        t.rule(f"weather_condition_withdrawn:{p}")
    missing = [p for p in missing if p not in draft["waived"]]
    if not missing:
        return False
    # 2. 카드를 만들지 않고 빠진 조건의 말로 묻는다. 이미 말한 날짜·시각·대상은 보관한다(D1·D13 방식).
    hold_said(t, draft, kind)
    draft.update(asked=missing, confirm_tries=1)
    t.rule("weather_condition_unconfirmed_asked")
    ask(t, draft, "weather_condition", condition_question(draft, missing, kind))
    return True


def condition_question(draft: dict[str, Any], conditions: list[str], kind: str) -> str:
    """빠진 날씨 조건 질문. 조건부 변경('비 오면 바꿀까?')은 '그 날씨도 함께 확인할까요?'로 묻는다(D50)."""
    if not draft["conditional"]:
        return weather_question(conditions, kind)
    names = "·".join(PREFERENCE_WORDS[p] for p in conditions)
    return (f"말씀에 날씨 이야기({names})도 있어요. {names} 조건도 함께 확인할까요? "
            "함께 확인하시려면 '응', 빼시려면 '날씨 상관없이'라고 말씀해 주세요.")


def settle_weather(t: Turn, draft: dict[str, Any], kind: str) -> str:
    """날씨 조건 확인 질문의 대답 처리: continue(진행) | stop(다시 물었거나 끝냄) | cancel(요청 그만두기)"""
    asked = list(draft["asked"])
    if not asked:
        draft["awaiting"] = None
        return "continue"
    answer, evidence = understand_confirm(t, draft)
    # 1. 대답을 해석한다. 모델이 조건을 다시 냈으면(근거 확인됨) 그 조건은 확정한다.
    confirmed = [p for p in asked if p in t.u["preferences"]]
    waived: list[str] = []
    if answer == "yes" and not confirm_conflict(t, answer, evidence, "weather_condition"):
        confirmed = asked
    elif answer == "yes":
        t.rule("confirm_answer_conflict")
    elif answer == "no_condition":
        # (5-6 D46 보완) 철회 여부는 발화 전체로 판단하고(부분 근거 '비 와도'만으로 판단하지 않음), 근거 표현은 철회할 조건의 범위만 정한다.
        # 근거가 없으면 철회하지 않는다.
        scope = [p for p in asked if CONDITION_WORDS[p].search(evidence)] or (list(asked) if WEATHER_GENERIC.search(evidence) else [])
        waived = [p for p in explicit_waiver(t.message, asked) if p in scope and p not in confirmed]
        if not waived:
            t.rule("withdrawal_not_verified")
    elif answer == "cancel" and t.u["request_mode"] == "negated":
        return "cancel"
    draft["preferences"] = sorted(set(draft["preferences"]) | set(confirmed))
    draft["waived"] = sorted(set(draft["waived"]) | set(waived))
    if confirmed:
        t.rule("weather_condition_confirmed")
    for p in waived:
        t.rule(f"weather_condition_withdrawn:{p}")
    remaining = [p for p in asked if p not in confirmed and p not in waived]
    if not remaining:
        draft.update(asked=[], confirm_tries=0, awaiting=None)
        return "continue"
    # 2. 확정하지 못한 조건은 다시 묻는다. 상한을 넘으면 카드 없이 끝낸다.
    if draft["confirm_tries"] >= CONFIRM_LIMIT:
        t.rule("weather_condition_unresolved")
        reply(t, "날씨 조건을 확인하지 못해 제안하지 않았어요. 원하시는 조건과 함께 다시 말씀해 주세요.", "schedule_condition_unknown")
        return "stop"
    hold_said(t, draft, kind)
    draft.update(asked=remaining, confirm_tries=draft["confirm_tries"] + 1)
    lead = "날씨 조건을 빼고 진행할까요, 아니면 요청을 그만둘까요? " if answer == "no" else ""
    ask(t, draft, "weather_condition", lead + condition_question(draft, remaining, kind))
    return "stop"


def settle_delegate(t: Turn, draft: dict[str, Any]) -> None:
    """(D42) '골라 드릴까요?' 대답: 시각을 말했으면 그 시각, 충돌 없는 긍정이면 위임, 그 밖은 위임하지 않고 시각을 묻는다."""
    u = t.u
    if any(u[k] for k in ("new_time_text", "time_text", "shift_text", "new_date_text")) or sv.parse_clock(t.message):
        draft["awaiting"] = "time"
        return
    answer, evidence = understand_confirm(t, draft)
    if answer == "yes" and not confirm_conflict(t, answer, evidence, "delegate_confirm"):
        draft["delegate"] = True
        t.rule("delegation_confirmed")
    else:
        draft["delegate_declined"] = True
        t.rule("delegation_not_confirmed")
    draft["awaiting"] = None


def delegate_question(draft: dict[str, Any]) -> str:
    asks = ", ".join(CONDITION_ASK[p] for p in draft["preferences"])
    return f"{asks}의 빈 시간을 제가 골라 드릴까요? 원하시는 시각이 있으면 그 시각을 말씀해 주세요."


def wants_delegate_confirm(draft: dict[str, Any]) -> bool:
    """날씨 조건은 확정됐는데 이 초안에서 시각을 한 번도 정하지 않았고 위임 표현도 없을 때 (위임을 확정하지 않고 먼저 묻는다).
    앞에서 시각을 말했는데 날짜 정정 등으로 다시 물어야 하는 경우는 기존처럼 시각을 묻는다."""
    return bool(draft["preferences"]) and not draft["time"] and not draft["delegate"] and not draft["delegate_declined"]


def ask_cancel_confirm(t: Turn, *, weather: bool = False) -> bool:
    """(D43) '안 갈래'가 취소 요청인지 단순 부정인지 모호하면(cancel + negated) 대상이 하나일 때 취소 여부를 묻는다. 물었으면 True
    (5-6 D50 보완) weather=True: 날씨 낱말이 있는 취소인데 모델이 조건을 표시하지 않았을 때, 날씨와 관계없이 취소할지 묻는다."""
    draft = new_draft("cancel")
    target = resolve_target(t, draft, "cancel")
    item = target.get("item")
    if item is None or item["status"] != "scheduled" or not item["reschedulable"]:
        return False
    draft.update(target_id=item["schedule_id"], confirm_tries=1)
    t.rule("cancel_weather_confirm_asked" if weather else "cancel_negation_confirm_asked")
    # (5-6 F1 보완) 조건부 취소는 지원하지 않는다는 사실을 먼저 밝히고, 날씨와 관계없이 지금 바로 취소할지만 묻는다.
    # 물어본 날씨 조건을 기존 asked 칸에 남겨, 대답에 조건을 다시 붙이면('응, 비 오면 취소해') 동의로 보지 않게 한다.
    if weather:
        draft["asked"] = said_conditions(t.message)
    lead = "날씨에 따라 취소하는 기능(예: 비가 오면 취소)은 아직 없어요. 날씨와 관계없이 " if weather else ""
    when = when_text(sv.interval(item)[0], t.now)
    ask(t, draft, "cancel_confirm", f"{lead}'{item['title']}'({when}) 일정을 {'지금 바로 ' if weather else ''}취소할까요? "
                                    "취소하시려면 '응', 그대로 두시려면 '아니'라고 말씀해 주세요.")
    return True


def settle_cancel(t: Turn, draft: dict[str, Any]) -> bool:
    """취소 확인 대답 처리. 진행하면 True(취소 제안), 아니면 답변·질문을 남기고 False"""
    answer, evidence = understand_confirm(t, draft)
    # (5-6 F1 보완) 날씨 취소 확인에서 대답에 조건을 다시 붙였으면('비 오면 취소해') 명시적으로 풀지 않은 한 즉시 취소 동의가 아니다.
    said = said_conditions(t.message) if draft["asked"] else []
    reattached = bool(set(said) - set(explicit_waiver(t.message, said)))
    if answer == "yes" and not confirm_conflict(t, answer, evidence, "cancel_confirm") and not reattached:
        draft["awaiting"] = None
        t.rule("cancel_confirmed")
        # (5-6 F1 보완) 날씨 취소 확인을 통과한 동의는 '날씨와 관계없이 취소'라는 명시적 조건 철회다. 다른 철회와 같은 규칙 이름으로 남겨
        # 측정 도구가 모델의 조건 누락과 구분하게 한다(검증된 철회만 기록, 동작은 바꾸지 않음).
        for p in draft["asked"]:
            t.rule(f"weather_condition_withdrawn:{p}")
        return True
    if answer in ("no", "cancel"):
        reply(t, "알겠어요. 일정은 취소하지 않고 그대로 둘게요.", "schedule_kept")
        return False
    if draft["confirm_tries"] >= CONFIRM_LIMIT:
        reply(t, "취소하실지 확인하지 못해 일정은 그대로 둘게요.", "schedule_kept")
        return False
    draft["confirm_tries"] += 1
    ask(t, draft, "cancel_confirm", draft["question"])
    return False


def answer_fits(t: Turn, draft: dict[str, Any]) -> bool:
    """(5-6 D36·D37 보완) 이번 발화가 Python이 물어본 항목의 대답으로 읽히는지 발화·이해 결과로 확인한다.
    (answers_previous_question 하나만 믿지 않음) 확인 질문은 짧은 대답이면 확인 단계가 뜻을 판단한다."""
    awaiting = draft["awaiting"]
    if awaiting in CONFIRM_KINDS:
        return "?" not in t.message and len(t.message) <= CONFIRM_ANSWER_LIMIT
    if "?" in t.message or len(t.message) > SHORT_ANSWER_LIMIT:
        return False
    if awaiting == "city":
        return bool(t.u["city_text"])
    if awaiting == "title":
        return bool(t.u["title"])
    if awaiting == "duration":
        return sv.parse_minutes(t.u["duration_text"] or t.message) is not None
    return _short_answer_fits(t, draft)


def new_request_signal(t: Turn, draft: dict[str, Any]) -> bool:
    """대답 속에 새 요청이 섞였는지: 등록은 다른 이름(이름 질문의 대답·정정 제외), 변경은 다른 대상(대상 정정 제외)
    (5-6 D49 보완) 확인 질문(날씨 조건·위임·취소)은 이름·대상을 묻지 않으므로, 그 대답에 다른 이름·대상이 있으면 정정 표기라도 새 요청이다."""
    u = t.u
    confirming = draft["awaiting"] in CONFIRM_KINDS
    if u["request_mode"] == "correction" and not confirming:
        return False
    other = (u["target_text"] if draft["operation"] == "create" else u["title"]) if confirming else ""
    # 실측에서 Qwen은 지역 대답('인천')을 target_text와 city_text에 함께 적었다. 지역 표현과 같은 값은 새 요청 신호가 아니다.
    if other and _squeeze(other) != _squeeze(u["city_text"] or ""):
        return True
    if draft["operation"] == "create":
        return bool(u["title"] and draft["title"] and u["title"] != draft["title"] and draft["awaiting"] != "title")
    current = item_by_id(t.doc, draft["target_id"]) if draft["target_id"] else None
    return bool(u["target_text"] and current is not None and not sv.text_matches(current, u["target_text"])
                and not u["exclude_previous_target"])


def handle_update(t: Turn) -> None:
    """일정 시각 변경 (상대 시간·직접 지정·시간대·선택 위임·날씨 조건)"""
    draft = t.draft if t.draft and t.draft["operation"] == "update" else new_draft("update")
    # (5-6 D1 보완) 모호한 지역을 되묻느라 보관한 대상·새 시각 표현을 이어 쓴다. 이번 턴에 새로 말한 값이 우선한다.
    u = restore_held(t, draft, "update")
    # (5-6 D34·D42 보완) 날씨 조건·위임 확인 질문의 대답을 먼저 처리한다.
    answering = draft["awaiting"] in ("weather_condition", "delegate_confirm")
    if draft["awaiting"] == "weather_condition":
        outcome = settle_weather(t, draft, "update")
        if outcome == "cancel":
            return handle_negated(t, "update")
        if outcome == "stop":
            return
    elif draft["awaiting"] == "delegate_confirm":
        settle_delegate(t, draft)
    if answering:
        # (5-6 D53 보완) 확인 질문의 대답 턴에서는 모델의 위임 표기(delegate_choice)를 다시 쓰지 않는다. 위임은 위임 확인 단계의
        # 근거 있는 yes로만 확정하고, 날씨 조건 확인 대답의 위임 표기는 위임 확인 질문(D42)으로 넘긴다. 직접 말한 시각은 그대로 쓴다.
        t.u = dict(t.u, delegate_choice=False)
    u = t.u
    merge_conditions(draft, u)      # (5-6 D44 보완) 새 조건은 더한다(바꿔치기하지 않음)
    draft["check_weather"] = draft["check_weather"] or u["check_weather"]
    draft["conditional"] = draft["conditional"] or u["request_mode"] == "conditional"
    draft["delegate"] = draft["delegate"] or u["delegate_choice"]
    # (2차 검토 보완) 지역 질문에 대한 대답('인천')은 초안에 남겨 예보 조회에만 쓴다. 일정에 저장된 지역은 바꾸지 않는다.
    if u["city_text"]:
        draft["city"] = u["city_text"]
    if ambiguous_city(t, draft, "update") or city_unconfirmed(t, draft, "update"):
        return
    if guard_conditions(t, draft, "update"):        # (5-6 D34 보완) 말한 날씨 조건을 모델이 빠뜨렸으면 카드 전에 묻는다
        return

    # 1. 대상 일정. 대상을 묻는 동안 사용자가 함께 말한 새 시각 표현은 초안에 보관했다가 대상이 정해지면 쓴다.
    target = _target_for(t, draft, "update")
    if "ask" in target:
        draft["pending"] = {**draft["pending"], **{k: u[k] for k in CHANGE_WORDS if u[k] and u[k] != "unknown"}}
        return ask(t, draft, target["awaiting"], target["ask"], candidates=target["candidates"])
    if "refuse" in target:
        return reply(t, target["refuse"], "schedule_refusal")
    item = target["item"]
    draft.update(target_id=item["schedule_id"], candidates=[])
    if not item["reschedulable"]:
        return reply(t, f"'{item['title']}'은(는) 변경 제한 일정이라 SafeMate가 바꾸지 않아요. 필요하면 해당 기관에 직접 연락해 주세요.",
                     "schedule_refusal")
    if draft["conditional"]:
        return handle_conditional(t, draft, item)
    pending = draft["pending"]
    draft["pending"] = {}
    if not any(u[k] for k in ("new_date_text", "new_time_text", "shift_text")):
        if isinstance(pending.get("start"), str) and ss.parse_iso(pending["start"]):
            # A기능 날씨 비교 제안을 여러 일정 중 하나에 연결하는 경우: 대상이 정해졌으므로 그 시각으로 제안한다.
            # (2차 검토 보완) 지역을 묻느라 보관한 시각도 같은 방법으로 이어 받아 날씨 조건을 다시 확인한다.
            return offer_update(t, draft, item, ss.parse_iso(pending["start"]) or t.now, reason=draft["reason"])
        t.u = u = dict(u, **{k: v for k, v in pending.items() if k in CHANGE_WORDS})
    # (3차 검토 보완) 지역 등을 되묻기 전에 말한 시간대('저녁')를 이어 쓴다. 새 시각·상대 시간을 말했으면 그 정정이 우선하고,
    # 날짜만 고쳤으면 날짜는 새 값, 시간대는 이어 쓴 값으로 계산한다.
    if pending.get("period") in od.PERIODS and not (u["new_time_text"] or u["shift_text"]):
        t.u = u = dict(u, new_time_text=pending["period"])
        t.rule("period_kept_after_follow_up_question")

    # 2. 새 시각
    start, end = sv.interval(item)
    minutes = int((end - start).total_seconds() // 60)
    new = resolve_new_time(t, draft, start, time_is_new=bool(target.get("time_is_new")))
    if "ask" in new:
        return ask(t, draft, new["awaiting"], new["ask"], reason=new.get("reason", ""))
    if "auto" in new:
        picked = choose_slot(t, draft, new["auto"], minutes=minutes, exclude_id=item["schedule_id"], prefer=start,
                             city=_forecast_city(item, draft), category=item["category"])
        if "ask" in picked:
            return ask(t, draft, picked["awaiting"], picked["ask"])
        if "none" in picked:
            return ask(t, draft, "time", picked["none"])
        period = new["auto"]["period"]
        notice = f"말씀하신 시간대({period}) 안에서 가장 가까운 빈 시간으로 골랐어요." if period else ""
        return offer_update(t, draft, item, picked["start"], checks=picked["checks"] or None, reason=picked["reason"],
                            notice=notice, verified=picked["verified"])
    if "start" in new:
        return offer_update(t, draft, item, new["start"], approximate=new["approximate"], inherited=new["inherited"])
    if wants_delegate_confirm(draft):
        # (5-6 D42 보완) 날씨 조건만 있고 시각·위임 표현이 없으면 위임을 확정하지 않고 골라 드려도 되는지 먼저 묻는다.
        t.rule("delegation_confirm_asked")
        return ask(t, draft, "delegate_confirm", f"'{item['title']}' 일정은 지금 {when_text(start, t.now)} 예정이에요. "
                                                 + delegate_question(draft))
    return ask(t, draft, "time", f"'{item['title']}' 일정은 지금 {when_text(start, t.now)} 예정이에요. 몇 시로 바꿀까요?")


def handle_conditional(t: Turn, draft: dict[str, Any], item: dict[str, Any]) -> None:
    """'비가 오면 바꿀까?': 실제 예보에서 그 조건(비)이 확인될 때만 변경을 제안한다.
    조회 실패·값 누락(unknown)을 '비가 안 온다'로 보지 않는다. 다른 위험 신호(바람 등)를 비 조건으로 보지 않는다."""
    preferences = draft["preferences"]
    if not preferences:
        return ask(t, draft, "condition", "어떤 날씨일 때 바꾸고 싶으신지 알려 주세요. (예: 비가 오면)")
    start, end = sv.interval(item)
    city = _forecast_city(item, draft)
    if item["category"] != "outing":
        draft["conditional"] = False
        return ask(t, draft, "time", f"'{item['title']}'은(는) 실외 외출 일정으로 저장되어 있지 않아 날씨 조건은 확인하지 않았어요. "
                                     "바꾸시려면 원하는 시각을 말씀해 주세요.")
    if not city:
        # (2차 검토 보완) 지역이 없는 실외 일정은 조건을 확인할 지역을 묻는다. 대답한 지역은 예보 조회에만 쓴다.
        return ask(t, draft, "city", f"'{item['title']}'의 예보를 확인할 지역이 저장되어 있지 않아요. 어느 지역인지 알려 주세요.")
    # (5-6 D21 보완) 일정 전체 구간 [시작, 종료)의 정시 예보를 모두 본다. 확인한 구간 중 하나라도 그 날씨(비)가 있으면 조건 충족,
    # 없는데 확인하지 못한 구간이 있으면 판단 불가(비가 안 온다고 보지 않음), 모두 확인했고 없으면 조건 불충족이다.
    check = fetch_span(t, city, start, end)
    topics = {PREFERENCE_TOPICS[p] for p in preferences}
    met = any(topics & set(part["signals"]) for part in check["parts"] if part["status"] != "unknown")
    words = "·".join(PREFERENCE_WORDS[p] for p in preferences)
    lines = weather_block(t, [("일정 시각", check)])
    when = when_text(start, t.now)
    if check["status"] == "unknown" and not met:
        t.rule("condition_unknown_not_treated_as_clear")
        return reply(t, "\n".join([f"'{item['title']}'({when}) 시간의 예보를 확인하지 못해 {words} 예보가 있는지 판단할 수 없어요. "
                                   f"일정은 그대로 두었어요. ({check['reason']})"] + lines), "schedule_condition_unknown")
    if not met:
        t.rule("condition_not_met")
        return reply(t, "\n".join([f"'{item['title']}'({when}) 시간에는 프로젝트 기준의 {words} 신호가 보이지 않아 "
                                   "그대로 두셔도 될 것 같아요. 일정은 바꾸지 않았어요."] + lines), "schedule_condition_not_met")
    # 조건이 맞으면, 그 날씨를 피할 수 있는 같은 날 빈 시간을 찾아 제안한다.
    t.rule("condition_met_by_forecast")
    draft.update(conditional=False, delegate=True)
    minutes = int((end - start).total_seconds() // 60)
    picked = choose_slot(t, draft, {"day": start.date(), "window": sv.AUTO_WINDOW, "period": None}, minutes=minutes,
                         exclude_id=item["schedule_id"], prefer=start, city=city, category=item["category"],
                         original_check=check)
    if "ask" in picked:
        return ask(t, draft, picked["awaiting"], picked["ask"])
    if "none" in picked:
        return ask(t, draft, "time", f"'{item['title']}' 시간({when})에 {words} 예보가 있어요.\n" + picked["none"])
    offer_update(t, draft, item, picked["start"], checks=picked["checks"] or None, reason=picked["reason"],
                 notice=f"'{item['title']}' 시간({when})에 {words} 예보가 있어요.", verified=picked["verified"])


def handle_create(t: Turn) -> None:
    """새 일정 등록: 이름 → 날짜 → 시각 순서로 한 번에 하나만 묻는다. 소요 시간을 모르면 기본 60분을 제안에 표시한다."""
    draft = t.draft if t.draft and t.draft["operation"] == "create" else new_draft("create")
    # (5-6 D13 보완) 진행 중인 등록과 이름이 다른 새 등록 요청이면(질문의 대답·정정이 아님) 앞 초안과 보관한 날짜·시각을 이어 쓰지 않는다.
    # (5-6 최종 보완) 모델이 낸 이름이 발화에 근거가 없어 버려졌더라도(진행 중인 이름과 다르면) 새 요청일 수 있으므로 이어 쓰지 않는다.
    # 이 경우 새 초안은 이름이 없어 이름부터 다시 묻는다(새 요청인지 확정할 수 없을 때 안전하게 질문).
    title = t.u["title"]
    dropped = [d.get("value") for d in t.result["llm"].get("discarded") or []
               if isinstance(d, dict) and d.get("field") == "title" and not d.get("kept")]     # 통째로 버려진 이름만
    # (5-6 D37 보완) 모델의 answers_previous_question만 믿지 않는다. 정정이거나, Python이 물어본 항목의 대답으로 확인되고
    # (모델의 대답 표시가 있거나 버려진 다른 이름이 없음), 근거 있는 다른 이름(새 요청 신호)이 없을 때만 이어 쓴다.
    # (실측에서 이 칸이 거의 늘 true여서 D13 보호가 풀렸음. 버려진 이름만 다르고 대답 표시도 없으면 기존 AG57b처럼 새 요청으로 본다)
    # (5-6 D49 보완) 확인 질문은 이름을 묻지 않으므로, 버려진 다른 이름이 있으면 대답 표시가 있어도 잇지 않는다.
    # 단, 실측에서 Qwen은 지역 대답('인천')에 '인천 방문' 같은 근거 없는 이름을 붙였다. 확인 질문 중 지역 표현을 담은 버려진 이름은
    # 새 요청의 근거로 보지 않는다(다른 경로의 기존 규칙은 그대로).
    confirming = draft["awaiting"] in CONFIRM_KINDS
    city = _squeeze(t.u["city_text"] or "") if confirming else ""
    foreign = [v for v in dropped if v != draft["title"] and not (city and city in _squeeze(str(v)))]
    other_dropped = bool(foreign)
    flag_ok = t.u["answers_previous_question"] and not confirming
    verified = draft["awaiting"] is not None and answer_fits(t, draft) and (flag_ok or not other_dropped)
    continuing = (t.u["request_mode"] == "correction" or verified) and not new_request_signal(t, draft)
    if draft["title"] and not continuing and ((title and title != draft["title"]) or foreign):
        draft = new_draft("create")
        t.rule("new_create_request_replaced_draft")
    # (5-6 D1·D13 보완) 지역·날짜·제목·소요 시간을 되묻느라 보관한 표현을 이어 쓴다. 이번 턴에 새로 말한 값이 우선한다.
    u = restore_held(t, draft, "create")
    # (5-6 D34·D42 보완) 날씨 조건·위임 확인 질문의 대답을 먼저 처리한다.
    answering = draft["awaiting"] in ("weather_condition", "delegate_confirm")
    if draft["awaiting"] == "weather_condition":
        outcome = settle_weather(t, draft, "create")
        if outcome == "cancel":
            return handle_negated(t, "create")
        if outcome == "stop":
            return
    elif draft["awaiting"] == "delegate_confirm":
        settle_delegate(t, draft)
    if answering:
        # (5-6 D53 보완) 확인 질문의 대답 턴에서는 모델의 위임 표기(delegate_choice)를 다시 쓰지 않는다. 위임은 위임 확인 단계의
        # 근거 있는 yes로만 확정하고, 날씨 조건 확인 대답의 위임 표기는 위임 확인 질문(D42)으로 넘긴다. 직접 말한 시각은 그대로 쓴다.
        t.u = dict(t.u, delegate_choice=False)
    u = t.u
    if u["title"]:
        draft["title"] = u["title"]
    if u["city_text"]:
        draft["city"] = u["city_text"]
    if u["category_guess"] in ss.CATEGORIES:
        draft["category"] = u["category_guess"]
    draft["delegate"] = draft["delegate"] or u["delegate_choice"]
    # (검토 보완) 등록에서도 사용자가 말한 날씨 선호('비 안 오는 시간에')를 초안에 남겨 시간 고르기에 쓴다.
    merge_conditions(draft, u)      # (5-6 D44 보완) 새 조건은 더한다(바꿔치기하지 않음)
    draft["check_weather"] = draft["check_weather"] or u["check_weather"]
    if ambiguous_city(t, draft, "create") or city_unconfirmed(t, draft, "create"):
        return
    if guard_conditions(t, draft, "create"):        # (5-6 D34 보완) 말한 날씨 조건을 모델이 빠뜨렸으면 카드 전에 묻는다
        return
    if ask_said_region(t, draft):                   # (5-6 F3 보완) 말한 지역 이름을 모델이 빠뜨렸으면 카드 전에 묻는다
        return
    if u["duration_text"]:
        minutes = sv.parse_minutes(u["duration_text"])
        if minutes is None:
            hold_said(t, draft, "create")      # (5-6 D13 보완) 소요 시간을 다시 묻는 동안 이미 말한 날짜·시각을 보관
            return ask(t, draft, "duration", "걸리는 시간은 10분에서 12시간 사이로 알려 주세요.")
        draft["minutes"] = minutes
    if not draft["title"]:
        hold_said(t, draft, "create")          # (5-6 D13 보완) 이름을 묻는 동안 이미 말한 날짜·시각·소요 시간을 보관
        return ask(t, draft, "title", "어떤 일정인지 이름을 알려 주세요.")
    # (2차 검토 보완) 날씨 조건을 확인할 지역을 묻느라 보관한 시각이 있으면, 새 시각을 말하지 않은 경우 그 시각으로 다시 확인한다.
    pending = draft["pending"]
    draft["pending"] = {}
    held = ss.parse_iso(pending.get("start"))
    if held is not None and not (u["date_text"] or u["time_text"]):
        return offer_create(t, draft, held)
    # (3차 검토 보완) 지역 등을 되묻기 전에 말한 시간대('저녁')를 이어 쓴다. 새 시각을 말했으면 그 정정이 우선하고,
    # 날짜만 고쳤으면 날짜는 새 값, 시간대는 이어 쓴 값으로 계산한다.
    if pending.get("period") in od.PERIODS and not u["time_text"]:
        t.u = u = dict(u, time_text=pending["period"])
        t.rule("period_kept_after_follow_up_question")
    new = resolve_new_time(t, draft, None)
    if "ask" in new:
        if new["awaiting"] == "date":
            hold_said(t, draft, "create")      # (5-6 D13 보완) 날짜를 묻는 동안 이미 말한 시각·소요 시간을 보관
        return ask(t, draft, new["awaiting"], new["ask"], reason=new.get("reason", ""))
    if "auto" in new:
        picked = choose_slot(t, draft, new["auto"], minutes=draft["minutes"] or sv.DEFAULT_MINUTES, exclude_id=None,
                             prefer=None, city=draft["city"], category=draft["category"])
        if "ask" in picked:
            return ask(t, draft, picked["awaiting"], picked["ask"])
        if "none" in picked:
            return ask(t, draft, "time", picked["none"])
        return offer_create(t, draft, picked["start"], checks=picked["checks"] or None, verified=picked["verified"])
    if "start" in new:
        return offer_create(t, draft, new["start"], approximate=new["approximate"])
    if not draft["date"]:
        return ask(t, draft, "date", f"'{draft['title']}' 일정은 어느 날인가요?")
    if wants_delegate_confirm(draft):
        # (5-6 D42 보완) 날씨 조건만 있고 시각·위임 표현이 없으면 위임을 확정하지 않고 골라 드려도 되는지 먼저 묻는다.
        t.rule("delegation_confirm_asked")
        return ask(t, draft, "delegate_confirm", f"'{draft['title']}' 일정의 시각을 아직 정하지 않으셨어요. " + delegate_question(draft))
    return ask(t, draft, "time", f"'{draft['title']}'은(는) 몇 시에 시작하나요?")


def handle_cancel_complete(t: Turn, operation: str) -> None:
    """취소(기록은 지우지 않고 cancelled) 또는 완료 기록(사용자가 끝냈다고 말한 일정만)"""
    draft = t.draft if t.draft and t.draft["operation"] == operation else new_draft(operation)
    # (5-6 D43 보완) '취소할까요?' 확인의 대답: 충돌 없는 긍정일 때만 취소 제안으로 진행한다(제안도 승인 버튼을 거쳐야 저장).
    if operation == "cancel" and draft["awaiting"] == "cancel_confirm" and not settle_cancel(t, draft):
        return
    target = _target_for(t, draft, operation)
    if "ask" in target:
        return ask(t, draft, target["awaiting"], target["ask"], candidates=target["candidates"])
    if "refuse" in target:
        return reply(t, target["refuse"], "schedule_refusal")
    item = target["item"]
    draft.update(target_id=item["schedule_id"], candidates=[])
    made = sv.propose_cancel(item, t.now, reason=t.u["reason_text"]) if operation == "cancel" else \
        sv.propose_complete(item, t.now, reason=t.u["reason_text"])
    if not made["ok"]:
        return reply(t, made["message"], "schedule_refusal")
    start = sv.interval(item)[0]
    if operation == "cancel":
        lines = [f"{when_text(start, t.now)} '{item['title']}' 일정을 취소하는 것을 제안해요. (기록은 지우지 않고 '취소됨'으로 남겨요)"]
    else:
        lines = [f"'{item['title']}'({when_text(start, t.now)}) 일정을 완료로 기록하는 것을 제안해요."]
    propose(t, draft, made["action"], lines + made["action"]["payload"]["notes"] + t.notices)


def handle_negated(t: Turn, operation: str) -> None:
    """'바꾸지 마', '원래 시간에 갈래': 제안을 만들지 않고 진행 중인 초안을 버린다."""
    draft = t.state["draft"]
    item = item_by_id(t.doc, draft["target_id"]) if draft and draft["target_id"] else None
    if item is None and (t.u["target_text"] or t.u["date_text"]):
        found = sv.find_targets(t.doc, now=t.now, operation="query", target_text=t.u["target_text"],
                                day=(sv.parse_day(t.u["date_text"], t.now) or {}).get("date"))
        item = found["matches"][0] if found["status"] == "single" else None
    t.rule("negated_request_no_proposal")
    if operation == "create":
        return reply(t, "알겠어요. 새 일정은 등록하지 않을게요.", "schedule_kept")
    if item is not None:
        return reply(t, f"알겠어요. '{item['title']}' 일정은 원래대로 {when_text(sv.interval(item)[0], t.now)}에 그대로 둘게요.",
                     "schedule_kept")
    return reply(t, "알겠어요. 일정은 바꾸지 않고 그대로 둘게요.", "schedule_kept")


# %% 조회 처리 함수
# 핵심 요약: 날짜별·대상·다음·남은·이번 주·이력·하루 정리를 저장소에서만 찾아 답한다. 조회는 승인이 필요 없고 파일을 바꾸지 않는다.
def _history_line(entry: dict[str, Any], now: datetime) -> str:
    before, after = entry["before"], entry["after"]
    snap = after or before or {}
    done = ss.parse_iso(entry["executed_at"])
    at = f"{sv.korean_day(done.date(), now)} {sv.korean_time(done)}" if done else "시각 미상"
    title = snap.get("title", "일정")
    start_of = (lambda s: ss.parse_iso(s["start_at"]) if isinstance(s, dict) else None)
    old, new = start_of(before), start_of(after)
    text = {"update": f"'{title}' {when_text(old, now) if old else ''} → {when_text(new, now) if new else ''}(으)로 변경",
            "create": f"'{title}' {when_text(new, now) if new else ''} 새로 등록",
            "cancel": f"'{title}' 취소", "complete": f"'{title}' 완료 기록"}.get(entry["operation"], entry["operation"])
    return f"• {at}: {text}"


def handle_query(t: Turn) -> None:
    u, doc, now = t.u, t.doc, t.now
    scope = u["query_scope"]
    day: date | None = None
    if u["date_text"]:
        parsed = sv.parse_day(u["date_text"], now)
        if parsed and "ambiguous" in parsed:
            return ask(t, t.draft or new_draft("query"), "date", parsed["ambiguous"])
        day = date.fromisoformat(parsed["date"]) if parsed else None
    ability = u["request_mode"] == "question" and u["operation"] in ("update", "cancel", "complete")
    # 이름 낱말이 있어야 '특정 일정' 조회로 본다. ('내일'만 target_text로 옮겨진 경우는 날짜 조회 — 실측 보완)
    named = bool(sv.name_words(u["target_text"])) or bool(sv.RECENT_RE.search(u["target_text"]))
    if ability or (scope in ("none", "date") and named and not day) or u["check_weather"]:
        scope = "target" if named or ability else scope
    if scope == "target" and not named and day:
        scope = "date"
    if scope == "none":
        scope = "date" if day else ("target" if named else "none")

    # 1. 특정 일정: 시각과 표시 상태. 날씨 확인을 요청하면 최신 예보를 다시 조회한다. (요구 14·17)
    if scope == "target":
        draft = t.draft if t.draft and t.draft["operation"] == "query" else new_draft("query")
        target = resolve_target(t, draft, "query")
        if "ask" in target:
            return ask(t, draft, target["awaiting"], target["ask"], candidates=target["candidates"])
        if "refuse" in target:
            return reply(t, target["refuse"])
        item = target["item"]
        start, end = sv.interval(item)
        t.result["schedules"] = [item["schedule_id"]]
        lines = [f"'{item['title']}' 일정은 {when_text(start, now)}~{sv.korean_time(end)}예요. "
                 f"(상태: {sv.STATE_WORDS[sv.display_state(item, now)]})"]
        follow = None
        if ability and u["operation"] == "update":
            lines.append("변경할 수 있는 일정이에요. 원하시는 시각을 말씀해 주시면 다른 일정과 겹치는지 확인해 드릴게요."
                         if item["reschedulable"] else "변경 제한 일정이라 SafeMate가 바꾸지 않아요.")
            follow = "update" if item["reschedulable"] and item["status"] == "scheduled" else None
        if u["check_weather"] and item["category"] == "outing" and item["status"] == "scheduled":
            check = fetch_forecast(t, item["city"], start)
            lines += weather_block(t, [("일정 시각", check)]) + guidance_lines(safety_guidance(t, check))
            if check["status"] == "signal" and item["reschedulable"]:
                lines.append("시간을 바꾸시려면 원하는 시각을 말씀하시거나 '알아서 바꿔 줘'라고 말씀해 주세요. 바꾸려면 새로 승인이 필요해요.")
                follow = "update"
        elif u["check_weather"]:
            lines.append("실외 외출 일정이 아니거나 이미 지난 일정이라 날씨는 확인하지 않았어요.")
        if follow:
            following = new_draft("update")
            following.update(target_id=item["schedule_id"], awaiting="time")
            t.state["draft"] = following
            return reply(t, "\n".join(lines), keep_draft=True)
        return reply(t, "\n".join(lines))

    # 2. 다음 일정·진행 중 (완료·취소 일정은 다음 일정으로 안내하지 않음)
    if scope == "next":
        running, upcoming = sv.running_schedules(doc, now), sv.next_schedule(doc, now)
        lines = (["지금 진행 중인 일정:", bullet(running, now)] if running else [])
        lines.append(f"다음 일정: {sv.describe(upcoming, now)}" if upcoming else "앞으로 예정된 일정이 없어요.")
        t.result["schedules"] = [s["schedule_id"] for s in running] + ([upcoming["schedule_id"]] if upcoming else [])
        return reply(t, "\n".join(lines))
    if scope == "remaining":
        items = sv.remaining_today(doc, now)
        t.result["schedules"] = [s["schedule_id"] for s in items]
        return reply(t, "오늘 남은 일정이에요.\n" + bullet(items, now, with_day=False) if items else "오늘 남은 일정이 없어요.")
    if scope == "week":
        first, last = sv.week_range(now)
        items = sv.schedules_between(doc, first, last)
        t.result["schedules"] = [s["schedule_id"] for s in items]
        return reply(t, "이번 주 일정이에요.\n" + bullet(items, now) if items else "이번 주에는 저장된 일정이 없어요.")

    # 3. 변경 이력: LLM 대화 기억이 아니라 실제 저장 기록(action_history)에서 찾는다. (요구 27·41)
    if scope == "history":
        target_id = None
        if u["target_text"]:
            found = sv.find_targets(doc, now=now, operation="query", target_text=u["target_text"])
            target_id = found["matches"][0]["schedule_id"] if found["status"] == "single" else None
        recent = sv.recent_reference(doc, u["target_text"] or t.message, now)
        if recent and recent.get("schedule_id") and target_id is None:
            target_id = recent["schedule_id"]
        entries = sv.history_entries(doc, schedule_id=target_id, day=None if target_id else (day or now.date()))
        if not entries:
            item = item_by_id(doc, target_id)
            if item is not None:
                return reply(t, f"'{item['title']}' 일정은 바꾼 기록이 없어요. 지금 일정: {sv.describe(item, now)}")
            return reply(t, "해당하는 변경 기록이 없어요." if target_id or day else "오늘 바꾸거나 등록한 기록이 없어요.")
        lines = ["저장된 변경 기록이에요."] + [_history_line(e, now) for e in entries[:LIST_LIMIT]]
        updates = [e for e in entries if e["operation"] == "update" and isinstance(e["before"], dict)]
        if target_id and updates:
            first_before = ss.parse_iso(updates[-1]["before"]["start_at"])
            if first_before:
                lines.append(f"처음에는 {when_text(first_before, now)} 일정이었어요.")
        return reply(t, "\n".join(lines))

    # 4. 하루 정리: 오늘 일정·오늘 한 변경·내일 일정. B기능 기록은 연결된 경우에만 읽는다. (요구 30)
    if scope == "summary":
        record = sv.life_record(doc, now, t.b_records)
        lines = ["오늘 일정:", bullet(record["today"], now, with_day=False) or "• 없음",
                 "오늘 바꾸거나 기록한 일:", "\n".join(_history_line(e, now) for e in record["changes_today"]) or "• 없음",
                 "내일 일정:", bullet(record["tomorrow"], now, with_day=False) or "• 없음"]
        if record["b_status"] == "not_connected":
            lines.append(B_NOT_CONNECTED)
        else:
            lines += ["오늘 보호자께 요청한 일:"] + [f"• {r['message']}" for r in record["b_requests"]] if record["b_requests"] \
                else ["오늘 보호자께 요청한 기록은 없어요."]
        return reply(t, "\n".join(lines))

    # 5. 날짜별 조회 (기본)
    if scope == "date" and day is not None:
        items = sv.schedules_on(doc, day)
        t.result["schedules"] = [s["schedule_id"] for s in items]
        label = sv.korean_day(day, now)
        return reply(t, f"{label} 일정이에요.\n" + bullet(items, now, with_day=False) if items
                     else f"{label}에는 저장된 일정이 없어요.")
    return ask(t, t.draft or new_draft("query"), "date", "어느 날 일정을 알려 드릴까요?")


# %% 대화 이어 받기와 작업 고르기
# 핵심 요약: 진행 중인 초안이 있으면 이번 말이 그 질문의 대답인지(모델 판단 + Python 확인) 보고, 새 요청이면 초안을 바꾼다.
# 그 뒤 request_mode와 operation으로 처리 함수를 고른다. (if문은 상태 분기에만 쓰고, 말의 뜻은 LLM이 판단함)
SHORT_ANSWER_LIMIT = 20     # '오후 네 시쯤', '첫 번째'처럼 질문에 대한 짧은 대답만 이어 붙인다 (프로젝트 설정)


def _short_answer_fits(t: Turn, draft: dict[str, Any]) -> bool:
    """모델이 other로 분류한 짧은 말이, Python이 물어본 항목(시각·날짜·후보 번호)에 대한 대답으로 읽히는지
    질문 문장('오늘 날씨 어때?')이나 긴 문장은 대답으로 보지 않는다. 시각을 물었으면 시각만, 날짜를 물었으면 날짜만 본다."""
    awaiting = draft["awaiting"]
    if "?" in t.message or len(t.message) > SHORT_ANSWER_LIMIT:
        return False
    if awaiting == "time":
        clock, moved = sv.parse_clock(t.message), sv.resolve_shift(t.message, t.now)
        return bool(clock and "time" in clock) or bool(moved and "at" in moved)
    if awaiting == "date":
        day = sv.parse_day(t.message, t.now)
        return bool(day and "date" in day)
    if awaiting in ("target", "confirm"):
        shown = [s for s in (item_by_id(t.doc, i) for i in draft["candidates"]) if s]
        return sv.choose_from_candidates(shown, t.message, t.now) is not None
    return False


UNCLEAR_SLOT_KEYS = ("date_text", "time_text", "new_date_text", "new_time_text", "shift_text", "duration_text")
UNCLEAR_LEAD = "앞에서 여쭌 질문에 대한 대답인지 확실하지 않아 날짜·시각은 바꾸지 않았어요. "


def conditional_unsupported(t: Turn, operation: str) -> None:
    """조건부 등록·취소·완료 미지원 안내 (일정은 바꾸지 않음)"""
    t.rule("conditional_operation_not_supported")
    word = sv.OPERATION_WORDS[operation]
    reply(t, f"날씨 조건에 따라 일정을 {word}하는 기능은 아직 없어서 일정은 바꾸지 않았어요. "
             f"조건 없이 {word}하시려면 다시 말씀해 주시고, 날씨가 걱정되시면 '비가 오면 시간을 바꿀까?'처럼 물어봐 주세요.",
          "schedule_condition_unsupported")


def _dispatch(t: Turn, mode: str) -> None:
    u = t.u
    draft = t.state["draft"]
    operation, request_mode = u["operation"], u["request_mode"]
    # 1. 진행 중인 일정 상담의 대답인지 확인한다.
    # (5-6 D36·D37 보완) answers_previous_question이나 correction 하나만으로 잇지 않는다. Python이 물어본 항목의 대답인지(answer_fits)와
    # 새 요청 신호(다른 이름·대상)를 함께 본다. 확인할 수 있는 항목을 물었는데 맞지 않으면 대답으로 보지 않는다.
    if draft and draft["awaiting"] == "city" and not u["city_text"]:
        # (5-6 F3 보완) 지역 질문의 대답인데 모델이 city_text를 비우면 사용자 원문에서 지역 표현을 꺼낸다(3단계 규칙과 같은 모양 검사).
        # 일정 이름이라 버린 값(F2)은 모델이 지역을 알아챈 근거로 보지 않는다.
        saw = any(isinstance(d, dict) and d.get("field") == "city_text" and d.get("reason") != "schedule_name"
                  for d in t.result["llm"].get("discarded") or [])
        recovered = recover_city_text(t.message, model_saw_city=saw)
        if recovered:
            t.u = u = dict(u, city_text=recovered, **{k: "" for k in TEXT_FIELDS if k != "city_text"})
            t.rule("city_answer_recovered_from_user_text")
    if draft:
        checkable = draft["awaiting"] in ("time", "date", "target", "confirm", "city", "title", "duration") + CONFIRM_KINDS
        confirming = draft["awaiting"] in CONFIRM_KINDS
        if confirming and operation != draft["operation"] and answer_fits(t, draft) and not new_request_signal(t, draft) \
                and any(u.get(k) for k in UNCLEAR_SLOT_KEYS):
            # (5-6 D49 보완) 확인 질문 중 모델이 다른 작업·other로 표기했는데 날짜·시각 칸이 있으면, 대답인지 새 요청인지 확실하지 않다.
            # 이전 초안에 섞거나 보관하지 않고 같은 확인 질문을 다시 묻는다. (지역만 있거나 내용 없는 대답은 아래 규칙대로 잇는다)
            t.rule("unclear_answer_not_merged")
            question = draft["question"].removeprefix(UNCLEAR_LEAD)
            return ask(t, draft, draft["awaiting"], UNCLEAR_LEAD + question)
        if operation == "other" and answer_fits(t, draft) and not (confirming and new_request_signal(t, draft)):
            operation = draft["operation"]
            t.rule("short_answer_linked_to_draft")
        elif operation == "other" and u["answers_previous_question"] and draft["awaiting"] and not checkable:
            operation = draft["operation"]
            t.rule("answer_linked_to_draft")
        elif operation not in ("other", draft["operation"]) and request_mode == "correction" and not new_request_signal(t, draft):
            operation = draft["operation"]
            t.rule("correction_linked_to_draft")
        elif operation not in ("other", draft["operation"]) and draft["awaiting"] and answer_fits(t, draft) \
                and not new_request_signal(t, draft):
            # 모델이 짧은 대답('인천')을 다른 작업으로 표기한 경우: 물어본 항목의 대답으로 잇고, 새 요청에만 의미 있는 표기는 쓰지 않는다.
            operation = draft["operation"]
            keep_mode = request_mode if request_mode in ("negated", "correction") else "request"
            t.u = u = dict(u, request_mode=keep_mode, delegate_choice=False, other_day_ok=False, multiple_requests=False,
                           target_ids=[], query_scope="none")
            request_mode = keep_mode
            t.rule("answer_linked_despite_operation_label")
        if operation == draft["operation"]:
            current = item_by_id(t.doc, draft["target_id"])
            if u["target_text"] and current is not None and not sv.text_matches(current, u["target_text"]) \
                    and not u["exclude_previous_target"]:
                t.rule("new_target_started_new_request")
            else:
                t.draft = draft
        elif operation != "other":
            t.rule("new_request_replaced_draft")
    u["operation"] = operation
    if u["multiple_requests"] and operation in CHANGE_OPERATIONS:
        t.notices.append("한 번에 하나씩 처리할게요. 나머지 요청은 이 일이 끝난 뒤 다시 말씀해 주세요.")

    awaiting_cancel = bool(t.draft) and t.draft["awaiting"] == "cancel_confirm"
    # 2. 작업 고르기. 재확인(recheck)으로 온 경우는 일정 변경 요청일 때만 일정 답변을 쓰고, 나머지는 3단계 답변을 그대로 쓴다.
    # (5-6 D34 보완) 날씨 조건·위임 확인 질문의 대답은 '아니'·조건부 표기라도 처리 함수에서 확인 단계로 판단한다(단순 '아니'로 요청을 버리지 않음).
    answering = bool(t.draft) and t.draft["awaiting"] in ("weather_condition", "delegate_confirm")
    if request_mode == "negated" and operation != "other" and not answering:
        if mode == "recheck":
            t.state["draft"] = None
            t.rule("recheck_negated_keeps_outing_answer")
            return hand_off(t)
        # (5-6 D43 보완) 진행 중인 취소 제안이 없는데 취소 + 부정이면 '안 갈래'(취소 요청)인지 모호하므로 취소 여부를 묻는다.
        if operation == "cancel" and not (draft and draft["operation"] == "cancel") and ask_cancel_confirm(t):
            return
        return handle_negated(t, operation)
    if operation == "other":
        return hand_off(t)
    # (검토 보완) 날씨 조건부 판단은 시각 변경(update)에만 있다. 조건부 등록·취소·완료를 조건 없이 제안하지 않는다.
    # '비 오면 취소해 줘'를 모델이 request로 분류해도, 취소·완료에 날씨 선호가 붙어 있으면 조건부로 본다.
    if not answering and operation in ("create", "cancel", "complete") and (
            request_mode == "conditional" or (operation != "create" and u["preferences"])):
        return conditional_unsupported(t, operation)
    if not answering and not awaiting_cancel and operation == "cancel" and request_mode != "question" \
            and said_conditions(t.message):
        # (5-6 D50 보완) '비 오면 취소해 줘'를 모델이 조건 없는 일반 요청으로 표기한 경우: 조건(비 오면)인지 이유(비 와서)인지
        # Python은 구분할 수 없으므로 카드 대신 날씨와 관계없이 취소할지 묻는다. 대상이 하나로 정해지지 않으면 미지원 안내(카드 없음).
        if ask_cancel_confirm(t, weather=True):
            return
        return conditional_unsupported(t, operation)
    if operation == "update" and mode != "recheck" and request_mode != "question" and not (draft and draft["awaiting"]):
        # (5-6 F1 보완) 진행 중인 질문이 없는데 모델의 변경(update)이 발화의 취소 요청·날씨가 붙은 완료 표현과 어긋나면 이동 경로
        # (위임 확인·조건부 변경)로 보내지 않는다. 취소는 확인 질문(날씨 낱말이 있으면 조건부 취소 미지원 안내 + 즉시 취소 확인), 완료는 미지원 안내.
        conflict = operation_conflict(t)
        if conflict == "cancel":
            t.rule("operation_conflict_cancel")
            weather = bool(said_conditions(t.message))
            if ask_cancel_confirm(t, weather=weather):
                return
            if weather:
                return conditional_unsupported(t, "cancel")
            t.u = dict(u, operation="cancel")
            return handle_cancel_complete(t, "cancel")
        if conflict == "complete":
            t.rule("operation_conflict_complete")
            return conditional_unsupported(t, "complete")
    if mode == "recheck" and (operation not in CHANGE_OPERATIONS or request_mode == "question"):
        t.rule("recheck_not_schedule_change")
        return hand_off(t)
    if operation == "query" or request_mode == "question":
        return handle_query(t)
    if operation == "update":
        return handle_update(t)
    if operation == "create":
        return handle_create(t)
    return handle_cancel_complete(t, operation)


# %% 4단계 그래프가 호출할 공개 함수
# 핵심 요약: 한 번의 발화를 '저장소 읽기 → 일정 이해 LLM → 대상·시각 계산 → (필요하면) 예보·RAG → 답변·질문·제안'으로 처리한다.
# 읽기 전용이다. 저장은 하지 않는다. 승인용 제안만 돌려주고, 승인 정보(approval)는 만들지 않는다.
def _read_message(request: Any) -> str:
    message = request.get("user_message") if isinstance(request, dict) else request
    if not isinstance(message, str) or not message.strip():
        raise od.DecisionError("invalid_input", "empty_message", "일정에 관해 한 문장으로 말씀해 주세요.")
    if len(message) > od.MESSAGE_LIMIT:
        raise od.DecisionError("invalid_input", "message_too_long", f"말씀이 너무 길어요. {od.MESSAGE_LIMIT}자 이내로 나눠 말씀해 주세요.")
    return message.strip()


def run_schedule_request(request: Any, *, store: ss.ScheduleStore, conversation_state: dict | None = None,
                         device: str | None = None, index: Any = None, now: datetime | None = None,
                         context: dict | None = None) -> dict[str, Any]:
    """일정 상담 한 턴. JSON으로 저장할 수 있는 딕셔너리를 돌려준다. (예외를 밖으로 던지지 않음)

    - store: 일정 저장소(ScheduleStore). 읽기만 한다.
    - conversation_state: 직전 턴의 S 상태. None이면 새 상담
    - device·index: 공식 안전수칙 RAG용 (A기능과 같은 장치·색인)
    - now: 기준 시각(테스트 고정용). 없으면 3단계 current_time()
    - context: {"mode": continue | handoff | recheck, "b_records": B 모의 요청 기록 목록 또는 None}
    route: schedule(답변·제안), needs_clarification(질문), handoff(일정이 아닌 말 → A기능), error"""
    result = empty_schedule_result()
    previous = load_schedule_state(conversation_state)
    result["conversation_state"] = previous
    ctx = context if isinstance(context, dict) else {}
    raw_mode = ctx.get("mode")
    mode: str = raw_mode if isinstance(raw_mode, str) and raw_mode in CONTEXT_MODES else "continue"
    raw_records = ctx.get("b_records")
    b_records = [r for r in raw_records if isinstance(r, dict)] if isinstance(raw_records, list) else None
    try:
        # 1. 입력·저장소·Ollama 설정 확인 (저장소를 읽지 못하면 일정을 추측하지 않고 멈춘다)
        message = _read_message(request)
        current = (now if isinstance(now, datetime) and now.tzinfo else od.current_time()).astimezone(KST)
        doc = store.load()
        settings = od.ollama_settings()
        t = Turn(message=message, u={}, state=copy.deepcopy(previous), doc=doc, now=current, result=result,
                 settings=settings, device=device, index=index, b_records=b_records)

        # 2. 일정 이해 LLM 1회(형식 오류면 1회 재요청) → 3. 작업별 처리
        t.u = understand_schedule(t)
        _dispatch(t, mode)

        # 4. 다음 턴에 넘길 상태: 직전 대화와 (남아 있으면) 초안. 일정 자체는 넣지 않는다.
        state = t.state
        state["turn"] += 1
        lines = [{"role": "user", "text": message[:200]}]
        if result["route"] != "handoff":
            lines.append({"role": "assistant", "text": result["answer"][:200]})
        state["history"] = (state["history"] + lines)[-HISTORY_LIMIT:]
        result["conversation_state"] = state
        decision = dict(result["decision"] or {})
        decision.update(operation=t.u["operation"], request_mode=t.u["request_mode"], context_mode=mode,
                        answer_source=t.answer_source, verified=None)
        result["decision"] = decision
        result["message"] = {"schedule": "일정 요청을 처리했습니다.", "needs_clarification": "추가 정보가 필요합니다.",
                             "handoff": "일정 관리가 아닌 요청이라 다른 기능으로 넘깁니다."}.get(result["route"], "")
    except od.DecisionError as exc:
        # Ollama 미실행·모델 형식 오류·입력 오류: 이전 상태를 그대로 돌려준다. (이번 턴 정보를 확정하지 못함)
        result.update(status=exc.status, detail_code=exc.code, message=str(exc), route="error",
                      answer=("지금은 일정 요청을 처리하지 못했어요. " + str(exc)) if exc.status != "invalid_input" else str(exc),
                      conversation_state=previous, decision={"kind": "error", "answer_source": None})
    except ss.ScheduleStoreError as error:
        # 저장소 없음·손상·권한 문제: 파일을 초기화하거나 덮어쓰지 않고 그대로 알린다.
        result.update(status="store_error", detail_code=error.code, message=str(error), route="error",
                      answer=f"일정 파일을 읽지 못해 일정을 확인하지 않았어요. {error}", conversation_state=previous,
                      decision={"kind": "error", "answer_source": None})
    return result


# %% 그래프 연결 도우미
# 핵심 요약: 4단계 그래프가 A기능의 외출 시간 변경 제안을 여러 일정 중 하나에 연결해야 할 때 쓸 S 초안과, 승인 카드 미리보기를 만든다.
def link_draft(state: Any, candidates: list[dict[str, Any]], proposed_start: datetime, reason: str,
               now: datetime) -> tuple[dict[str, Any], str]:
    """(S 상태, 질문). 사용자가 후보를 고르면 handle_update가 이 시각으로 변경 제안을 만든다."""
    cleaned = load_schedule_state(state)
    shown = candidates[:LIST_LIMIT]
    question = (f"날씨를 비교해 {when_text(proposed_start, now)}을(를) 제안드렸는데, 저장된 일정 중 어느 것을 바꿀지 확인이 필요해요. "
                "번호로 알려 주세요.\n" + numbered(shown, now))
    draft = new_draft("update")
    draft.update(awaiting="target", question=question, candidates=[s["schedule_id"] for s in shown],
                 reason=reason[:ss.REASON_LIMIT], pending={"start": ss.iso(proposed_start)})
    cleaned["draft"] = draft
    return cleaned, question


def approval_preview(action: Any) -> str:
    """승인 카드 미리보기: 일정 이름, 변경 전·후 날짜와 시각, 소요 시간, 이유, 충돌 여부, 확인한 예보와 공식 안내"""
    if not isinstance(action, dict):
        return ""
    try:
        payload = action["payload"]
        operation, before, after = payload["operation"], payload["before"], payload["after"]

        def span(snap: dict[str, Any]) -> tuple[str, int]:
            start, end = ss.parse_iso(snap["start_at"]), ss.parse_iso(snap["end_at"])
            if start is None or end is None:
                raise ValueError("시각")
            return (f"{start.month}월 {start.day}일 {sv.korean_time(start)}~{sv.korean_time(end)}",
                    int((end - start).total_seconds() // 60))

        if operation == "update":
            old, minutes = span(before)
            lines = [f"[변경] {after['title']}: {old} → {span(after)[0]}"]
        elif operation == "create":
            new, minutes = span(after)
            lines = [f"[새 일정] {after['title']}: {new}" + (f" · {after['city']}" if after.get("city") else "")]
        else:
            old, minutes = span(before)
            lines = [f"[{'취소' if operation == 'cancel' else '완료 기록'}] {before['title']}: {old}"]
        lines.append(f"소요 시간 {minutes}분")
        if operation in ("update", "create"):
            lines.append("다른 일정과 겹치지 않음 (저장 직전에 다시 확인)")
        if payload.get("reason"):
            lines.append(f"이유: {payload['reason']}")
        lines += [str(n) for n in payload.get("notes") or []] + [str(w) for w in payload.get("weather_lines") or []
                                                                 if not str(w).startswith("[확인한 예보]")]
        guidance = [g for g in payload.get("official_guidance") or [] if isinstance(g, dict) and g.get("text")]
        if guidance:
            lines.append(f"[공식 안전 안내] {guidance[0]['text']} ({guidance[0].get('publisher') or '출처 확인'})")
        return "\n".join(lines)
    except (KeyError, TypeError, ValueError, AttributeError):
        summary = action.get("summary")
        return summary if isinstance(summary, str) else ""
