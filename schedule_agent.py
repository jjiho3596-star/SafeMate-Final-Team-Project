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
            "check_weather": False, "pending": {}}


def _clean_draft(raw: Any) -> dict[str, Any] | None:
    """형식이 맞는 초안만 남긴다. 알 수 없는 키·잘못된 자료형은 기본값으로 둔다. (그래프 State에서 받은 값을 그대로 믿지 않음)"""
    if not isinstance(raw, dict) or raw.get("operation") not in CHANGE_OPERATIONS + ("query",):
        return None
    draft = new_draft(raw["operation"])
    for key, default in draft.items():
        value = raw.get(key, default)
        if key in ("candidates", "excluded_ids", "preferences"):
            draft[key] = [v for v in value if isinstance(v, str)] if isinstance(value, list) else []
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
        at = ss.parse_iso(check["at"]) or t.now
        if check["weather"]:
            slots.append(dict(check["weather"], fact_id=f"W{len(slots) + 1}"))
        state = {"signal": "프로젝트 기준 위험 신호 있음(" + ", ".join(check["signals"]) + ")",
                 "no_signal": "프로젝트 기준 위험 신호 없음(안전 보장 아님)"}.get(check["status"], "확인 못함 · " + check["reason"])
        notes.append(f"• {label} {sv.korean_time(at)}: {state}")
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
    topics = od.choose_topics({"slots": [dict(check["weather"], fact_id="W1")]}, "outing_consultation", [])
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
    t.result["llm"]["understanding"] = grounded
    t.result["llm"]["model_understanding"] = understanding
    t.result["llm"]["discarded"] = discarded + [{"field": "target_ids", "value": i} for i in dropped]
    return grounded


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
        return after["temperature_c"] > before["temperature_c"]
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
    original = original_check
    if outdoor and original is None and prefer is not None:
        original = fetch_forecast(t, city, prefer)
    checks = [fetch_forecast(t, city, s) for s in starts] if outdoor else \
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
    return [{"label": label, "at": c["at"], "status": c["status"], "reason": c["reason"], "fetched_at": c["fetched_at"]}
            for label, c in checks]


def preference_gate(t: Turn, draft: dict[str, Any], start: datetime, city: str | None, category: Any,
                    label: str) -> dict[str, Any]:
    """(2차 검토 보완) 사용자가 피하려는 날씨를 말했는데 시각을 직접·상대 시간·충돌 대안으로 정한 경우, 그 시각의 예보를 확인한다.
    {"checks": [(label, 예보)]}(조건 확인됨), {"ask", "awaiting"}(지역 질문) 또는 {"none": 안내}(확인 못함·조건에 맞지 않음)
    choose_slot()과 같은 기준(eligible)을 쓴다. 확인하지 못한 예보를 '비가 안 온다'로 보지 않는다."""
    preferences = draft["preferences"]
    words = "·".join(PREFERENCE_WORDS[p] for p in preferences)
    when = when_text(start, t.now)
    if category != "outing":
        return {"none": f"실외 외출 일정으로 저장되어 있지 않아 {words} 예보를 확인할 수 없어요. 일정은 바꾸지 않았어요. "
                        "날씨와 관계없이 그 시각으로 정하시려면 다시 말씀해 주세요."}
    if not city:
        draft["pending"] = {"start": ss.iso(start)}     # 지역을 받으면 같은 시각으로 다시 확인한다
        return {"ask": f"{when}의 {words} 예보를 확인하려면 지역이 필요해요. 어느 지역인지 알려 주세요.", "awaiting": "city"}
    check = fetch_forecast(t, city, start)
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
        notice = conflict_notice(made["message"], options, weather_pending=bool(draft["preferences"]) and not verified)
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
        gate = preference_gate(t, draft, start, city, item["category"], "바꿀 시각")
        if "ask" in gate:
            return ask(t, draft, gate["awaiting"], "\n".join(([notice] if notice else []) + [gate["ask"]]))
        if "none" in gate:
            return ask(t, draft, "time", "\n".join(([notice] if notice else []) + [gate["none"]]))
        checks = [("원래 시각", fetch_forecast(t, city, old_start))] + gate["checks"]
    # 3. 실외 일정: 원래 시각과 바꿀 시각의 예보 (이미 조회했으면 다시 조회하지 않음)
    if checks is None and weather_needed(u, item["category"], item["city"]):
        checks = [("원래 시각", fetch_forecast(t, item["city"], old_start)), ("바꿀 시각", fetch_forecast(t, item["city"], start))]
    extra_lines = _attach_weather(t, action, checks or [])
    if checks and city and city != item["city"]:
        extra_lines.append(f"(예보는 말씀하신 지역 '{city}' 기준이에요. 일정에 저장된 지역은 바꾸지 않아요.)")
    draft.update(target_id=item["schedule_id"], date=start.date().isoformat(), time=f"{start:%H:%M}", pending={})
    new_when = sv.korean_time(start) if start.date() == old_start.date() else when_text(start, t.now)
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
        notice = conflict_notice(made["message"], options, weather_pending=bool(draft["preferences"]) and not verified)
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
        gate = preference_gate(t, draft, start, city, category, "일정 시각")
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


def ambiguous_city(t: Turn, draft: dict[str, Any]) -> bool:
    """(검토 보완) '광주'처럼 시·도 없이는 정할 수 없는 지역은 1단계 규칙(AMBIGUOUS_CITY_NAMES)을 따라 확정하지 않고 묻는다.
    물었으면 True (등록·변경 공통)"""
    city = draft["city"]
    if not isinstance(city, str) or city.strip().lower() not in AMBIGUOUS_CITY_NAMES:
        return False
    options = AMBIGUOUS_CITY_NAMES[city.strip().lower()]
    draft["city"] = None
    ask(t, draft, "city", f"어느 지역인지 정할 수 없어요. {options} 중 어디인지 알려 주세요.")
    return True


def handle_update(t: Turn) -> None:
    """일정 시각 변경 (상대 시간·직접 지정·시간대·선택 위임·날씨 조건)"""
    u = t.u
    draft = t.draft if t.draft and t.draft["operation"] == "update" else new_draft("update")
    draft["preferences"] = u["preferences"] or draft["preferences"]
    draft["check_weather"] = draft["check_weather"] or u["check_weather"]
    draft["conditional"] = draft["conditional"] or u["request_mode"] == "conditional"
    draft["delegate"] = draft["delegate"] or u["delegate_choice"]
    # (2차 검토 보완) 지역 질문에 대한 대답('인천')은 초안에 남겨 예보 조회에만 쓴다. 일정에 저장된 지역은 바꾸지 않는다.
    if u["city_text"]:
        draft["city"] = u["city_text"]
    if ambiguous_city(t, draft):
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
    check = fetch_forecast(t, city, start)
    words = "·".join(PREFERENCE_WORDS[p] for p in preferences)
    lines = weather_block(t, [("일정 시각", check)])
    when = when_text(start, t.now)
    if check["status"] == "unknown":
        t.rule("condition_unknown_not_treated_as_clear")
        return reply(t, "\n".join([f"'{item['title']}'({when}) 시간의 예보를 확인하지 못해 {words} 예보가 있는지 판단할 수 없어요. "
                                   f"일정은 그대로 두었어요. ({check['reason']})"] + lines), "schedule_condition_unknown")
    if not {PREFERENCE_TOPICS[p] for p in preferences} & set(check["signals"]):
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
    u = t.u
    draft = t.draft if t.draft and t.draft["operation"] == "create" else new_draft("create")
    if u["title"]:
        draft["title"] = u["title"]
    if u["city_text"]:
        draft["city"] = u["city_text"]
    if u["category_guess"] in ss.CATEGORIES:
        draft["category"] = u["category_guess"]
    draft["delegate"] = draft["delegate"] or u["delegate_choice"]
    # (검토 보완) 등록에서도 사용자가 말한 날씨 선호('비 안 오는 시간에')를 초안에 남겨 시간 고르기에 쓴다.
    draft["preferences"] = u["preferences"] or draft["preferences"]
    draft["check_weather"] = draft["check_weather"] or u["check_weather"]
    if ambiguous_city(t, draft):
        return
    if u["duration_text"]:
        minutes = sv.parse_minutes(u["duration_text"])
        if minutes is None:
            return ask(t, draft, "duration", "걸리는 시간은 10분에서 12시간 사이로 알려 주세요.")
        draft["minutes"] = minutes
    if not draft["title"]:
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
    return ask(t, draft, "time", f"'{draft['title']}'은(는) 몇 시에 시작하나요?")


def handle_cancel_complete(t: Turn, operation: str) -> None:
    """취소(기록은 지우지 않고 cancelled) 또는 완료 기록(사용자가 끝냈다고 말한 일정만)"""
    draft = t.draft if t.draft and t.draft["operation"] == operation else new_draft(operation)
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


def _dispatch(t: Turn, mode: str) -> None:
    u = t.u
    draft = t.state["draft"]
    operation, request_mode = u["operation"], u["request_mode"]
    # 1. 진행 중인 일정 상담의 대답인지 확인한다.
    if draft:
        if operation == "other" and _short_answer_fits(t, draft):
            operation = draft["operation"]
            t.rule("short_answer_linked_to_draft")
        elif operation == "other" and u["answers_previous_question"] and draft["awaiting"]:
            operation = draft["operation"]
            t.rule("answer_linked_to_draft")
        elif operation not in ("other", draft["operation"]) and request_mode == "correction":
            operation = draft["operation"]
            t.rule("correction_linked_to_draft")
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

    # 2. 작업 고르기. 재확인(recheck)으로 온 경우는 일정 변경 요청일 때만 일정 답변을 쓰고, 나머지는 3단계 답변을 그대로 쓴다.
    if request_mode == "negated" and operation != "other":
        if mode == "recheck":
            t.state["draft"] = None
            t.rule("recheck_negated_keeps_outing_answer")
            return hand_off(t)
        return handle_negated(t, operation)
    if operation == "other":
        return hand_off(t)
    # (검토 보완) 날씨 조건부 판단은 시각 변경(update)에만 있다. 조건부 등록·취소·완료를 조건 없이 제안하지 않는다.
    # '비 오면 취소해 줘'를 모델이 request로 분류해도, 취소·완료에 날씨 선호가 붙어 있으면 조건부로 본다.
    if operation in ("create", "cancel", "complete") and (
            request_mode == "conditional" or (operation != "create" and u["preferences"])):
        t.rule("conditional_operation_not_supported")
        word = sv.OPERATION_WORDS[operation]
        return reply(t, f"날씨 조건에 따라 일정을 {word}하는 기능은 아직 없어서 일정은 바꾸지 않았어요. "
                        f"조건 없이 {word}하시려면 다시 말씀해 주시고, 날씨가 걱정되시면 '비가 오면 시간을 바꿀까?'처럼 물어봐 주세요.",
                     "schedule_condition_unsupported")
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
