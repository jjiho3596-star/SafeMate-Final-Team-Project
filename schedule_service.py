"""SafeMate 5단계: 일정 서비스 (LLM 없음, 정확성이 필요한 일정 계산 담당)

핵심 요약
- 날짜·시각·소요 시간·상대 시간 표현을 정확한 KST 날짜와 시각으로 바꾼다. (3단계 정규화 함수를 재사용하고 빠진 표현만 보강)
- 저장된 일정을 조회하고, 사용자가 가리킨 일정을 찾고, 시간 충돌과 빈 시간을 계산한다.
- 등록·변경·취소·완료 제안(승인 봉투)을 만들고, 승인된 제안을 저장 직전에 다시 검사해 schedule_store에 한 번만 저장한다.
- 3단계가 만든 기존 외출 시간 변경 제안을 저장된 일정과 안전하게 연결한다.

LLM은 사용자의 말을 이해하는 일(schedule_agent·3단계)을 맡고, 이 파일은 그 결과를 검증·계산하는 일만 맡는다.
이 파일의 함수는 사용자에게 없는 일정·시각을 만들어 내지 않는다.
"""

# %% 공통 라이브러리와 기존 단계 함수 불러오기
# 핵심 요약: 날짜·시각 해석은 3단계 함수를, 예보 위험 신호는 3단계 기준을, 저장은 5단계 저장소를 그대로 쓴다.
from __future__ import annotations

import copy
import json
import math
import re
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import schedule_store as ss
from a_outing import KST
from outing_decision import (CLOCK_RE, FIELDS, NATIVE_TEENS, PERIODS, WORD_NUMBERS, forecast_signals, normalize_minute_words,
                             parse_clock_text, parse_date_text, parse_lead_text, unreadable_number)


# %% 일정 서비스 설정값
# 핵심 요약: 자동 추천 범위·후보 수·기본 소요 시간처럼 일정 계산이 함께 쓰는 프로젝트 설정을 한곳에 모은다.
# 07~21시는 '자동으로 추천할 때만' 쓰는 범위다. 사용자가 직접 말한 시각은 이 범위로 막지 않는다.
AUTO_WINDOW = (7, 21)          # 자동 추천 시작·끝 시각 (프로젝트 설정, 어르신 활동 시간 가정)
SLOT_STEP_MINUTES = 30
MAX_CANDIDATES = 3
MAX_EXTRA_DAYS = 3             # 사용자가 다른 날도 괜찮다고 했을 때만 더 찾는 날 수
START_MARGIN_MINUTES = 10      # 자동 추천은 지금부터 10분 뒤 이후만
DEFAULT_MINUTES = 60           # 소요 시간을 모를 때 제안하는 기본값 (승인 카드에 표시)
MIN_MINUTES, MAX_MINUTES = 10, 720
EARLY_HOUR, LATE_HOUR = 6, 22  # 이 범위를 벗어난 직접 요청은 막지 않고 '이른·늦은 시각' 표시만 붙인다
MAX_AHEAD_DAYS = 366 * 3       # 상대 날짜·이동 상한 (이보다 먼 날은 계산하지 않고 다시 묻는다. 오버플로 방지)

SOURCE = "schedule_service"
ACTION_TYPES = {"create": "create_schedule", "update": "update_schedule",
                "cancel": "cancel_schedule", "complete": "complete_schedule"}
OPERATION_OF = {kind: op for op, kind in ACTION_TYPES.items()}
SNAPSHOT_KEYS = ("title", "start_at", "end_at", "city", "category", "status", "reschedulable")
REFUSAL_STATUSES = ("stale", "conflict", "protected", "not_found", "invalid_action")
OPERATION_WORDS = {"create": "등록", "update": "변경", "cancel": "취소", "complete": "완료 기록"}
STATE_WORDS = {"scheduled": "예정", "running": "진행 중", "overdue": "시간 지남(완료 기록 없음)",
               "completed": "완료", "cancelled": "취소됨"}


# %% 한글 숫자 표현 보강 함수
# 핵심 요약: 3단계 시각·기간 함수가 읽지 못하는 '십오 분', '열두 시간' 같은 한글 숫자만 아라비아 숫자로 바꾼다.
# (3단계 parse_lead_text는 '열두 시간'의 '두 시간'만 읽어 120분이 되므로, 넘기기 전에 12시간으로 바꾼다)
# (5-6 D24 보완) 분 앞 한자어 수 규칙(SINO)은 3단계 outing_decision.normalize_minute_words로 옮겨 출발 여유 시간과 함께 쓴다.
# (5-6 P1 보완) 13~19시간(열세~열아홉)도 통째로 읽는다. 13~19 사전과 '읽을 수 없는 숫자'(범위·붙은 수·너무 큰 수) 규칙은
# 3단계 outing_decision에 한 번만 두고 그대로 가져다 쓴다(중복 규칙 없음, outing_decision은 이 파일을 import하지 않아 순환 없음).
NATIVE_HOURS = {**WORD_NUMBERS, **NATIVE_TEENS}
NATIVE_HOUR_RE = re.compile(r"(?<![가-힣])(열다섯|열여섯|열일곱|열여덟|열아홉|열한|열두|열세|열네|다섯|여섯|일곱|여덟|아홉|[한두세네열])\s*시간")
PARTIAL_NUMBER_ASK = "말씀하신 숫자를 정확히 알 수 없어요. '13시간'처럼 숫자로 다시 알려 주세요."


def normalize_numbers(text: str) -> str:
    """분 앞의 한자어 수(3단계 공통 규칙)와 '시간' 앞의 고유어 수(한~열아홉)를 숫자로 바꾼 문자열"""
    text = normalize_minute_words(text or "")
    return NATIVE_HOUR_RE.sub(lambda m: f"{NATIVE_HOURS[m[1]]}시간", text)


def has_partial_number(text: str) -> bool:
    """(5-6 P1 보완) 원문 또는 정규화한 문장에 하나로 정할 수 없거나 뒷부분만 잘라 읽힐 숫자 표현이 있는지
    (범위 '두 세 시간'은 정규화하면 '두 3시간'이 되므로 원문도 본다. 있으면 계산하지 않고 다시 묻는다)"""
    return unreadable_number(text or "") or unreadable_number(normalize_numbers(text))


# %% 날짜 해석 함수
# 핵심 요약: 3단계 parse_date_text를 기본으로 쓰고, 일정 대화에 필요한 '다음 달 N일', '일주일 뒤', '그날·다음 날'만 보강한다.
APPROX_RE = re.compile(r"쯤|경(?![기상])|무렵|즈음|조금\s*(?:넘|지나)|넘어서|가량")


def _date_result(day: date, now: datetime) -> dict[str, str]:
    """3단계와 같은 이름표(예: 내일(11월 26일 목요일))를 붙인다."""
    parsed = parse_date_text(day.isoformat(), now) or {}
    return {"date": day.isoformat(), "label": parsed.get("label", day.isoformat())}


def parse_day(text: str, now: datetime, context_date: str | None = None) -> dict[str, str] | None:
    """{'date','label'}, {'ambiguous': 이유} 또는 None(날짜 표현 없음)

    context_date: 앞 대화에서 확정한 날짜. '그날', '다음 날'은 이 날짜를 기준으로 한다."""
    t = re.sub(r"\s+", "", text or "")
    today = now.astimezone(KST).date()
    base = date.fromisoformat(context_date) if context_date else None

    # 1. 앞 대화의 날짜를 가리키는 말: 기준 날짜가 없으면 추측하지 않고 다시 묻는다.
    if re.search(r"다음날|이튿날|그다음날", t):
        return _date_result(base + timedelta(days=1), now) if base else {"ambiguous": "어느 날의 다음 날인지 알려 주세요."}
    if re.search(r"그날|그때그날", t):
        return _date_result(base, now) if base else {"ambiguous": "어느 날인지 알려 주세요."}

    # 2. '다음 달·이번 달 N일': 3단계 함수는 '다음 달'을 무시하므로 여기서 먼저 처리한다.
    if m := re.search(r"(다음|담|이번)달(\d{1,2})일", t):
        year, month = today.year, today.month
        if m[1] != "이번":
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        try:
            found = date(year, month, int(m[2]))
        except ValueError:
            return {"ambiguous": "달력에 없는 날짜입니다."}
        if found < today:
            return {"ambiguous": "이미 지난 날짜입니다."}
        return _date_result(found, now)

    # 3. '일주일 뒤', '2주 후' (너무 먼 날짜는 계산하지 않고 다시 묻는다)
    # (5-6 P1 보완) '열두 주 뒤'·'한두 주 뒤'의 '두 주'만 잘라 읽지 않고 다시 묻는다.
    if has_partial_number(text or ""):
        return {"ambiguous": PARTIAL_NUMBER_ASK}
    if m := re.search(r"(일주일|한주|([0-9]+|두|세|네)주)(?:일)?(?:뒤|후)", t):
        weeks = 1 if m[1] in ("일주일", "한주") else (int(m[2]) if m[2].isdecimal() else WORD_NUMBERS[m[2]])
        if weeks > MAX_AHEAD_DAYS // 7:
            return {"ambiguous": "너무 먼 날짜예요. 날짜를 다시 알려 주세요."}
        return _date_result(today + timedelta(weeks=weeks), now)

    # 4. 서로 다른 날을 가리키는 말이 함께 있으면 임의로 고르지 않는다. ('오늘 내일', '내일 금요일'(내일이 목요일))
    words = {"모레" if w in ("내일모레", "모레") else w for w in re.findall(r"내일모레|오늘|내일|모레|글피", t)}
    if len(words) > 1:
        return {"ambiguous": "어느 날인지 하나로 알려 주세요."}
    if m := re.search(r"(?<!다음주)(?<!이번주)(?<!담주)([월화수목금토일])요일", t):
        if words:
            relative = parse_date_text(words.pop(), now) or {}
            weekday = date.fromisoformat(relative["date"]).weekday() if "date" in relative else None
            if weekday is not None and weekday != "월화수목금토일".index(m[1]):
                return {"ambiguous": "말씀하신 날짜와 요일이 서로 맞지 않아요. 날짜를 다시 알려 주세요."}

    # 5. 나머지(오늘~글피, 요일, M월 D일, N일 뒤, 주말 등)는 3단계 함수가 처리한다.
    # 3단계 함수는 'N일 뒤'의 N이 매우 크면 OverflowError를 내므로, 여기서 잡아 다시 묻는다.
    if (m := re.search(r"([0-9]+)일?(?:뒤|후)", t)) and int(m[1]) > MAX_AHEAD_DAYS:
        return {"ambiguous": "너무 먼 날짜예요. 날짜를 다시 알려 주세요."}
    try:
        return parse_date_text(text, now)
    except (OverflowError, ValueError):
        return {"ambiguous": "날짜를 계산할 수 없어요. 날짜를 다시 알려 주세요."}


# %% 시각 해석 함수
# 핵심 요약: 3단계 parse_clock_text로 시각을 읽고, '쯤·조금 넘어서' 같은 대략 표현을 표시한다.
# '네 시'처럼 오전·오후가 없으면 묻는다. 단, 직전에 정한 시각을 정정하는 말('일곱 시 아니고 여덟 시')이면
# 직전 시각의 오전·오후를 이어받는다. (이어받았다는 사실을 inherited로 남겨 안내에 표시)
def parse_clock(text: str, inherit_from: str | None = None) -> dict[str, Any] | None:
    """{'time','approximate','inherited'}, {'ambiguous': 이유} 또는 None"""
    normalized = normalize_numbers(text)
    # 시각이 둘 이상이면(예: '2시나 3시') 어느 것인지 임의로 고르지 않는다. (3단계 함수는 이때 None을 돌려줌)
    if len(CLOCK_RE.findall(normalized)) > 1:
        return {"ambiguous": "시각이 두 개 이상이에요. 하나로 알려 주세요."}
    parsed = parse_clock_text(normalized)
    if parsed is None:
        return None
    approximate = bool(APPROX_RE.search(normalized))
    if "time" in parsed:
        return {"time": parsed["time"], "approximate": approximate, "inherited": False}

    # 1. 오전·오후가 없어서 모호한 경우만, 직전 시각(HH:MM)의 오전·오후를 이어받는다.
    if inherit_from and "오전인지 오후인지" in parsed["ambiguous"]:
        half = "오후" if int(inherit_from[:2]) >= 12 else "오전"
        again = parse_clock_text(f"{half} {normalized}")
        if again and "time" in again:
            return {"time": again["time"], "approximate": approximate, "inherited": True}
    return {"ambiguous": parsed["ambiguous"]}


def parse_minutes(text: str) -> int | None:
    """'한 시간 반', '삼십 분', '열두 시간'을 분으로 바꾼다. 범위를 벗어나거나 읽을 수 없는 숫자(P1)면 None"""
    if has_partial_number(text):
        return None
    minutes = parse_lead_text(normalize_numbers(text))
    return minutes if minutes is not None and MIN_MINUTES <= minutes <= MAX_MINUTES else None


# %% 상대 시간 계산 함수
# 핵심 요약: '한 시간 미뤄', '30분 앞당겨', '다음 날 같은 시간', '일주일 뒤', '지금부터 9시간 뒤'를 기준 시각에서 계산한다.
# 기준(지금인지, 원래 일정인지)은 호출하는 쪽이 정해서 넘긴다. 계산은 LLM이 아니라 datetime이 한다.
EARLIER_RE = re.compile(r"앞당|당겨|당기|일찍|빨리|이르게|(?<![가-힣])전(?:에|으로|$)")
LATER_RE = re.compile(r"뒤|후|늦|미뤄|미루|미룰|연기|나중")


def resolve_shift(text: str, base: datetime) -> dict[str, Any] | None:
    """{'at': datetime, 'minutes': 이동한 분}, {'ambiguous': 이유} 또는 None(상대 시간 표현 없음)"""
    t = normalize_numbers(text or "")
    squeezed = re.sub(r"\s+", "", t)
    # (5-6 P1 보완) '스물두 시간'·'115시간'·'열두 주'처럼 뒷부분만 읽힐 숫자는 계산하지 않고 다시 묻는다.
    if has_partial_number(text or ""):
        return {"ambiguous": PARTIAL_NUMBER_ASK}

    # 1. 날 단위: '다음 날 같은 시간', '일주일 뒤', 'N일 뒤'
    days = 0
    if re.search(r"다음날|이튿날", squeezed):
        days = 1
    elif m := re.search(r"(일주일|한주|([0-9]+|두|세|네)주)", squeezed):
        days = 7 * (1 if m[1] in ("일주일", "한주") else (int(m[2]) if m[2].isdecimal() else WORD_NUMBERS[m[2]]))
    elif m := re.search(r"([0-9]+|하루|이틀|사흘)일?(?=뒤|후|늦|미뤄|미루|연기)", squeezed):
        days = int(m[1]) if m[1].isdecimal() else WORD_NUMBERS[m[1]]

    # 2. 시간·분 단위: 3단계 parse_lead_text 재사용 ('한 시간', '1시간 반', '30분')
    minutes = 0 if days else (parse_lead_text(t) or 0)
    if not days and not minutes:
        return None

    # 3. 방향: 앞당기기(−)인지 미루기(+)인지 말하지 않았으면 다시 묻는다. ('다음 날·일주일'은 앞으로 가는 표현)
    earlier, later = bool(EARLIER_RE.search(t)), bool(LATER_RE.search(t))
    if days and not earlier:
        later = True
    if earlier == later:
        return {"ambiguous": "앞당기시는 건지 미루시는 건지 알려 주세요."}

    # 4. 너무 먼 이동(예: 999999999주 뒤)은 계산하지 않고 다시 묻는다. (OverflowError 방지)
    if days > MAX_AHEAD_DAYS:
        return {"ambiguous": "너무 먼 시각이에요. 다시 알려 주세요."}
    delta = timedelta(days=days, minutes=minutes)
    total = int(delta.total_seconds() // 60)
    try:
        moment = base - delta if earlier else base + delta
    except OverflowError:
        return {"ambiguous": "시각을 계산할 수 없어요. 다시 알려 주세요."}
    return {"at": moment, "minutes": -total if earlier else total}


# %% 일정 시각·상태 함수
# 핵심 요약: 저장된 ISO 문자열을 시각 구간으로 바꾸고, 지금 시각 기준의 표시 상태를 계산한다.
# 시간이 지났다는 이유로 저장된 상태를 '완료'로 바꾸지 않는다. 표시만 '시간 지남(완료 기록 없음)'으로 한다.
def interval(item: dict[str, Any]) -> tuple[datetime, datetime]:
    """(시작, 종료) KST datetime"""
    start, end = ss.parse_iso(item["start_at"]), ss.parse_iso(item["end_at"])
    if start is None or end is None:
        raise ValueError("일정 시각 형식이 올바르지 않습니다.")
    return start, end


def display_state(item: dict[str, Any], now: datetime) -> str:
    """scheduled, running, overdue, completed, cancelled 중 하나"""
    if item["status"] != "scheduled":
        return item["status"]
    start, end = interval(item)
    if now < start:
        return "scheduled"
    return "running" if now < end else "overdue"


def snapshot(item: dict[str, Any]) -> dict[str, Any]:
    """제안·이력에 남길 일정 요약 (승인 전후 비교용)"""
    return {key: copy.deepcopy(item[key]) for key in SNAPSHOT_KEYS}


# %% 한국어 시각 표현 함수
# 핵심 요약: 내부의 24시간제 시각을 어르신이 읽기 쉬운 '오후 4시 30분' 같은 말로 바꾼다.
def korean_time(moment: datetime) -> str:
    """0시 → 밤 12시, 12시 → 낮 12시, 13~23시 → 오후 1~11시"""
    hour, minute = moment.hour, moment.minute
    if hour == 0:
        text = "밤 12시"
    elif hour < 12:
        text = f"오전 {hour}시"
    elif hour == 12:
        text = "낮 12시"
    else:
        text = f"오후 {hour - 12}시"
    return text + (f" {minute}분" if minute else "")


def korean_day(day: date, now: datetime) -> str:
    """'내일(11월 26일 목요일)' 같은 이름표. 3단계와 같은 형식"""
    return _date_result(day, now)["label"]


def describe(item: dict[str, Any], now: datetime, *, with_day: bool = True) -> str:
    """'내일(11월 26일 목요일) 오후 2시~오후 3시 시장 장보기 [예정]'"""
    start, end = interval(item)
    day = f"{korean_day(start.date(), now)} " if with_day else ""
    end_text = korean_time(end) if end.date() == start.date() else f"{korean_day(end.date(), now)} {korean_time(end)}"
    return f"{day}{korean_time(start)}~{end_text} {item['title']} [{STATE_WORDS[display_state(item, now)]}]"


# %% 일정 조회 함수
# 핵심 요약: 날짜·기간·다음 일정·진행 중·남은 일정·변경 이력을 저장된 기록에서만 찾는다. (없는 일정을 만들지 않음)
def _sorted(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda s: (s["start_at"], s["schedule_id"]))


def schedules_between(doc: dict[str, Any], first: date, last: date) -> list[dict[str, Any]]:
    """first~last(포함) 날짜와 겹치는 모든 상태의 일정. 자정을 넘는 일정도 겹치는 날에 들어간다."""
    lower = datetime(first.year, first.month, first.day, tzinfo=KST)
    upper = datetime(last.year, last.month, last.day, tzinfo=KST) + timedelta(days=1)
    return _sorted(s for s in doc["schedules"] if interval(s)[0] < upper and lower < interval(s)[1])


def schedules_on(doc: dict[str, Any], day: date) -> list[dict[str, Any]]:
    """그날과 겹치는 일정 (완료·취소 포함, 표시 상태로 구분)"""
    return schedules_between(doc, day, day)


def week_range(now: datetime) -> tuple[date, date]:
    """'이번 주': 오늘부터 이번 주 일요일까지"""
    today = now.astimezone(KST).date()
    return today, today + timedelta(days=6 - today.weekday())


def next_schedule(doc: dict[str, Any], now: datetime) -> dict[str, Any] | None:
    """지금 이후 시작하는 예정 일정 중 가장 이른 것 (완료·취소·진행 중은 제외)"""
    upcoming = [s for s in doc["schedules"] if s["status"] == "scheduled" and interval(s)[0] > now]
    return _sorted(upcoming)[0] if upcoming else None


def running_schedules(doc: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
    """지금 진행 중인 예정 일정"""
    return _sorted(s for s in doc["schedules"] if display_state(s, now) == "running")


def remaining_today(doc: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
    """오늘 아직 끝나지 않은 예정 일정 (진행 중 포함)"""
    return [s for s in schedules_on(doc, now.date()) if s["status"] == "scheduled" and interval(s)[1] > now]


def history_entries(doc: dict[str, Any], *, day: date | None = None, schedule_id: str | None = None,
                    operations: Iterable[str] = ss.OPERATIONS) -> list[dict[str, Any]]:
    """실제로 적용된 작업 기록. 최신 순 (같은 초에 실행된 기록은 나중에 추가된 것이 먼저)"""
    wanted = set(operations)
    found = [(order, h) for order, h in enumerate(doc["action_history"]) if h["operation"] in wanted
             and (schedule_id is None or h["schedule_id"] == schedule_id)
             and (day is None or (ss.parse_iso(h["executed_at"]) or datetime.min.replace(tzinfo=KST)).date() == day)]
    return [h for _, h in sorted(found, key=lambda pair: (pair[1]["executed_at"], pair[0]), reverse=True)]


# %% 일정 충돌 검사 함수
# 핵심 요약: [시작, 종료) 구간이 겹치는 예정 일정을 찾는다. 날짜가 붙은 시각으로 비교하므로 자정을 넘는 일정도 맞게 계산된다.
def find_conflicts(doc: dict[str, Any], start: datetime, end: datetime,
                   exclude_id: str | None = None) -> list[dict[str, Any]]:
    """겹치는 예정 일정 목록. 끝나는 시각과 다음 시작 시각이 같으면 겹치지 않는다."""
    return _sorted(s for s in doc["schedules"] if s["status"] == "scheduled" and s["schedule_id"] != exclude_id
                   and interval(s)[0] < end and start < interval(s)[1])


# %% 빈 시간 탐색 함수
# 핵심 요약: 자동 추천이 필요할 때만, 정한 범위 안에서 소요 시간을 유지하며 겹치지 않는 시작 시각을 최대 3개 찾는다.
# 원하는 시각에 가까운 순서이고, 같은 거리면 늦은 쪽을 먼저 둔다. (사용자가 미루려는 경우가 많기 때문)
def find_free_slots(doc: dict[str, Any], *, day: date, minutes: int, now: datetime, prefer: datetime | None = None,
                    window: tuple[int, int] = AUTO_WINDOW, exclude_id: str | None = None,
                    extra_days: int = 0, limit: int = MAX_CANDIDATES) -> list[datetime]:
    """후보 시작 시각 목록. 같은 날 후보를 먼저 두고, extra_days(사용자가 허락한 경우만, 최대 3일)만큼 다음 날도 본다."""
    earliest = now + timedelta(minutes=START_MARGIN_MINUTES)
    length = timedelta(minutes=minutes)
    found: list[datetime] = []
    for offset in range(min(max(extra_days, 0), MAX_EXTRA_DAYS) + 1):
        # 1. 그날의 탐색 범위 안에서 30분 간격으로 시작 시각을 만든다.
        current = day + timedelta(days=offset)
        start = datetime(current.year, current.month, current.day, window[0], tzinfo=KST)
        stop = datetime(current.year, current.month, current.day, tzinfo=KST) + timedelta(hours=window[1])
        options = []
        while start + length <= stop:
            # 2. 지난 시각·충돌하는 시각은 빼고, 소요 시간이 범위 안에 들어가는 시각만 남긴다.
            if start >= earliest and not find_conflicts(doc, start, start + length, exclude_id):
                options.append(start)
            start += timedelta(minutes=SLOT_STEP_MINUTES)

        # 3. 원하는 시각과 가까운 순서(같은 거리면 늦은 쪽 먼저)로 정렬한다.
        anchor = prefer if prefer is not None and prefer.date() == current else None
        if anchor is not None:
            options.sort(key=lambda moment: (abs((moment - anchor).total_seconds()), moment < anchor))
        found.extend(options)
        if len(found) >= limit:
            break
    return found[:limit]


# %% 예보 상태 구분 함수
# 핵심 요약: 후보 시각의 예보를 '위험 신호 있음(signal)·위험 신호 없음(no_signal)·확인 못함(unknown)'으로 나눈다.
# no_signal은 프로젝트 기준의 위험 신호가 '보이지 않았다'는 뜻일 뿐, 안전을 보장한다는 뜻이 아니다.
# 1단계 예보 계약(5개 항목, 없는 값은 None, missing_fields 일치)과 3단계 read_weather()의 값 검사 규칙을 그대로 적용하고,
# 하나라도 어긋나면 unknown으로 둔다. (불완전한 예보를 '좋은 날씨'의 근거로 쓰지 않기 위함)
NONNEGATIVE_FIELDS = ("wind_speed_mps", "precipitation_mm", "precipitation_probability_pct")
TEMPERATURE_RANGE = (-90.0, 60.0)    # 지구 관측 범위를 벗어난 기온은 형식 오류로 본다 (프로젝트 검사 범위)


def forecast_status(reply: Any) -> dict[str, Any]:
    """reply: 3단계 read_weather()가 돌려준 딕셔너리 또는 None. {"status", "signals", "reason"}"""
    def unknown(reason: str) -> dict[str, Any]:
        return {"status": "unknown", "signals": {}, "reason": reason}

    # 1. 조회 성공 상태와 예보 딕셔너리가 있어야 한다.
    if not isinstance(reply, dict) or reply.get("status") not in ("ok", "partial"):
        return unknown("예보를 조회하지 못했습니다.")
    weather = reply.get("weather")
    if not isinstance(weather, dict):
        return unknown("예보 자료가 없습니다.")

    # 2. 1단계 계약의 5개 항목이 모두 있고, 값은 유한한 숫자이며 범위 안이어야 한다. (bool은 숫자로 보지 않음)
    missing = []
    for key in FIELDS:
        if key not in weather:
            return unknown(f"예보 항목 {key}가 없습니다.")
        value = weather[key]
        if value is None:
            missing.append(key)
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            return unknown(f"예보 항목 {key}의 값 형식이 올바르지 않습니다.")
        if key in NONNEGATIVE_FIELDS and value < 0 or key == "precipitation_probability_pct" and value > 100 \
                or key.startswith(("temperature", "apparent")) and not TEMPERATURE_RANGE[0] <= value <= TEMPERATURE_RANGE[1]:
            return unknown(f"예보 항목 {key}의 값이 범위를 벗어났습니다.")

    # 3. 빠진 항목 목록이 실제 None 항목과 같아야 하고, 하나라도 빠졌으면 판단하지 않는다.
    listed = weather.get("missing_fields")
    if not isinstance(listed, list) or sorted(str(x) for x in listed) != sorted(missing):
        return unknown("예보의 빠진 항목 목록이 값과 맞지 않습니다.")
    if missing or reply.get("status") != "ok":
        return unknown("확인되지 않은 예보 항목이 있습니다: " + ", ".join(missing))

    # 4. 3단계와 같은 기준으로 위험 신호를 계산한다.
    signals = forecast_signals(weather)
    return {"status": "signal" if signals else "no_signal", "signals": signals,
            "reason": "프로젝트 기준의 위험 신호가 있습니다." if signals else "프로젝트 기준의 위험 신호가 보이지 않았습니다(안전 보장 아님)."}


# %% 대상 일정 식별 함수
# 핵심 요약: 사용자의 말(제목·별칭), 날짜·시각·시간대, '아까 바꾼 일정' 같은 표현으로 저장된 일정을 찾는다.
# LLM이 고른 일정 id는 '후보를 좁히는 데만' 쓴다. 사용자 발화에 근거가 없으면 그 일정으로 확정하지 않고 확인을 묻는다.
STOP_WORDS = {"일정", "약속", "시간", "거", "것", "그거", "이거", "오늘", "내일", "모레", "글피", "오전", "오후", "아침",
              "점심", "저녁", "밤", "낮", "새벽", "다른", "그", "내", "제", "원래", "아까", "방금", "가는", "하는", "가기로",
              "하기로", "했던", "한", "할", "갈"}
WORD_ENDINGS = re.compile(r"(이랑|하고|에서|으로|에게|까지|부터|은|는|이|가|을|를|에|로|도|만|요|거|것)$")
RECENT_RE = re.compile(r"아까|방금|조금\s*전|최근")
RECENT_OPERATION = (("바꾼|바꿨|변경|옮긴|미룬|당긴", "update"), ("등록|추가|잡은|만든|넣은", "create"),
                    ("취소|없앤|뺀", "cancel"), ("완료|끝낸|다녀온", "complete"))


def _squeeze(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def name_words(text: str) -> list[str]:
    """일정 이름 비교에 쓸 낱말: 조사·어미를 떼고, 두 글자 이상이며 '일정·내일' 같은 일반 낱말이 아닌 것"""
    words = []
    for word in re.split(r"[\s,]+", text or ""):
        stem = word
        for _ in range(2):
            stem = WORD_ENDINGS.sub("", stem)
        if len(stem) >= 2 and stem not in STOP_WORDS:
            words.append(stem)
    return words


def text_matches(item: dict[str, Any], text: str) -> bool:
    """사용자 표현의 낱말이 일정 제목·별칭과 맞는지 ('장보는 거' ↔ '장보기', '병원 약속' ↔ '병원')"""
    names = [_squeeze(n) for n in [item["title"], *item["aliases"]]]
    return any(word in name or name in word for word in name_words(text) for name in names)


def covers_clock(item: dict[str, Any], clock: str, day: str | None = None) -> bool:
    """그 시각에 시작하거나 그 시각이 일정 구간 안에 있는지 (시각만 말한 '오후 2시 일정')

    day를 알면 그 날짜의 시각으로 비교한다. 모르면 일정이 걸친 날짜(자정을 넘으면 이틀)의 같은 시각을 모두 본다.
    예: 23:00~다음 날 01:00 일정은 다음 날 00:30을 포함한다."""
    start, end = interval(item)
    hour, minute = (int(x) for x in clock.split(":"))
    days = [date.fromisoformat(day)] if day else sorted({start.date(), end.date()})
    return any(start <= datetime(d.year, d.month, d.day, hour, minute, tzinfo=KST) < end for d in days)


def recent_reference(doc: dict[str, Any], text: str, now: datetime) -> dict[str, Any] | None:
    """'아까 바꾼 일정'이면 오늘 기록 중 해당 작업의 가장 최근 일정을 돌려준다. 해당 표현이 아니면 None"""
    if not RECENT_RE.search(text or ""):
        return None
    operations = [op for pattern, op in RECENT_OPERATION if re.search(pattern, text)] or list(ss.OPERATIONS)
    entries = history_entries(doc, day=now.date(), operations=operations)
    if not entries:
        return {"schedule_id": None}
    return {"schedule_id": entries[0]["schedule_id"], "operation": entries[0]["operation"]}


def find_targets(doc: dict[str, Any], *, now: datetime, operation: str, target_text: str = "",
                 day: str | None = None, clock: str | None = None, period: str | None = None,
                 llm_ids: Iterable[str] = (), exclude_ids: Iterable[str] = ()) -> dict[str, Any]:
    """{"status": single | multiple | none | unspecified | confirm, "matches": [...], "basis": [...]}

    - single: 사용자 발화 근거로 하나가 정해짐
    - multiple: 여러 개 → 번호를 붙여 다시 묻기
    - none: 조건에 맞는 일정이 없음 (없는 일정을 바꿨다고 처리하지 않음)
    - unspecified: 어떤 일정인지 말하지 않음
    - confirm: 발화 근거는 없고 LLM만 골랐음 → '~ 말씀이신가요?'로 확인"""
    # 1. 작업에 맞는 상태의 일정만 후보로 둔다. 변경·취소·완료는 예정 일정만, 조회는 모든 상태.
    excluded = set(exclude_ids)
    pool = [s for s in doc["schedules"] if s["schedule_id"] not in excluded
            and (operation == "query" or s["status"] == "scheduled")]
    basis: list[str] = []

    # 2. '아까 바꾼 일정': 실제 변경 이력에서 찾는다.
    recent = recent_reference(doc, target_text, now)
    if recent is not None:
        found = [s for s in pool if s["schedule_id"] == recent["schedule_id"]]
        return {"status": "single" if found else "none", "matches": found, "basis": ["recent_history"],
                "llm_suggested": []}

    # 3. 날짜·시각·시간대로 거른다.
    if day:
        pool = [s for s in pool if s in schedules_on(doc, date.fromisoformat(day))]
        basis.append("date")
    if clock:
        pool = [s for s in pool if covers_clock(s, clock, day)]
        basis.append("time")
    if period in PERIODS:
        start_hour, end_hour = PERIODS[period]
        pool = [s for s in pool if start_hour <= interval(s)[0].hour < end_hour]
        basis.append("period")

    # 4. 제목·별칭으로 거른다. 사용자가 이름을 말했는데 맞는 일정이 없으면 '없음'이다.
    if name_words(target_text):
        pool = [s for s in pool if text_matches(s, target_text)]
        basis.append("name")

    # 5. LLM이 고른 id는 대상을 확정하지 않는다. 사용자 발화로 하나가 정해질 때만 single이다.
    # 발화로 여러 개가 남았는데 LLM이 그중 하나를 골랐다면, 확정하지 않고 '추천'으로만 표시해 후보 목록과 함께 묻게 한다.
    # (예: '내일 오후 일정 바꿔 줘' → 시장·복지관 2개 → LLM이 복지관을 골라도 multiple + llm_suggested)
    wanted = {i for i in llm_ids if isinstance(i, str)}
    if basis:
        status = {0: "none"}.get(len(pool), "single" if len(pool) == 1 else "multiple")
        suggested = [s["schedule_id"] for s in _sorted(pool) if s["schedule_id"] in wanted] if status == "multiple" else []
        return {"status": status, "matches": _sorted(pool), "basis": basis, "llm_suggested": suggested}
    guessed = [s for s in pool if s["schedule_id"] in wanted]
    if guessed:
        return {"status": "confirm", "matches": _sorted(guessed), "basis": ["llm_only"], "llm_suggested": []}
    return {"status": "unspecified", "matches": [], "basis": [], "llm_suggested": []}


NEGATION_RE = re.compile(r"말고|아니|빼고|대신|싫")


def choose_from_candidates(candidates: list[dict[str, Any]], answer: str, now: datetime) -> dict[str, Any] | None:
    """번호를 붙여 보여 준 후보 중에서 '첫 번째', '2번', '내일 거', '시장'으로 고른 일정. 정해지지 않으면 None(다시 묻기)"""
    # 1. 부정·정정 표현('첫 번째 말고 두 번째', '아니 두 번째', '첫 번째 아니야')은 마지막 부정 표현 뒤의 말만 선택으로 본다.
    # 부정 뒤에 고른 말이 없으면(예: '첫 번째 말고', '첫 번째 아니야') 확정하지 않고 다시 묻는다.
    text = answer if isinstance(answer, str) else ""
    negations = list(NEGATION_RE.finditer(text))
    if negations:
        text = text[negations[-1].end():]

    # 2. 순서 표현 (첫·둘·두·세·셋째, 1번·2번). 순서가 두 개 이상이면 하나로 정할 수 없으므로 다시 묻는다.
    order = {"첫": 0, "1": 0, "두": 1, "둘": 1, "2": 1, "세": 2, "셋": 2, "3": 2}
    orders = list(re.finditer(r"(첫|두|둘|세|셋)\s*(?:번째|째)|([1-3])\s*번", text))
    if len(orders) > 1:
        return None
    if orders:
        index = order[orders[0][1] or orders[0][2]]
        return candidates[index] if index < len(candidates) else None

    # 3. 날짜나 이름으로 하나만 남으면 그것을 고른다.
    parsed = parse_day(text, now)
    if parsed and "date" in parsed:
        by_day = [c for c in candidates if interval(c)[0].date().isoformat() == parsed["date"]]
        if len(by_day) == 1:
            return by_day[0]
    by_name = [c for c in candidates if text_matches(c, text)]
    return by_name[0] if len(by_name) == 1 else None


# %% 완료 기록 조건 함수
# 핵심 요약: 사용자가 끝냈다고 말한 일정의 상태와 날짜로 완료 기록을 제안할 수 있는지 정한다.
# 예정 시각이 지났다는 이유만으로 완료로 바꾸지 않는다. (이 함수는 사용자가 완료를 말했을 때만 부른다)
def completion_check(item: dict[str, Any], now: datetime) -> dict[str, Any]:
    """{"ok": bool, "code", "message", "note"}"""
    start, _ = interval(item)
    if item["status"] == "completed":
        return {"ok": False, "code": "already_completed", "message": "이미 완료로 기록된 일정이에요.", "note": None}
    if item["status"] == "cancelled":
        return {"ok": False, "code": "cancelled", "message": "취소된 일정이라 완료로 기록할 수 없어요.", "note": None}
    if start.date() > now.date():
        return {"ok": False, "code": "future_day", "message": "아직 날짜가 오지 않은 일정이에요.", "note": None}
    note = f"예정 시각({korean_time(start)}) 전이지만 완료로 기록합니다." if now < start else None
    return {"ok": True, "code": "ok", "message": "", "note": note}


# %% 승인 제안 생성 함수
# 핵심 요약: 등록·변경·취소·완료를 공통 승인 봉투(명세 2.4)로 만든다. action_id는 여기서 한 번만 만든다.
# 제안 단계에서 이미 대상·버전·충돌·변경 제한을 확인하고, 그 시점의 일정(before)과 버전을 함께 담아 저장 직전에 다시 비교한다.
def _refuse(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "code": code, "message": message, **extra}


def _envelope(operation: str, now: datetime, summary: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"type": ACTION_TYPES[operation], "action_id": uuid.uuid4().hex, "summary": summary,
            "requires_approval": True, "applied": False, "created_at": ss.iso(now), "source": SOURCE,
            "payload": {"operation": operation, **payload}}


def _time_notes(start: datetime, end: datetime, approximate: bool = False, inherited: bool = False) -> list[str]:
    """승인 카드에 붙일 안내: 이른·늦은 시각, 대략 표현, 오전·오후 이어받음"""
    notes = []
    if start.hour < EARLY_HOUR or end.hour >= LATE_HOUR or end.date() != start.date():
        notes.append("이른 시각이나 늦은 시각이에요. 맞는지 확인해 주세요.")
    if approximate:
        notes.append(f"말씀하신 대략적인 시각을 {korean_time(start)}로 잡았어요.")
    if inherited:
        notes.append(f"앞에서 말씀하신 시간대를 이어 {korean_time(start)}로 계산했어요.")
    return notes


def propose_update(doc: dict[str, Any], item: dict[str, Any], new_start: datetime, now: datetime, *,
                   reason: str = "", approximate: bool = False, inherited: bool = False,
                   extras: dict[str, Any] | None = None) -> dict[str, Any]:
    """{"ok": True, "action"} 또는 {"ok": False, "code", "message", "conflicts", "alternatives"}"""
    # 1. 대상 상태와 변경 제한 확인
    start, end = interval(item)
    if item["status"] != "scheduled":
        return _refuse("not_scheduled", f"{STATE_WORDS[item['status']]} 일정은 바꿀 수 없어요.")
    if not item["reschedulable"]:
        return _refuse("protected", f"'{item['title']}'은(는) 변경 제한 일정이라 SafeMate가 바꾸지 않아요. "
                                    "필요하면 해당 기관에 직접 연락해 주세요.")

    # 2. 원래 소요 시간을 유지한 새 구간을 만들고, 지난 시각·같은 시각을 확인한다.
    new_end = new_start + (end - start)
    if new_start == start:
        return _refuse("same_time", f"이미 {korean_time(start)}로 되어 있어요.")
    if new_start < now:
        return _refuse("past_time", "이미 지난 시각으로는 바꿀 수 없어요.")

    # 3. 충돌이 있으면 제안하지 않고, 같은 날 빈 시간 후보를 함께 돌려준다. (보호된 일정을 대신 옮기지 않음)
    conflicts = find_conflicts(doc, new_start, new_end, exclude_id=item["schedule_id"])
    if conflicts:
        minutes = int((end - start).total_seconds() // 60)
        alternatives = find_free_slots(doc, day=new_start.date(), minutes=minutes, now=now, prefer=new_start,
                                       exclude_id=item["schedule_id"])
        names = ", ".join(f"{korean_time(interval(c)[0])} {c['title']}" for c in conflicts)
        return _refuse("conflict", f"{korean_time(new_start)}에는 다른 일정({names})이 있어요.",
                       conflicts=conflicts, alternatives=alternatives)

    # 4. 봉투를 만든다. 승인 카드에 보일 변경 전후·이유·안내를 담는다.
    after = dict(snapshot(item), start_at=ss.iso(new_start), end_at=ss.iso(new_end))
    summary = (f"{item['title']}: {korean_day(start.date(), now)} {korean_time(start)} → "
               f"{korean_day(new_start.date(), now)} {korean_time(new_start)}로 바꿉니다.")
    payload = {"schedule_id": item["schedule_id"], "expected_version": item["version"], "before": snapshot(item),
               "after": after, "reason": reason[:ss.REASON_LIMIT], "conflicts": [],
               "notes": _time_notes(new_start, new_end, approximate, inherited), "checked_at": ss.iso(now),
               "weather_lines": [], "official_guidance": [], **(extras or {})}
    return {"ok": True, "action": _envelope("update", now, summary, payload)}


def propose_create(doc: dict[str, Any], *, title: str, start: datetime, minutes: int | None, now: datetime,
                   city: str | None = None, category: str = "other", reason: str = "",
                   approximate: bool = False, extras: dict[str, Any] | None = None) -> dict[str, Any]:
    """새 일정 등록 제안. 소요 시간을 모르면 기본 60분을 쓰고 승인 카드에 표시한다."""
    # 1. 제목·소요 시간·분류·시각 확인
    title = re.sub(r"\s+", " ", title or "").strip()
    if not 1 <= len(title) <= ss.TITLE_LIMIT:
        return _refuse("title_needed", "어떤 일정인지 이름을 알려 주세요.")
    notes = []
    if minutes is None:
        minutes = DEFAULT_MINUTES
        notes.append(f"소요 시간을 말씀하지 않아 {DEFAULT_MINUTES}분으로 잡았어요.")
    if not MIN_MINUTES <= minutes <= MAX_MINUTES:
        return _refuse("bad_duration", "소요 시간은 10분에서 12시간 사이로 알려 주세요.")
    if category not in ss.CATEGORIES:
        category = "other"
    if start < now:
        return _refuse("past_time", "이미 지난 시각에는 등록할 수 없어요.")
    end = start + timedelta(minutes=minutes)

    # 2. 충돌하면 빈 시간 후보와 함께 거절한다.
    conflicts = find_conflicts(doc, start, end)
    if conflicts:
        alternatives = find_free_slots(doc, day=start.date(), minutes=minutes, now=now, prefer=start)
        names = ", ".join(f"{korean_time(interval(c)[0])} {c['title']}" for c in conflicts)
        return _refuse("conflict", f"{korean_time(start)}에는 다른 일정({names})이 있어요.",
                       conflicts=conflicts, alternatives=alternatives)

    # 3. 봉투
    after = {"title": title, "start_at": ss.iso(start), "end_at": ss.iso(end), "city": city, "category": category,
             "status": "scheduled", "reschedulable": True}
    summary = f"{korean_day(start.date(), now)} {korean_time(start)}에 '{title}'을(를) 새 일정으로 등록합니다."
    payload = {"schedule_id": None, "expected_version": None, "before": None, "after": after,
               "reason": reason[:ss.REASON_LIMIT], "conflicts": [],
               "notes": notes + _time_notes(start, end, approximate), "checked_at": ss.iso(now),
               "weather_lines": [], "official_guidance": [], **(extras or {})}
    return {"ok": True, "action": _envelope("create", now, summary, payload)}


def propose_cancel(item: dict[str, Any], now: datetime, *, reason: str = "") -> dict[str, Any]:
    """취소 제안. 기록을 지우지 않고 상태만 cancelled로 바꾸는 작업이다."""
    start, end = interval(item)
    if item["status"] != "scheduled":
        return _refuse("not_scheduled", f"{STATE_WORDS[item['status']]} 일정은 취소할 수 없어요.")
    if not item["reschedulable"]:
        return _refuse("protected", f"'{item['title']}'은(는) 변경 제한 일정이라 SafeMate가 취소하지 않아요. "
                                    "필요하면 해당 기관에 직접 연락해 주세요.")
    if end <= now:
        return _refuse("past_time", "이미 지난 일정이에요. 다녀오셨다면 완료로 기록할 수 있어요.")
    summary = f"{korean_day(start.date(), now)} {korean_time(start)} '{item['title']}'을(를) 취소합니다."
    payload = {"schedule_id": item["schedule_id"], "expected_version": item["version"], "before": snapshot(item),
               "after": dict(snapshot(item), status="cancelled"), "reason": reason[:ss.REASON_LIMIT],
               "conflicts": [], "notes": [], "checked_at": ss.iso(now), "weather_lines": [], "official_guidance": []}
    return {"ok": True, "action": _envelope("cancel", now, summary, payload)}


def propose_complete(item: dict[str, Any], now: datetime, *, reason: str = "") -> dict[str, Any]:
    """완료 기록 제안. completion_check()를 통과한 경우만 만든다."""
    check = completion_check(item, now)
    if not check["ok"]:
        return _refuse(check["code"], check["message"])
    summary = f"'{item['title']}'({korean_time(interval(item)[0])})을(를) 완료로 기록합니다."
    payload = {"schedule_id": item["schedule_id"], "expected_version": item["version"], "before": snapshot(item),
               "after": dict(snapshot(item), status="completed"), "reason": reason[:ss.REASON_LIMIT],
               "conflicts": [], "notes": [check["note"]] if check["note"] else [], "checked_at": ss.iso(now),
               "weather_lines": [], "official_guidance": []}
    return {"ok": True, "action": _envelope("complete", now, summary, payload)}


# %% 기존 외출 시간 변경 제안 연결 함수
# 핵심 요약: 3단계가 날씨 비교로 만든 propose_outing_time_change를, 승인 카드를 만들기 전에 저장된 일정과 연결한다.
# 시작 시각 하나만으로 연결하지 않는다. 날짜·시각 구간·지역·목적을 함께 확인한다.
# - 정확히 1개 → 변경 제안(대상 id·버전·변경 전 내용 포함), 여러 개 → 되묻기, 0개 → 기존 일정은 고치지 않고 신규 등록 제안
def _city_core(text: str | None) -> str:
    first = (text or "").split(",")[0]
    return re.sub(r"(특별자치시|특별자치도|특별시|광역시|시|군|구)$", "", _squeeze(first))


def link_outing_proposal(doc: dict[str, Any], a_result: dict[str, Any], now: datetime) -> dict[str, Any]:
    """{"kind": update | ambiguous | create | invalid, "action", "candidates", "message"}"""
    # 1. 3단계 제안과 외출 요청을 읽는다. 형식이 틀리면 연결하지 않는다.
    proposal = a_result.get("proposed_action") if isinstance(a_result, dict) else None
    request = a_result.get("outing_request") if isinstance(a_result, dict) else None
    request = request if isinstance(request, dict) else {}
    original = ss.parse_iso(proposal.get("original_outing_at")) if isinstance(proposal, dict) else None
    proposed = ss.parse_iso(proposal.get("proposed_outing_at")) if isinstance(proposal, dict) else None
    if not isinstance(proposal, dict) or original is None or proposed is None:
        return {"kind": "invalid", "action": None, "candidates": [], "message": "외출 시간 변경 제안의 형식을 확인하지 못했어요."}
    raw_reason, raw_city, raw_purpose = proposal.get("reason"), request.get("city"), request.get("purpose")
    reason: str = raw_reason if isinstance(raw_reason, str) else ""
    city: str | None = raw_city if isinstance(raw_city, str) and raw_city.strip() else None
    purpose: str = raw_purpose if isinstance(raw_purpose, str) else ""

    # 2. 같은 날, 원래 시각이 일정 구간 안에 있는 예정 일정
    timed = [s for s in doc["schedules"] if s["status"] == "scheduled"
             and interval(s)[0] <= original < interval(s)[1]]
    if city:
        timed = [s for s in timed if not s["city"] or _city_core(s["city"]) == _city_core(city)]

    # 3. 목적이 일정 이름과 맞거나(목적을 말한 경우), 목적 없이 지역까지 맞을 때만 확정한다.
    confirmed = [s for s in timed if (purpose and text_matches(s, purpose))
                 or (not purpose and city and s["city"] and _city_core(s["city"]) == _city_core(city))]
    if len(confirmed) == 1:
        linked = propose_update(doc, confirmed[0], proposed, now, reason=reason)
        if linked["ok"]:
            return {"kind": "update", "action": linked["action"], "candidates": confirmed, "message": ""}
        return {"kind": "refused", "action": None, "candidates": confirmed, "message": linked["message"],
                "refusal": linked}
    if confirmed or timed:
        return {"kind": "ambiguous", "action": None, "candidates": _sorted(confirmed or timed),
                "message": "어느 일정을 바꾸실지 확인이 필요해요."}

    # 4. 저장된 일정이 없으면 기존 일정을 고치지 않고, 바꾼 시각으로 새 일정 등록을 제안한다.
    title = purpose.strip() or "외출"
    created = propose_create(doc, title=title, start=proposed, minutes=None, now=now, city=city,
                             category="outing", reason=reason)
    if created["ok"]:
        return {"kind": "create", "action": created["action"], "candidates": [],
                "message": "저장된 일정에 이 외출이 없어 새 일정으로 등록을 제안해요."}
    return {"kind": "refused", "action": None, "candidates": [], "message": created["message"], "refusal": created}


# %% 승인 후 실행 함수
# 핵심 요약: 4단계 그래프가 사용자 승인을 확인한 봉투만 받아, 저장소 잠금 안에서 최신 일정을 다시 읽고 검사한 뒤 한 번만 저장한다.
# 반환 형식은 B의 execute_living_action과 같다: {status, message, action_id, applied, mock, record, answer}
def _result(status: str, message: str, action_id: Any, *, applied: bool = False,
            record: dict[str, Any] | None = None, answer: str = "") -> dict[str, Any]:
    return {"status": status, "message": message, "action_id": action_id if isinstance(action_id, str) else None,
            "applied": applied, "mock": True, "record": record, "answer": answer or message}


def _check_envelope(action: Any) -> str | None:
    """봉투 형식 문제를 돌려준다. None이면 형식이 맞다."""
    if not isinstance(action, dict):
        return "봉투가 딕셔너리가 아님"
    kind, payload = action.get("type"), action.get("payload")
    if not isinstance(kind, str) or kind not in OPERATION_OF or action.get("source") != SOURCE:
        return "작업 종류·출처가 일정 작업이 아님"
    if action.get("requires_approval") is not True or action.get("applied") is not False:
        return "승인 전 상태가 아님"
    if not isinstance(action.get("action_id"), str) or not ss.HEX32_RE.fullmatch(action["action_id"]):
        return "action_id 형식"
    if not isinstance(payload, dict) or payload.get("operation") != OPERATION_OF[kind]:
        return "payload 작업 종류"
    if not isinstance(payload.get("after"), dict):
        return "변경 후 내용 없음"
    if kind != "create_schedule" and (not isinstance(payload.get("schedule_id"), str)
                                      or type(payload.get("expected_version")) is not int
                                      or not isinstance(payload.get("before"), dict)):
        return "대상 일정 id·버전·변경 전 내용 없음"
    return None


def _same_snapshot(actual: dict[str, Any], approved: Any) -> bool:
    """실제 일정 요약과 승인받은 요약이 같은지. 시각은 같은 순간이면 같다고 본다. (승인 내용과 저장 결과 대조)"""
    if not isinstance(approved, dict) or set(approved) != set(SNAPSHOT_KEYS):
        return False
    for key in SNAPSHOT_KEYS:
        if key in ("start_at", "end_at"):
            if ss.parse_iso(approved[key]) is None or ss.parse_iso(approved[key]) != ss.parse_iso(actual[key]):
                return False
        elif type(approved[key]) is not type(actual[key]) or approved[key] != actual[key]:
            return False
    return True


def _mutation(action: dict[str, Any], clock: Callable[[], datetime]):
    """저장소 transact()에 넘길 변경 함수. 최신 문서에서 제안 시점과 같은지 다시 확인한다.

    현재 시각은 잠금을 잡은 뒤 이 함수 안에서 읽는다. (승인을 기다리는 동안 지난 시각·끝난 일정을 정확히 거르기 위함)
    작업별로 바꿀 수 있는 항목만 바뀌었는지 확인하고, 바꾼 결과가 승인받은 내용(after)과 같을 때만 저장을 허락한다."""
    payload = action["payload"]
    operation = payload["operation"]
    approved_at = action["approval"]["at"]

    def refuse(code: str, message: str) -> dict[str, Any]:
        return {"ok": False, "code": code, "message": message}

    def mutate(doc: dict[str, Any]) -> dict[str, Any]:
        now = clock()
        after = payload["after"]
        reason = payload.get("reason") if isinstance(payload.get("reason"), str) else ""
        if set(after) != set(SNAPSHOT_KEYS):
            return refuse("invalid_action", "승인받은 일정 내용의 형식이 올바르지 않아 저장하지 않았어요.")

        # 1. 등록: 이름·분류·지역·소요 시간(10분~12시간)·지난 시각·충돌을 다시 확인한다.
        if operation == "create":
            start, end = ss.parse_iso(after["start_at"]), ss.parse_iso(after["end_at"])
            title, city = after["title"], after["city"]
            if start is None or end is None or not isinstance(title, str) or title != title.strip() \
                    or not 1 <= len(title) <= ss.TITLE_LIMIT:
                return refuse("invalid_action", "등록할 일정의 시각·이름을 확인하지 못했어요.")
            if not MIN_MINUTES * 60 <= (end - start).total_seconds() <= MAX_MINUTES * 60:
                return refuse("invalid_action", "등록할 일정의 소요 시간이 허용 범위(10분~12시간)를 벗어나 저장하지 않았어요.")
            if after["status"] != "scheduled" or after["reschedulable"] is not True or after["category"] not in ss.CATEGORIES \
                    or not (city is None or isinstance(city, str) and 1 <= len(city) <= ss.CITY_LIMIT):
                return refuse("invalid_action", "등록할 일정의 상태·분류·지역이 올바르지 않아 저장하지 않았어요.")
            if start < now:
                return refuse("invalid_action", "승인하시는 사이에 등록 시각이 지나 저장하지 않았어요.")
            if find_conflicts(doc, start, end):
                return refuse("conflict", "승인 대기 중에 같은 시간에 다른 일정이 생겨 저장하지 않았어요.")
            item = {"schedule_id": ss.next_schedule_id(doc), "title": title,
                    "aliases": [w for w in title.split() if len(w) >= 2][:ss.ALIAS_LIMIT],
                    "start_at": ss.iso(start), "end_at": ss.iso(end), "city": city, "category": after["category"],
                    "status": "scheduled", "reschedulable": True, "version": 1, "source": "user",
                    "created_at": ss.iso(now), "updated_at": ss.iso(now), "completed_at": None, "cancelled_at": None}
            if not _same_snapshot(snapshot(item), after):
                return refuse("invalid_action", "저장할 내용이 승인받은 내용과 달라 저장하지 않았어요.")
            doc["schedules"].append(item)
            return {"ok": True, "item": item, "now": now,
                    "history": {"operation": "create", "schedule_id": item["schedule_id"], "before": None,
                                "after": snapshot(item), "reason": reason, "approved_at": approved_at}}

        # 2. 변경·취소·완료: 대상 존재, 버전, 변경 전 내용, 상태를 제안 시점과 비교한다.
        item = next((s for s in doc["schedules"] if s["schedule_id"] == payload["schedule_id"]), None)
        if item is None:
            return refuse("not_found", "승인 대기 중에 일정이 사라져 저장하지 않았어요.")
        if item["version"] != payload["expected_version"] or not _same_snapshot(snapshot(item), payload["before"]) \
                or item["status"] != "scheduled":
            return refuse("stale", "승인 대기 중에 일정이 바뀌어 저장하지 않았어요. 다시 확인해 주세요.")
        if operation in ("update", "cancel") and not item["reschedulable"]:
            return refuse("protected", "변경 제한 일정이라 저장하지 않았어요.")
        before = snapshot(item)
        old_start, old_end = interval(item)

        # 3. 작업별로 바꿀 수 있는 항목만 바뀌었는지 확인한다.
        # 변경: 시작·종료만, 원래 소요 시간 유지 / 취소: 상태만 cancelled / 완료: 상태만 completed
        unchanged = {key: before[key] for key in SNAPSHOT_KEYS if key not in ("start_at", "end_at", "status")}
        if any(type(after[key]) is not type(value) or after[key] != value for key, value in unchanged.items()):
            return refuse("invalid_action", "승인받은 내용에 바꿀 수 없는 항목의 변경이 있어 저장하지 않았어요.")
        if operation == "update":
            start, end = ss.parse_iso(after["start_at"]), ss.parse_iso(after["end_at"])
            if start is None or end is None or after["status"] != "scheduled":
                return refuse("invalid_action", "바꿀 시각을 확인하지 못했어요.")
            if end - start != old_end - old_start:
                return refuse("invalid_action", "원래 소요 시간과 다른 변경이라 저장하지 않았어요.")
            if start < now:
                return refuse("invalid_action", "승인하시는 사이에 바꿀 시각이 지나 저장하지 않았어요.")
            if find_conflicts(doc, start, end, exclude_id=item["schedule_id"]):
                return refuse("conflict", "승인 대기 중에 그 시간에 다른 일정이 생겨 저장하지 않았어요.")
            item.update(start_at=ss.iso(start), end_at=ss.iso(end))
        elif operation == "cancel":
            if after["status"] != "cancelled" or ss.parse_iso(after["start_at"]) != old_start \
                    or ss.parse_iso(after["end_at"]) != old_end:
                return refuse("invalid_action", "취소 요청 내용이 올바르지 않아 저장하지 않았어요.")
            if old_end <= now:
                return refuse("invalid_action", "승인하시는 사이에 일정이 이미 끝나 취소하지 않았어요. 다녀오셨다면 완료로 기록할 수 있어요.")
            item.update(status="cancelled", cancelled_at=ss.iso(now))
        else:
            if after["status"] != "completed" or ss.parse_iso(after["start_at"]) != old_start \
                    or ss.parse_iso(after["end_at"]) != old_end:
                return refuse("invalid_action", "완료 기록 요청 내용이 올바르지 않아 저장하지 않았어요.")
            check = completion_check(item, now)
            if not check["ok"]:
                return refuse("invalid_action", check["message"])
            item.update(status="completed", completed_at=ss.iso(now))
        item.update(version=item["version"] + 1, updated_at=ss.iso(now))

        # 4. 실제로 바뀐 결과가 승인받은 내용과 같아야 저장한다. (다르면 transact가 사본을 버려 파일은 그대로)
        if not _same_snapshot(snapshot(item), after):
            return refuse("invalid_action", "저장할 내용이 승인받은 내용과 달라 저장하지 않았어요.")
        return {"ok": True, "item": item, "now": now,
                "history": {"operation": operation, "schedule_id": item["schedule_id"], "before": before,
                            "after": snapshot(item), "reason": reason, "approved_at": approved_at}}
    return mutate


def _done_answer(operation: str, item: dict[str, Any], now: datetime) -> str:
    start = interval(item)[0]
    when = f"{korean_day(start.date(), now)} {korean_time(start)}"
    return {"create": f"'{item['title']}' 일정을 {when}에 등록했어요.",
            "update": f"'{item['title']}' 일정을 {when}로 바꿔 저장했어요.",
            "cancel": f"{when} '{item['title']}' 일정을 취소로 기록했어요.",
            "complete": f"'{item['title']}' 일정을 완료로 기록했어요."}[operation]


def apply_schedule_action(approved_action: Any, store: ss.ScheduleStore) -> dict[str, Any]:
    """승인된 일정 봉투를 저장한다. 예외를 밖으로 던지지 않고 status로 알린다.

    status: ok, already_executed, rejected(승인 없음), invalid_action, stale, conflict, protected, not_found, error"""
    action_id = approved_action.get("action_id") if isinstance(approved_action, dict) else None

    # 1. 봉투 형식과 승인 확인 (승인이 없으면 저장하지 않음)
    problem = _check_envelope(approved_action)
    if problem:
        return _result("invalid_action", f"승인 봉투 형식 오류: {problem}", action_id,
                       answer="요청 내용을 확인하지 못해 저장하지 않았어요.")
    assert isinstance(approved_action, dict)
    approval = approved_action.get("approval")
    if not isinstance(approval, dict) or approval.get("status") != "approved" or ss.parse_iso(approval.get("at")) is None:
        return _result("rejected", "사용자 승인이 확인되지 않았습니다.", action_id,
                       answer="승인이 확인되지 않아 저장하지 않았어요.")

    # 2. 저장소 잠금 안에서 다시 검사하고 한 번만 저장한다. 현재 시각은 잠금을 잡은 뒤 변경 함수 안에서 읽는다.
    try:
        outcome = store.transact(approved_action["action_id"], ss.content_hash(approved_action),
                                 _mutation(approved_action, store.clock))
    except ss.ScheduleStoreError as error:
        return _result("error", f"{error.code}: {error}", action_id, answer=f"일정 파일 문제로 저장하지 않았어요. {error}")

    # 3. 결과를 실행 함수 계약 형식으로 바꾼다. 실제로 파일이 바뀐 경우만 applied=True
    operation = approved_action["payload"]["operation"]
    if outcome["status"] == "applied":
        item, now = outcome["outcome"]["item"], outcome["outcome"]["now"]
        return _result("ok", "저장했습니다.", action_id, applied=True, record=outcome["record"],
                       answer=_done_answer(operation, item, now))
    if outcome["status"] == "already_executed":
        return _result("already_executed", "이미 적용한 작업입니다.", action_id, applied=True, record=outcome["record"],
                       answer="이미 처리한 요청이에요. 다시 저장하지 않았어요.")
    code = outcome["code"] if outcome["code"] in REFUSAL_STATUSES else "invalid_action"
    return _result(code, outcome["message"], action_id, answer=outcome["message"])


def apply_outing_change(approved_action: Any, store: ss.ScheduleStore) -> dict[str, Any]:
    """명세 3.7 호환 이름. 대상 일정 id·버전·변경 전 내용이 확인된 제안만 일정 변경으로 저장한다.
    3단계의 옛 봉투(propose_outing_time_change, 대상 미확인)는 저장하지 않는다."""
    action_id = approved_action.get("action_id") if isinstance(approved_action, dict) else None
    if not isinstance(approved_action, dict):
        return _result("invalid_action", "봉투가 딕셔너리가 아님", action_id, answer="요청 내용을 확인하지 못해 저장하지 않았어요.")
    if approved_action.get("type") == "propose_outing_time_change":
        payload = approved_action.get("payload")
        if not isinstance(payload, dict) or not all(k in payload for k in ("schedule_id", "expected_version", "before", "after")):
            return _result("invalid_action", "대상 일정을 확인하지 않은 외출 시간 변경 제안입니다.", action_id,
                           answer="어느 일정을 바꿀지 확인되지 않아 저장하지 않았어요.")
        approved_action = dict(approved_action, type="update_schedule", source=SOURCE,
                               payload=dict(payload, operation="update"))
    return apply_schedule_action(approved_action, store)


# %% 생활 기록 요약 함수
# 핵심 요약: 오늘 한 일(완료·취소·변경 기록)과 내일 일정을 저장된 기록에서 모은다.
# B기능(보호자 도움 요청) 기록은 B가 연결된 경우에만 넘겨받는다. 연결되지 않았으면 '확인할 수 없음'으로 둔다.
def life_record(doc: dict[str, Any], now: datetime, b_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """{"today", "tomorrow", "changes_today", "b_status", "b_requests"}"""
    today = now.date()
    return {"today": schedules_on(doc, today), "tomorrow": schedules_on(doc, today + timedelta(days=1)),
            "changes_today": history_entries(doc, day=today),
            "b_status": "not_connected" if b_records is None else "ok",
            "b_requests": [] if b_records is None else
            [r for r in b_records if (ss.parse_iso(r.get("executed_at")) or now - timedelta(days=9999)).date() == today]}


# %% B기능 요청 기록 읽기 어댑터 (읽기 전용, B 연결 후에만 사용)
# 핵심 요약: 명세 2.3의 모의 요청 기록(JSON Lines: action_id·type·payload.message·approved_at·executed_at·mock)을 읽기만 한다.
# 형식이 틀린 줄은 건너뛰고 개수만 센다. 파일을 만들거나 고치지 않는다.
def read_living_action_log(path: str | Path) -> dict[str, Any]:
    """{"status": ok | missing | error, "records", "skipped"}"""
    file = Path(path)
    if not file.exists():
        return {"status": "missing", "records": [], "skipped": 0}
    records, skipped = [], 0
    try:
        lines = file.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return {"status": "error", "records": [], "skipped": 0}
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            skipped += line.strip() != ""
            continue
        payload = row.get("payload") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or row.get("type") != "request_caregiver_help" or row.get("mock") is not True
                or not isinstance(row.get("action_id"), str) or not isinstance(payload, dict)
                or not isinstance(payload.get("message"), str) or ss.parse_iso(row.get("executed_at")) is None):
            skipped += 1
            continue
        records.append({"action_id": row["action_id"], "message": payload["message"],
                        "items": payload.get("items") if isinstance(payload.get("items"), list) else [],
                        "approved_at": row.get("approved_at"), "executed_at": row["executed_at"]})
    return {"status": "ok", "records": records, "skipped": skipped}
