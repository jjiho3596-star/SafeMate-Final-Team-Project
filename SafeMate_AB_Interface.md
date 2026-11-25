# SafeMate A·B기능 공통 인터페이스 명세서

- 기준일: 2026-11-25 (4단계 공통 LangGraph 구현 반영)
- 기준 코드: `a_outing.py`(1단계), `safety_rag.py`(2단계), `outing_decision.py`(3단계), `safemate_graph.py`(4단계, 커밋 `4a2338e`)
- 1·2·3단계 파일은 4단계에서 바꾸지 않았습니다. (커밋 `c4e2ca0`과 SHA-256 동일)
- 이번 개정(4단계 현행화)에서 고친 곳
  - **1장·2장: 바꾸지 않았습니다.** A 공개 함수·반환 규격과 B 공개 인터페이스는 이전에 전달한 그대로입니다.
  - 3장: 실제 구현에 맞춰 다시 썼습니다. (State, 노드, 조건부 분기, interrupt·Command 승인, 중복 실행 방지, A·B 독립 설정, 앱 함수, 5단계 `apply_outing_change()` 계약, B 담당자 확인 사항)
  - 4장: 실제 테스트 결과와 아직 검증하지 않은 사항을 나눠 적었습니다. (실제 Qwen 일정 변경 분류 한계, B기능 미연결 포함)
- 이전 개정(3단계 2차 실측 결함 수정)에서 고친 곳: 1.3의 처리 순서·반환값·오류 턴 대화 상태, 2장의 B 오류 턴 상태·`living_evidence`·`payload.message`·`approval` 규칙

## 이 문서를 읽는 방법

| 표시 | 뜻 |
|---|---|
| **[구현됨]** | 지금 코드에 실제로 있고, 테스트로 확인한 내용입니다. 이름·키·값을 그대로 믿고 써도 됩니다. |
| **[제안]** | 아직 코드가 없습니다. 팀이 합의해서 앞으로 만들 규격입니다. 이름은 바꿀 수 있습니다. |

- [구현됨]은 Mock 테스트 기준입니다.
- 실제 Ollama·Open-Meteo·Intel XPU로 어디까지 확인했는지는 4장 표에 따로 적었습니다.
- **B기능 코드가 아직 없으므로 실제 A·B 통합은 검증하지 않았습니다.** 4단계 그래프의 B 연결부는 2장 계약을 지키는 가짜 B로만 확인했습니다.

B기능 담당자는 **2장만 AI에게 전달해도** 상담 함수와 모의 실행 함수를 만들 수 있게 썼습니다.
2장에서 A기능 쪽 정보가 필요한 부분(공통 승인 데이터 구조)은 2장 안에 다시 적어 두었습니다.

| 구분 | 함수 | 상태 | 담당 |
|---|---|---|---|
| A 1단계 | `get_outing_weather()`, `resolve_city_coordinates()`, `parse_outing_time()` | [구현됨] | A기능 |
| A 2단계 | `configure_embedding_device()`, `prepare_safety_index()`, `search_safety_guidance()` | [구현됨] | A기능 |
| A 3단계 | `run_outing_decision()` | [구현됨] | A기능 |
| B 상담 | `run_living_support()` | [제안] | B기능 담당 |
| B 모의 실행 | `execute_living_action()` | [제안] | B기능 담당 |
| 4단계 공통 | `build_graph()`, `send_message()`, `press_button()`, `end_consultation()` 등 (LangGraph 그래프, 승인·거절 처리) | [구현됨] (3장) | 통합 담당 |
| 5단계 | `apply_outing_change()` (승인된 가상 일정 변경 저장) | [제안] (3.7) | 통합 담당 |
| 6단계 | Gradio 화면, 최종 통합 | [제안] (3.6) | 통합 담당 |

---

# 1. A기능 1·2·3단계에 실제로 구현된 인터페이스 [구현됨]

## 1.1 1단계 `a_outing.py` (날씨)

| 항목 | 내용 |
|---|---|
| `KST` | 한국 표준시 `tzinfo`. 모든 날짜 계산의 기준 |
| `FORECAST_DAYS` | `16` (오늘부터 16일까지만 예보 조회) |
| `WeatherError(status, message)` | 1단계 예외. `status`는 아래 상태 값 중 하나 |

```text
get_outing_weather(outing_date: str | None = None, outing_time: str | None = None,
                   city: str | None = None) -> dict
resolve_city_coordinates(city: str) -> tuple[float, float]      # 실패 시 WeatherError
parse_outing_time(outing_date: str | None, outing_time: str | None, now: datetime) -> datetime
departure_hour_window(outing: datetime) -> tuple[datetime, datetime]   # [정시, 정시+1시간)
```

- `outing_date`: `"오늘"`, `"내일"`, `"모레"` 또는 `"YYYY-MM-DD"`
- `outing_time`: `"HH:MM"` (24시간제)
- 반환: `{"status", "message", "weather"}`
- 상태 값: `ok`, `partial`, `needs_input`, `out_of_range`, `api_error`, `network_error`, `data_unavailable`
- `weather`는 조회에 실패하면 `None`입니다. 조회했을 때의 키는 아래와 같습니다.
  - `data_kind`, `source`(`"Open-Meteo (CC BY 4.0) · https://open-meteo.com/"`), `city`
  - `outing_at`, `hour_start`, `hour_end`
  - `temperature_c`(°C), `apparent_temperature_c`(°C), `wind_speed_mps`(m/s), `precipitation_mm`(mm), `precipitation_probability_pct`(%)
  - `missing_fields`
- 규칙
  - 값이 없으면 `None`입니다. 0으로 채우지 않습니다.
  - 기온·풍속은 구간 시작 행, 강수량·강수확률은 구간 끝 행(직전 1시간 합계)에서 읽습니다.
  - `'광주'`처럼 이름만으로 정할 수 없는 지역은 `needs_input`으로 다시 묻습니다.
  - 지난 시각이나 16일을 넘는 날짜는 `out_of_range`입니다.
  - 시각은 내부에서 `datetime.now(KST)`로 판단합니다.

## 1.2 2단계 `safety_rag.py` (공식 안전 문서 RAG)

```text
configure_embedding_device(requested: str) -> str        # "xpu" | "cuda" | "cpu", 작은 텐서 연산으로 실제 확인
prepare_safety_index(device: str, *, data_dir=None, cache_dir=None) -> SafetyIndex
search_safety_guidance(question: str | None, index: SafetyIndex) -> dict
```

- 장치 규칙
  - 요청한 장치에서 실제 연산이 되지 않으면 `SafetyError("device_error")`를 냅니다.
  - **CPU로 조용히 바꾸지 않습니다.**
- `SafetyIndex`의 키
  - `retriever`, `device`(dict), `chunks`, `reused`, `path`
  - 색인은 `.safemate/chroma/<corpus_key>`에 저장됩니다.
- 임베딩 모델
  - `paraphrase-multilingual-MiniLM-L12-v2`(고정 revision)
  - 조각 크기 500, 겹침 50, 최소 유사도 0.30
- 반환: `{"status", "message", "count", "results"}`
  - 상태 값: `ok`, `insufficient_evidence`, `needs_input`, `search_error`
  - `results[]`의 키: `content`, `title`, `publisher`, `url`, `filename`, `pdf_page`, `printed_page`, `similarity`
  - 상위 2개만 돌려줍니다.
- 문서 3종(`SOURCES`)
  - 질병관리청 폭염: `kdca_heat_2026.pdf`
  - 경기도 자연재난과 호우·강풍: `gg_rain_wind_2026.pdf`
  - 경기도 뉴스포털 한파·대설: `gg_cold_snow_2025.pdf`

## 1.3 3단계 `outing_decision.py` (외출·날씨 상담, LangGraph가 부를 공개 함수)

```text
run_outing_decision(request: dict | str, *, device: str | None = None, index=None,
                    conversation_state: dict | None = None) -> dict
```

**처리 순서**

1. LLM 이해 (Ollama `/api/chat`, JSON 스키마, 온도 0)
2. Python 정규화
   - 날짜·시간대·'이따' 해석
   - 짧은 대답('인천이야')을 직전 질문에 연결
   - '~말고'처럼 의도를 바꾼 말은 직전 질문의 대답으로 보지 않음
3. 필요한 도구만 실행: 1단계 날씨(최대 4회), 2단계 RAG(최대 3주제)
4. 답변 만들기
   - **부족한 정보만 물으면 되는 턴**(보여 줄 예보·공식 근거가 없고 의도가 분명함)은 LLM 답변을 부르지 않습니다.
     Python 질문(`clarification_question()`)을 그대로 씁니다. → `answer_source="python_question"`
   - 그 밖의 턴은 LLM이 답변을 씁니다.
5. Python 검증
   - 숫자·시각·출처·행동 주장
   - **원본 예보와 모순되는 날씨 설명**: 비 여부, 조회하지 않은 하늘 상태, 다른 지역, "전혀" 같은 보장 표현
6. 검증을 통과하지 못하거나 답변 단계에서 Ollama 오류가 나면, Python이 확인된 자료만으로 대체 답변을 만듭니다. → `answer_source="python_fallback"`
7. 예보를 조회한 턴의 `answer`에는 Python이 원본 값으로 쓴 `[확인한 예보]` 줄이 붙습니다.
   - 모델은 수치를 옮겨 적지 않고 상황 판단·조언·일정 제안에 집중합니다.

LLM 호출은 요청당 최대 4회입니다. 보통 이해 1회 + 답변 1회이고, 부족한 정보만 묻는 턴은 이해 1회입니다.

### 입력 `request`

문자열이면 `{"user_message": 문자열}`로 처리합니다. 딕셔너리로 넘길 때의 키는 아래와 같습니다.
화면이나 LangGraph가 직접 넘긴 값은 LLM이 추출한 값보다 우선합니다.

| 키 | 형식 | 비고 |
|---|---|---|
| `user_message` | str, 600자 이내 | 사용자 발화 |
| `city`, `purpose` | str, 160자 이내 | |
| `outing_date` | `"오늘"`, `"내일"`, `"YYYY-MM-DD"` 등 | |
| `outing_time`, `alternative_time` | `"HH:MM"` | |
| `alternative_date` | 날짜 | `alternative_time`과 함께 넘겨야 합니다 |
| `appointment_at` | `"YYYY-MM-DDTHH:MM"` | 시간대가 없으면 KST로 봅니다 |
| `time_period` | `새벽·아침·오전·점심·낮·오후·저녁·밤·now·soon` | |
| `can_reschedule` | bool | |
| `departure_lead_minutes` | int, 1~2880 | |
| `user_reported_conditions` | list[str], 최대 8개 | 사용자가 직접 말한 상황. 관측값으로 쓰지 않습니다 |
| `intent` | `INTENTS` 중 하나 | 문장 없이 구조화 값만 넘길 때 사용 |

- `device`: `"xpu"`, `"cuda"`, `"cpu"` 중 하나입니다. RAG가 필요할 때만 씁니다.
  - 없으면 환경변수 `SAFEMATE_EMBED_DEVICE`를 봅니다.
  - 둘 다 없는데 RAG가 필요하면 `rag_unavailable`/`device` 오류를 냅니다.
- `index`: 미리 준비한 `SafetyIndex`입니다. 주면 색인을 다시 만들지 않습니다.
- `conversation_state`: 직전 턴 결과의 `conversation_state`를 그대로 넘깁니다. `None`이면 새 대화를 시작합니다.

### 반환 딕셔너리 (최상위 키 26개, 항상 모두 있음, JSON 직렬화 가능)

| 키 | 형식 | 뜻 |
|---|---|---|
| `status` | str | `ok`, `needs_input`, `weather_unavailable`, `evidence_insufficient`, `rag_unavailable`, `llm_unavailable`, `llm_invalid_output` |
| `detail_code` | str \| None | 구체적 이유. 예: `missing_fields`, `intent_unclear`, `city_needs_input`, `answer_verification`, `connection_refused`, `model_missing`, `device` |
| `message` | str | 개발자용 상태 설명 (화면용 아님) |
| `route` | str | LangGraph 분기 이름 (아래 표) |
| `intent` | str \| None | `weather_question`, `outing_consultation`, `safety_guidance`, `schedule_change`, `daily_conversation`, `living_support` |
| `answer` | str | **어르신에게 보여 줄 최종 문장** (항상 비어 있지 않음). 순서: 답변 본문 → `[공식 안전 안내]` 원문 인용 → `[확인한 예보]`(Python 작성) → 필요한 질문 1개 |
| `outing_request` | dict | 정규화한 지역·날짜·시간대·시각 등 |
| `conversation_state` | dict | 다음 턴에 넘길 A 상태. 항상 dict. 오류 턴에 무엇을 돌려주는지는 아래 '오류 턴의 대화 상태' 표 |
| `missing_fields` | list[str] | `city`, `outing_date`, `outing_time`, `appointment_at` 중 부족한 것 |
| `clarification` | dict \| None | `{"fields", "question", "reasons"}`. `missing_fields`가 있을 때만 있음 |
| `weather_status`, `weather_message` | str \| None | 1단계 결과 상태 |
| `weather` | dict \| None | `slots[]`(구간별 예보 + `fact_id` W1…), `city`, `date`, `date_label`, `basis`, `label`, `failed_windows`, `skipped_past_windows`, `complete`, `reused` 등 |
| `topics`, `rag_status` | dict | 검색 주제와 주제별 검색 상태 |
| `safety_evidence` | list | 검증을 통과한 검색 원문(`id` E1…, `content`, 출처) |
| `official_guidance` | list | 화면에 인용한 공식 원문 `{"evidence_id", "text", "publisher", "title", "pdf_page", "url"}` |
| `embedding_device` | dict \| None | 실제 사용한 임베딩 장치 정보 |
| `comparison` | dict \| None | 다른 시각과 비교한 결과 (일정 변경 질문일 때) |
| `llm` | dict | `model`, `calls`, `understanding`, `discarded_slots`, `rules_applied`, `verification[]` (답변 검증 기록. 뒤에 다른 오류가 나도 지우지 않음) |
| `decision` | dict \| None | `kind`, `suggest_schedule_change`, `official_points`, `used_fact_ids`, `question`, `verified`, `answer_source`. 답변 단계 전에 오류가 나면 `None` |
| `requires_user_confirmation` | bool | 승인이 필요한 제안이 있으면 `True` |
| `proposed_action` | dict \| None | 승인 대기 제안 (아래) |
| `usage` | dict | `llm_calls`, `weather_calls`, `rag_searches`, `weather_reused` |
| `tool_trace` | list | 도구 실행 기록 (`ollama_chat`, `get_outing_weather`, `search_safety_guidance`, `clarification_question`, `answer_verification` 등) |
| `limitations` | list[str] | 확인하지 않은 범위 (예: "하루 전체 예보가 아닙니다") |

**`route` 값 (LangGraph 분기)**

| route | 언제 | LangGraph가 할 일 |
|---|---|---|
| `weather_question`, `outing_consultation`, `safety_guidance`, `daily_conversation` | 답변 완료 | `answer` 표시 |
| `schedule_change` | 일정 비교 완료 | `requires_user_confirmation`이면 승인 노드로 |
| `living_support` | 생활 지원 요청으로 이해함 (도구 미실행) | B기능으로 넘김 |
| `needs_clarification` | `status == "needs_input"` | `answer`(질문) 표시 후 다음 발화를 기다림 |
| `error` | 입력·설정·이해 LLM·색인 준비 오류, 또는 답변 단계 LLM 실패 후 Python 대체 답변 | `answer` 표시, 필요하면 재시도 안내. `conversation_state`는 그대로 저장 |

**오류 턴의 대화 상태** (이전 명세서 정정)

이전 명세서에는 "오류 턴에는 이전 상태를 그대로 돌려준다"고 적었습니다. 이 설명은 아래 첫 번째 줄에만 맞습니다.

| 오류가 난 단계 | 예 (`status`/`detail_code`) | 돌려주는 `conversation_state` | `route` |
|---|---|---|---|
| 이번 턴 정보를 확정하기 전: 입력 검사, 대화 상태 검사, Ollama 설정, **이해 LLM**, 날짜 정규화, RAG 색인·장치 준비 | `needs_input/conversation_state`, `llm_unavailable/connection_refused`, `llm_invalid_output/understanding`, `rag_unavailable/device` | **이전 상태 그대로** (`turn`도 그대로) | `error` (입력 오류 `needs_input`이면 `needs_clarification`) |
| **답변 단계**: 답변 LLM의 검증 실패·응답 형식 오류·연결 끊김·호출 상한 | `llm_invalid_output/answer_verification`, `llm_invalid_output/response_format`, `llm_unavailable/inference_timeout` 등 | **이번 턴까지 저장한 새 상태** (확인한 지역·날짜, 예보 캐시, `pending_question`, `turn`+1) | `error`. 정보를 묻는 턴이면 `needs_input`을 유지해 `needs_clarification` |

- 답변 단계 오류여도 실제 오류는 숨기지 않습니다. `status`·`detail_code`·`message`·`llm.verification`·`tool_trace`에 그대로 남습니다.
  - 정보를 묻는 턴(`needs_input` 유지)에서는 `message` 끝에 "(답변 생성 오류(…)로 Python 추가 질문을 사용했습니다)"가 붙습니다.
- 확인하지 않은 정보는 채우지 않습니다. 저장하는 것은 이번 턴에 사용자가 말했거나 도구로 확인한 값뿐입니다.
- 정상적으로 Python 질문을 쓴 턴(`python_question`)은 오류가 아닙니다. `llm.verification == []`, `decision.verified is None`입니다.
- LangGraph는 오류 턴이어도 **돌려받은 `conversation_state`를 그대로 저장하면 됩니다.** 어느 상태를 돌려줄지는 A가 이미 정했습니다.

**승인 제안 `proposed_action` (A가 만드는 유일한 종류)**

```json
{"type": "propose_outing_time_change",
 "original_outing_at": "2026-11-25T14:00+09:00", "proposed_outing_at": "2026-11-25T16:00+09:00",
 "reason": "두 예보의 차이입니다. 한 시간 강수량 2 → 0mm, 강수확률 80 → 10%",
 "requires_approval": true, "applied": false}
```

A는 일정을 실제로 바꾸지 않습니다. 파일을 쓰지 않고, 보호자에게 연락하지도 않습니다.
`applied`는 항상 `false`입니다.

**`conversation_state` (A 전용, 다른 기능이 내용을 고치면 안 됨)**

```json
{"version": 1, "turn": 0, "intent": null,
 "slots": {"city": null, "city_candidate": null, "outing_date": null, "date_label": null, "time_period": null,
           "outing_time": null, "time_ambiguous_text": null, "appointment_time": null, "departure_lead_minutes": null,
           "alternative_date": null, "alternative_time": null, "purpose": null, "can_reschedule": null,
           "user_reported_conditions": []},
 "pending_question": null, "history": [], "weather_cache": null, "checked_city": null}
```

- `pending_question`의 키: `{"fields", "question", "for_weather", "intent"}`
- `history`: 최근 6문장만 남깁니다.
- `weather_cache`: 30분 안에만 다시 씁니다.
- 형식이 다른 상태를 넘기면 `needs_input`/`conversation_state` 오류를 냅니다. (B 상태를 실수로 넘겨도 섞이지 않음)

### 반환값 일관성 규칙 (T19·T20·P03·P04 테스트로 확인)

1. `status == "needs_input"` ⇔ `route == "needs_clarification"`
2. `missing_fields`가 비어 있지 않음 ⇔ `clarification`이 있음
   - `clarification.fields == missing_fields`
   - `clarification.question`은 비어 있지 않음
3. 답변까지 만든 턴의 `conversation_state.pending_question`
   - 이번 `clarification`과 같은 `fields`·`question`을 가집니다.
   - `missing_fields`가 없으면 `None`입니다.
4. `requires_user_confirmation` ⇔ `proposed_action`이 있음
   - 제안은 `requires_approval: true`, `applied: false`입니다.
5. `decision.answer_source`는 `"llm"`, `"python_question"`, `"python_fallback"` 중 하나입니다.

   | `answer_source` | 뜻 | `decision.verified` |
   |---|---|---|
   | `llm` | 검증을 통과한 모델 답변 | `True` |
   | `python_question` | 부족한 정보만 묻는 Python 질문 (정상 처리, 오류 아님). 이때 `status == "needs_input"`, `llm.verification == []` | `None` (검증할 모델 답변이 없음) |
   | `python_fallback` | 답변 단계 LLM이 실패해 Python 대체 답변을 씀 | `False` |

### 연동 측이 알아야 할 변경 ① 1차 최종 안정화

- **`decision`에 키 추가**
  - `answer_source`가 새로 생겼습니다.
  - `verified`는 이제 `False`일 수 있습니다. (최상위 키는 그대로)
- **검증 실패 처리**
  - 모델 답변이 검증을 통과하지 못하면, Python이 만든 사용자용 대체 답변이 `answer`에 들어갑니다.
  - 이전처럼 "답변 문장을 검증하지 못해…"라는 개발자용 문장은 나오지 않습니다.
  - 정보를 묻는 턴이면 `needs_input`을 유지하고 `pending_question`을 저장합니다.
  - 그 밖의 턴은 `llm_invalid_output`/`answer_verification`으로 표시하고, `route`는 `error`입니다.
  - 두 경우 모두 대화 상태는 다음 턴으로 이어집니다.
  - `tool_trace`에 `{"tool": "answer_verification", "status": "replaced_by_python_fallback"}`가 남습니다.
- **`'이따·있다가'` 해석**
  - 오늘로 봅니다. 그 시간대가 오늘 이미 끝났으면 날짜를 다시 묻습니다.
  - 시간대 범위(저녁 18~21시 등)는 SafeMate 프로젝트 설정이며, 공식 기상 구분이 아닙니다.

### 연동 측이 알아야 할 변경 ② 2차 실측 결함 수정 (이번)

공개 함수 시그니처, 최상위 반환 키 26개, `decision` 키 7개, `route` 값, LLM 호출 상한, 승인 전 실행 금지는 **바뀌지 않았습니다.**

| 바뀐 점 | 연동 측 영향 | 관련 실측 |
|---|---|---|
| 의도를 바꾼 말('날씨 말고 …', '그게 아니라 …')은 새 의도를 따릅니다. 모델이 '직전 질문의 대답'이라고 표시해도 직전 질문이 물은 항목을 채우지 않았으면 대답으로 보지 않습니다. | `llm.rules_applied`에 `not_an_answer_to_pending_question` | L1_05 |
| 부족한 정보만 묻는 턴은 답변 LLM을 부르지 않습니다. | `usage.llm_calls` 2 → 1, `answer == clarification.question`, `answer_source="python_question"`, `tool_trace`에 `clarification_question`, `rules_applied`에 `clarification_by_python_question` | L1_07 1턴 |
| 답변 단계 오류가 나도 대화 상태를 저장합니다. | 위 '오류 턴의 대화 상태' 표. 다음 턴의 짧은 대답이 이어지고, 이미 조회한 예보는 다시 조회하지 않습니다(30분 캐시). | L1_07 2턴 |
| Python이 원본 예보로 '날씨 요약'을 만들어 모델에게 주고, 모순되는 모델 답변은 검증에서 막습니다. | `llm.verification`에 이유가 남고, 대체 답변이면 `python_fallback` | L1_08 |
| 예보 수치는 Python이 원본 값으로 `[확인한 예보]` 줄에 씁니다. | 화면은 `answer`를 그대로 보여 주면 됩니다. 수치를 따로 붙이지 마세요. | L1_08 |
| 번호·기호만 있는 원문 조각("3.")은 인용하지 않습니다. 쓸 문장이 없으면 인용 줄을 붙이지 않습니다. 원문은 고치지 않습니다. | `official_guidance`에 그런 조각이 들어가지 않음 | L1_04 |
| 모델 답변이 이미 같은 항목을 물었으면 Python 질문을 덧붙이지 않습니다. | 한 답변에 질문이 두 번 나오지 않음 | L1_01·L1_03·L1_06 |

**날씨 설명 검사 기준** (Python, 원본 예보 기준)

- 비 예보 여부
  - 한 시간 강수량 > 0mm 또는 강수확률 ≥ 50%인 구간은 '비 예보 있음', 그 밖에 값이 있으면 '비 예보 없음'입니다.
  - 50%는 SafeMate 프로젝트 설정(`RAIN_LIKELY_PCT`)이며 공식 기상 기준이 아닙니다.
  - '비 예보 없음'인데 비가 온다고 하거나, '비 예보 있음'인데 안 온다고 하면 검증 실패입니다.
- "전혀·절대·100%"처럼 예보를 보장하는 말은 검증 실패입니다.
- 1단계는 하늘 상태(맑음·흐림·햇빛·구름)를 조회하지 않으므로, 이를 단정하면 검증 실패입니다.
- 조회한 지역이 아닌 특별·광역시(서울·부산·대구·인천·광주·대전·울산·세종)의 날씨를 단정하면 검증 실패입니다.
  - 도·시군 이름은 검사하지 않습니다. (4장 한계)
- 기온과 체감온도, mm와 %의 혼동은 기존 숫자 검사(`verify_answer` 4번: 항목 이름·단위·예보 값 대조)가, 조회하지 않은 시간대는 시각 검사(5·6번)가, 조회하지 않은 항목은 8번이 막습니다.

### Ollama 설정 (환경변수, 요청마다 읽음)

| 변수 | 기본값 |
|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | `qwen2.5:3b` (A기능 확정 모델) |
| `OLLAMA_CONNECT_TIMEOUT` / `OLLAMA_TIMEOUT` | 5초 / 120초 |
| `OLLAMA_MAX_TOKENS` / `OLLAMA_TEMPERATURE` / `OLLAMA_NUM_CTX` | 1024 / 0.3 / 8192 |

- 클라우드 모델(`-cloud`, `ollama.com`)은 거절합니다.
- 모델 자동 다운로드는 하지 않습니다.

---

# 2. B기능이 맞춰야 하는 최소 공통 인터페이스 [제안]

> B기능 담당자의 AI에게 이 2장을 그대로 전달하세요.
> 목표는 **상담(제안)**과 **승인 후 모의 실행**을 서로 다른 두 함수로 나누는 것입니다.

## 2.0 역할 분담

| 담당 | 하는 일 |
|---|---|
| B기능 담당 | 생활지원 RAG, LLM 상담, 도움 요청 제안(`run_living_support`), 승인 후 모의 요청 기록(`execute_living_action`) |
| 통합 담당 | 공통 LangGraph, 사용자 승인·거절 처리, Gradio 화면, 최종 통합 |

B기능 담당은 LangGraph·Gradio·승인 버튼을 만들지 않습니다. 아래 두 함수만 이 계약대로 만들면 됩니다.

## 2.1 지켜야 할 원칙

1. `run_living_support()`는 **제안만** 합니다.
   - 보호자 요청이나 기록 저장을 실행하지 않습니다.
   - `execute_living_action()`을 부르지 않습니다.
2. `execute_living_action()`은 **LangGraph가 사용자 승인을 받은 뒤에만** 부릅니다.
   - 실제 메시지를 보내지 않는 **모의 실행**입니다. 기록만 남깁니다.
3. 개인정보(이름·전화번호·주소·보호자 연락처)를 묻거나 저장하지 않습니다.
   - 보호자는 `"caregiver"`라는 역할 이름으로만 표시합니다.
4. 두 함수의 반환값은 모두 JSON으로 저장할 수 있는 `dict`여야 합니다. 예외를 밖으로 던지지 않고 `status`로 알립니다.
5. 유료 API와 새 모델 자동 다운로드는 쓰지 않습니다. LLM은 로컬 Ollama만 씁니다. (모델 설정은 3.4)
6. 승인 정보(`approval`)는 만들지 않습니다. 실제 사용자 승인을 확인한 공통 LangGraph만 붙입니다. (2.4)
7. 공식 자료에 없는 지원제도·기관·조건을 지어내지 않습니다. 답변은 `living_evidence`에 있는 내용 안에서만 말합니다. (2.2)
8. 물품·수량을 확인하지 못했으면 지어내지 말고 묻습니다. (2.4 `payload.message`)

## 2.2 상담 함수

```python
def run_living_support(request: dict | str, *, device: str | None = None, index=None,
                       conversation_state: dict | None = None) -> dict:
    ...
```

**입력**

- `request`
  - 문자열이면 사용자 발화입니다.
  - 딕셔너리이면 `{"user_message": str(600자 이내), "context": dict | None}`입니다.
  - `context`(선택)는 A기능이 넘긴 정보입니다. 예: `{"source": "outing_decision", "user_goal": "장보기 도움"}`
- `device`, `index`(선택): B가 문서 검색을 쓸 때만 씁니다. 쓰지 않으면 무시해도 됩니다. A와 호출 모양을 맞추기 위한 자리입니다.
- `conversation_state`(선택): B가 직전 턴에 돌려준 `conversation_state`입니다. `None`이면 새 대화입니다.

**반환 (필수 키, 항상 모두 있어야 함)**

| 키 | 형식 | 규칙 |
|---|---|---|
| `status` | str | `ok`, `needs_input`, `llm_unavailable`, `llm_invalid_output`, `tool_error` |
| `detail_code` | str \| None | `ok`이면 `None`, 그 밖에는 짧은 영문 코드 (예: `missing_items`, `connection_refused`) |
| `message` | str | 개발자용 설명. 없으면 `""` |
| `route` | str | `living_support`, `needs_clarification`, `error` |
| `answer` | str | 어르신에게 보여 줄 문장. **항상 비어 있지 않음** |
| `conversation_state` | dict | B 전용 상태. **항상 dict**. 오류 턴 규칙은 아래 |
| `missing_fields` | list[str] | 부족한 정보 이름. 없으면 `[]` (예: `items`, `quantity`) |
| `clarification` | dict \| None | `{"fields": list[str], "question": str, "reasons": dict}`. 물을 것이 없으면 `None` |
| `requires_user_confirmation` | bool | 승인이 필요한 제안이 있으면 `True` |
| `proposed_action` | dict \| None | 2.4의 승인 데이터 구조. 제안이 없으면 `None` |
| `usage` | dict | 최소 `{"llm_calls": int}` |
| `tool_trace` | list[dict] | 도구·LLM 호출 기록. 없으면 `[]` |
| `limitations` | list[str] | 확인하지 못한 점. 없으면 `[]` |
| `living_evidence` | list[dict] | **(이번 추가)** 답변에 쓴 생활지원 자료 원문. 없으면 `[]` (아래 규격) |

선택 키: `items`(정리한 물품 목록), `llm`(모델 이름·검증 기록), `decision`.
B가 필요하면 더 넣어도 됩니다. LangGraph는 위 필수 키만 사용합니다.

**`None`과 빈 값 규칙**

- 목록은 항상 `[]`를 씁니다. 목록에 `None`을 쓰지 않습니다.
- 해당 사항이 없는 "덩어리"(`clarification`, `proposed_action`)는 `None`입니다.
- 문장(`answer`, `message`)은 `None`을 쓰지 않습니다. `message`가 없으면 `""`입니다.

**일관성 규칙 (B 테스트에 꼭 넣을 것)**

1. `requires_user_confirmation == (proposed_action is not None)`
2. `proposed_action`이 있으면 2.4 규격을 만족합니다.
   - `requires_approval is True`, `applied is False`이고, `action_id`가 있습니다.
   - 이때 `status == "ok"`, `route == "living_support"`입니다.
3. `status == "needs_input"` ⇔ `route == "needs_clarification"`
   - 이때 `proposed_action is None`입니다.
4. `bool(missing_fields) == (clarification is not None)`
   - `clarification["fields"] == missing_fields`
5. `status`가 `llm_unavailable`, `llm_invalid_output`, `tool_error`이면 `route == "error"`, `proposed_action is None`입니다.
6. 사용자에게 "보호자에게 연락했어요"처럼 **실행이 끝난 것처럼 말하지 않습니다.** 승인 전이기 때문입니다.
7. 결과 어디에도 `approval` 키를 만들지 않습니다.

**오류 턴의 `conversation_state` 규칙** (이전 명세서 정정)

이전 명세서는 "오류 턴에는 이전 상태를 그대로"라고만 적었습니다. A의 실제 동작(1.3 '오류 턴의 대화 상태')에 맞춰 아래처럼 정정합니다.

| 오류가 난 때 | 돌려줄 `conversation_state` |
|---|---|
| 이번 턴의 정보를 확정하기 전 (입력 검사, 상태 검사, 이해 단계 LLM 실패) | 이전 상태 그대로 |
| 이해한 정보가 확정된 뒤 (검색·답변 단계 실패) | 이전 상태 그대로 **또는** 이번 턴에 사용자가 말한 정보까지 저장한 새 상태. B가 고릅니다. |

- 어느 쪽이든 확인되지 않은 물품·수량을 채우지 않습니다.
- 실제 오류는 `status`·`detail_code`·`tool_trace`에 남깁니다. 오류를 숨기려고 `ok`로 바꾸지 않습니다.
- LangGraph는 어느 경우든 돌려받은 상태를 그대로 저장합니다.

**`living_evidence` 규격** (B 전용, 이번 추가)

```json
"living_evidence": [
  {"content": "실제로 검색된 생활지원 자료 원문",
   "title": "문서 제목",
   "publisher": "발행 기관",
   "url": "실제 출처 URL",
   "pdf_page": 3}
]
```

| 키 | 형식 | 규칙 |
|---|---|---|
| `content` | str | 검색된 원문 그대로. 비어 있지 않음. 요약·재작성하지 않음 |
| `title`, `publisher`, `url` | str \| None | 자료에서 실제로 확인한 값. 확인할 수 없으면 `None` (지어내지 않음) |
| `pdf_page` | int \| None | 1부터 세는 PDF 쪽 번호. PDF가 아니거나 알 수 없으면 `None` |

- 검색을 하지 않았거나 근거를 찾지 못했으면 `[]`입니다.
  - 이때도 답변을 만들었으면 `status`는 `ok`입니다. `limitations`에 "공식 생활지원 자료에서 근거를 찾지 못했습니다."를 넣습니다.
  - 검색 자체가 실패하면 `tool_error`입니다. (예: `detail_code="rag_search"`)
- 답변은 `living_evidence`에 있는 지원제도·기관·조건만 말합니다. 근거가 없으면 "공식 자료에서 확인하지 못했다"고 말합니다.
- A의 `safety_evidence`·`official_guidance` 구조를 따르지 않아도 됩니다. Gradio 출처 표시는 3.1의 변환 함수가 맞춥니다.
- 이 필드는 Gradio 출처 표시와 B RAG 검증에 씁니다.

**A와 다른 점 (B가 따라 하지 않아도 되는 것)**

A의 내부 구조나 예보 전용 데이터는 B에 요구하지 않습니다. 차이는 공통 LangGraph가 3.1의 작은 변환으로 맞춥니다.

| 항목 | A (구현됨) | B (제안) | 공통 LangGraph가 할 일 |
|---|---|---|---|
| `status` 값 | `ok`, `needs_input`, `weather_unavailable`, `evidence_insufficient`, `rag_unavailable`, `llm_unavailable`, `llm_invalid_output` | `ok`, `needs_input`, `llm_unavailable`, `llm_invalid_output`, `tool_error` | 분기는 `route`로만 합니다. 경고 표시는 `status not in ("ok", "needs_input")` 하나로 판단 |
| `route` 값 | 의도 이름 6종, `needs_clarification`, `error` | `living_support`, `needs_clarification`, `error` | `needs_clarification`·`error`·`requires_user_confirmation`만 공통으로 보고, A의 `living_support`는 B로 넘김 |
| 근거 | `official_guidance`(인용한 원문, 키 `text`), `safety_evidence` | `living_evidence`(키 `content`) | 출처 표시용으로 키 이름만 맞춤 |
| 예보·비교 | `weather`, `comparison`, `outing_request` | 없음 | B에 요구하지 않음 |
| 진단 | `decision.answer_source`, `llm.verification` | 선택 (`llm`, `decision`) | 개발자 패널에만 표시. 없으면 생략 |
| `usage` | `llm_calls`, `weather_calls`, `rag_searches`, `weather_reused` | 최소 `llm_calls` | `usage.get("llm_calls", 0)`처럼 읽음 |
| 승인 제안 | `propose_outing_time_change` (공통 봉투 키 일부 없음) | `request_caregiver_help` (공통 봉투) | A 제안만 래퍼로 보완 (2.4) |

## 2.3 승인 후 모의 실행 함수

```python
def execute_living_action(proposed_action: dict) -> dict:
    ...
```

LangGraph 승인 노드가 사용자의 "승인"을 받은 뒤에만 부릅니다.
LangGraph는 부르기 전에 `proposed_action["approval"] = {"status": "approved", "at": "<KST ISO 시각>"}`를 붙입니다.
`approval`은 Gradio 승인 버튼으로 실제 사용자 승인을 확인한 공통 LangGraph만 붙입니다. LLM이나 상담 함수가 만들지 않습니다.

**처리 순서**

1. 검증 (하나라도 틀리면 기록하지 않고 `invalid_action`)
   - `type`이 허용 목록(`request_caregiver_help`)에 있어야 합니다.
   - `requires_approval is True`, `applied is False`이고, `action_id`가 비어 있지 않은 문자열이어야 합니다.
   - `payload`가 2.4 규격을 만족해야 합니다.
2. 승인 확인: `approval`이 없거나 `approval.status != "approved"`이면 `rejected`를 돌려주고 기록하지 않습니다. (실수로 불려도 실행되지 않게)
3. 중복 방지(멱등성): 같은 `action_id`가 이미 기록돼 있으면 새로 기록하지 않습니다. `already_executed`와 기존 기록을 돌려줍니다.
4. 모의 실행 기록
   - JSON Lines 한 줄을 추가합니다.
   - 기본 경로는 `.safemate/mock_action_log.jsonl`이고, 환경변수 `SAFEMATE_ACTION_LOG`로 바꿀 수 있습니다.
   - `.safemate/`는 Git에 올리지 않습니다.
   - 기록 항목: `action_id`, `type`, `payload`(`message` 포함), `approved_at`, `executed_at`, `"mock": true`
   - `payload.message`는 승인받은 문장 그대로 기록합니다. 실행 함수가 문장을 다시 만들거나 LLM을 부르지 않습니다.
5. 실제 문자·메신저·전화 API는 부르지 않습니다.

**반환**

| 키 | 형식 | 규칙 |
|---|---|---|
| `status` | str | `ok`, `already_executed`, `rejected`, `invalid_action`, `error` |
| `message` | str | 개발자용 설명 |
| `action_id` | str \| None | 입력의 `action_id` (없으면 `None`) |
| `applied` | bool | `ok`, `already_executed`일 때만 `True` |
| `mock` | bool | 항상 `True` |
| `record` | dict \| None | 기록한(또는 이미 있던) 한 줄. 기록하지 않았으면 `None` |
| `answer` | str | 화면 문장 (아래 예시) |

화면 문장 예시: `"보호자께 장보기 도움(우유 1개, 계란 1판)을 요청한 것으로 기록했어요. (모의 실행이라 실제 메시지는 보내지 않았어요)"`

## 2.4 승인 데이터 구조 (A·B 공통 봉투)

모든 승인 대기 제안은 아래 공통 키를 가집니다. `type`에 따라 추가 키가 붙습니다.

| 키 | 형식 | 규칙 |
|---|---|---|
| `type` | str | 허용 목록 중 하나 |
| `action_id` | str | 제안마다 새로 만든 고유값 (예: `uuid4().hex`). 중복 실행 방지 열쇠 |
| `summary` | str | 승인 버튼 위에 보여 줄 한 문장 |
| `requires_approval` | bool | 항상 `True` |
| `applied` | bool | 제안 단계에서는 항상 `False` |
| `created_at` | str | KST ISO 시각 |
| `source` | str | `"outing_decision"`(A) 또는 `"living_support"`(B) |
| `payload` | dict | `type`별 내용 (아래) |
| `approval` | dict | **실제 사용자 승인을 확인한 공통 LangGraph만** 붙입니다: `{"status": "approved" \| "rejected", "at": ...}`. 제안 단계에는 이 키가 없어야 하고, LLM·상담 함수가 만들면 안 됩니다. |

- 같은 `action_id`로 여러 번 실행해도 모의 기록은 한 번만 남습니다. (2.3의 3번)

**`type == "request_caregiver_help"` (B의 첫 번째 실행 후보)**

```json
{"type": "request_caregiver_help",
 "action_id": "3f2a9c0e5b7d4e1f8a6b2c9d0e1f2a3b",
 "summary": "보호자께 장보기 도움(우유 1개, 계란 1판)을 요청합니다.",
 "requires_approval": true, "applied": false,
 "created_at": "2026-11-25T15:00+09:00", "source": "living_support",
 "payload": {"recipient_role": "caregiver",
             "category": "groceries",
             "items": [{"name": "우유", "quantity": 1, "unit": "개"},
                       {"name": "계란", "quantity": 1, "unit": "판"}],
             "note": "무거운 것은 문 앞에 두어 달라고 전해 주세요.",
             "urgency": "normal",
             "message": "어르신께서 장보기에 어려움을 겪고 계십니다. 우유 1개와 계란 1판을 구매하는 데 도움을 부탁드립니다."}}
```

`payload` 규칙:

| 항목 | 규칙 |
|---|---|
| `recipient_role` | `"caregiver"` 고정 (연락처를 넣지 않음) |
| `category` | `groceries`, `daily_necessities`, `other` 중 하나 |
| `items` | 1~20개 |
| `items[].name` | 1~40자 |
| `items[].quantity` | int, 1~99 |
| `items[].unit` | 10자 이내 문자열 또는 `""` |
| `note` | 200자 이내, 없으면 `""` |
| `urgency` | `normal`, `soon` 중 하나 |
| `message` | **(이번 추가)** 보호자에게 전할 완성된 요청 문장. 1~300자 |

**`payload.message` 규칙**

- 승인 전에는 **미리보기**입니다.
  - Gradio가 승인·거절 버튼 위에 `summary`와 함께 보여 줍니다.
  - 이때는 어디에도 저장하거나 보내지 않습니다.
- 승인 후에만 `execute_living_action()`이 모의 요청 기록에 저장합니다. 실제로 보내지 않습니다.
- 내용은 `items`·`category`·`note`에 있는 것만 씁니다.
  - `items`에 없는 물품·수량을 넣지 않습니다.
  - 사용자가 말하지 않은 사정(건강 상태 등)을 덧붙이지 않습니다.
- 물품이나 수량이 확인되지 않았으면 제안을 만들지 않습니다.
  - `needs_input`으로 묻습니다. (예: `missing_fields=["quantity"]`, "우유와 계란을 각각 얼마나 사면 될까요?")
  - "1개"처럼 임의로 채우지 않습니다.
- 이름·전화번호·주소·보호자 연락처 같은 개인정보를 넣지 않습니다. 받는 사람은 `recipient_role="caregiver"`로만 나타냅니다.
- LLM이 문장을 다듬어도 됩니다. 다만 Python이 아래를 확인하는 것을 권장합니다.
  - `items`의 이름·수량·단위가 문장에 그대로 있는지
  - `items`에 없는 물품 이름이 없는지
  - 확인이 어렵다면 Python 틀 문장이 가장 안전합니다.

```python
def build_help_message(category: str, items: list[dict], note: str = "") -> str:
    """확인된 물품·수량만으로 보호자에게 전할 요청 문장을 만든다. (개인정보 없음)"""
    # 1. 분류별 상황 문장: 사용자가 말하지 않은 사정을 덧붙이지 않는다.
    need = {"groceries": "장보기", "daily_necessities": "생필품 구매", "other": "생활"}[category]
    # 2. 물품 목록: items에 있는 이름·수량·단위만 그대로 쓴다.
    listed = ", ".join(f"{item['name']} {item['quantity']}{item['unit']}" for item in items)
    text = f"어르신께서 {need}에 어려움을 겪고 계십니다. 필요한 물품은 {listed}입니다. 구매에 도움을 부탁드립니다."
    # 3. 참고 사항: 사용자가 직접 말한 내용만 붙인다.
    return text + (f" 참고: {note}" if note else "")
```

**A의 현재 제안과의 차이**

- A의 `propose_outing_time_change`(1.3)는 지금 `action_id`, `summary`, `created_at`, `source`, `payload`가 없습니다. [구현됨 상태]
- 4단계에서 통합 담당이 LangGraph 래퍼로 보완합니다. **A의 공개 계약은 바꾸지 않습니다.** [제안]

```python
def wrap_outing_proposal(action: dict, now_iso: str) -> dict:
    """A의 일정 변경 제안을 공통 승인 봉투로 감싼다. (A 원본 딕셔너리는 고치지 않음)"""
    # 1. A가 만든 값은 그대로 payload로 옮긴다.
    payload = {key: action[key] for key in ("original_outing_at", "proposed_outing_at", "reason")}
    # 2. 공통 키를 채운다. action_id는 여기서 한 번만 만든다.
    return {"type": action["type"], "action_id": uuid.uuid4().hex,
            "summary": f"외출 시각을 {payload['proposed_outing_at'][11:16]}로 바꾸는 것을 제안합니다.",
            "requires_approval": True, "applied": False, "created_at": now_iso,
            "source": "outing_decision", "payload": payload}
```

- 래퍼는 `a_node`에서 한 번만 실행해 `pending_action`에 저장합니다.
- `approval_node` 안(interrupt 앞)에서 만들면 재개 때 노드가 다시 실행되어 `action_id`가 바뀝니다. 그러면 중복 방지가 깨집니다.

## 2.5 B 담당자가 바로 쓸 수 있는 계약 검사 코드 [제안]

```python
def check_living_result(result: dict) -> None:
    """run_living_support() 반환값이 공통 계약을 지키는지 확인한다. (B 테스트에서 사용)"""
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
            assert item["name"] in payload["message"]      # 확인된 물품만 문장에 들어갔는지
    json.dumps(result, ensure_ascii=False, allow_nan=False)
```

---

# 3. 공통 LangGraph 구현 (4단계) 및 Gradio·5단계 연결

> 이 장은 통합 담당용입니다. **B기능 담당자는 2장만 따르면 되고, 이 장 때문에 B가 바꿀 것은 없습니다.** (3.8에서 확인)

| 절 | 상태 | 코드 |
|---|---|---|
| 3.1~3.6 공통 State·노드·분기·승인·독립 설정·앱 함수 | **[구현됨]** | `safemate_graph.py`, 테스트 `test_safemate_graph.py` (커밋 `4a2338e`) |
| 3.6 중 Gradio 화면 자체 | [제안] (6단계) | 없음. 화면은 3.6의 앱 함수만 부르면 됨 |
| 3.7 5단계 `apply_outing_change()` | [제안] (5단계) | 없음. 4단계는 주입 자리만 있음 |

수업 4장 체크리스트 대응 [구현됨]

- State는 `TypedDict`이고, 누적 기록 두 칸에 reducer(`operator.add`)를 씁니다.
- Node는 바뀐 값만 돌려줍니다.
- `InMemorySaver` + `thread_id`로 대화방을 나눕니다.
- 승인은 `interrupt()` / `Command(resume=...)`로 받고, **부작용은 interrupt가 있는 노드가 아닌 별도 `execute` 노드에만** 둡니다.
- 모든 실행에 `recursion_limit=12`를 줍니다.
- 흐름은 분기 함수(Python)가 정하고, 의도 판단은 A·B 상담 함수 안의 LLM이 맡습니다. 별도 라우터 LLM은 없습니다.
- 확인 환경: LangGraph 1.2.24, langgraph-checkpoint 4.3.2, Python 3.14.6 (`.venv-xpu`)

## 3.1 공통 State [구현됨]

```python
class SafeMateState(TypedDict, total=False):
    user_message: str
    control: Literal["end_consultation"] | None   # 화면의 '상담 끝내기' 버튼
    a_conversation_state: dict | None            # A 전용 (run_outing_decision만 읽고 씀)
    b_conversation_state: dict | None            # B 전용 (run_living_support만 읽고 씀)
    active_feature: Literal["B"] | None          # B가 추가 질문 중이면 "B"
    handoff_context: dict | None                 # A→B로 넘기는 정보 {"source", "user_goal"} (대화 상태 아님)
    feature: Literal["A", "B", "system"] | None  # 이번 턴을 처리한 쪽
    last_result: dict | None                     # A·B 원본 결과 (JSON 사본)
    view: dict | None                            # to_common_view() 결과
    answer: str                                  # 화면에 보여 줄 최종 문장
    pending_action: dict | None                  # 승인을 기다리는 공통 봉투(2.4) 하나
    approval_decision: dict | None               # {"decision", "action_id", "at", "reason"}
    execution_result: dict | None                # 실행 함수 반환값
    action_log: Annotated[list[dict], operator.add]          # 승인·거절·취소·실행·계약 위반 기록
    executed_action_ids: Annotated[list[str], operator.add]  # 성공한 action_id
```

| 규칙 | 내용 |
|---|---|
| 턴마다 비우는 칸 | `intake`가 `control`, `handoff_context`, `feature`, `last_result`, `view`, `answer`, `approval_decision`, `execution_result`를 비웁니다 |
| thread에 남는 칸 | 두 대화 상태, `active_feature`, `action_log`, `executed_action_ids` |
| A·B 상태 분리 | 서로 합치거나 넘기지 않습니다. A가 B로 넘길 때는 `handoff_context`만 넘깁니다 |
| 오류 턴 | A·B가 돌려준 `conversation_state`가 dict이면 **오류 턴이어도 그대로** 저장합니다 (1.3·2.2 오류 턴 규칙). 상담 함수가 예외를 내면 이전 상태를 유지합니다 |
| 저장 값 | JSON으로 저장할 수 있는 값만 넣습니다. 결과는 `json_safe()` 사본으로 저장합니다. 색인·함수 객체는 State에 넣지 않고 `build_graph()`의 클로저로 노드에 전달합니다 |
| 개인정보 | 저장하지 않습니다. `thread_id`는 `uuid4().hex`입니다 |
| 보관 기간 | `InMemorySaver`는 **프로그램이 실행되는 동안만** 보관합니다. 프로그램을 끄면 사라집니다. 영구 저장은 5단계 범위입니다 |

## 3.2 노드와 흐름 [구현됨]

```text
START → intake ─┬─ 입력 오류 / 상담 끝내기 ───────────→ END
                ├─ active_feature == "B" ─────────────→ b_node
                └─ 그 밖 ─────────────────────────────→ a_node
a_node ─┬─ 계약을 지킨 제안 → pending_action ─────────→ approval
        ├─ route == "living_support" ─────────────────→ b_node
        └─ 그 밖 (답변·추가 질문·오류) ────────────────→ END
b_node ─┬─ 계약을 지킨 제안 → pending_action ─────────→ approval
        └─ 그 밖 (B 미연결 안내 포함) ─────────────────→ END
approval: interrupt(승인 카드) ⏸  Command(resume={"decision", "action_id"})
        ├─ approve (현재 action_id) ──→ execute ──→ END
        ├─ reject / cancel (현재 action_id) ──→ close ──→ END
        └─ 형식 오류·오래된 action_id ──→ approval (다시 대기)
```

`python safemate_graph.py`를 실행하면 Mermaid 그림이 출력됩니다. 발표 자료용입니다.

| 노드 | 하는 일 | 부작용 |
|---|---|---|
| `intake` | 이번 턴 칸을 비웁니다. 빈 입력·600자 초과는 안내 후 끝냅니다. '상담 끝내기'를 처리합니다(3.5). 앱 함수를 거치지 않고 승인 대기 중에 새 입력이 들어오면 남은 `pending_action`을 지우고 `cancelled_by_new_input`으로 기록합니다 | 없음 |
| `a_node` | `a_runner(message, device=a_device, index=a_index, conversation_state=a_state)`를 1회 호출합니다. A 상태를 저장합니다. 계약 검사(3.3)를 통과한 제안만 `wrap_outing_proposal()`로 감쌉니다. **`action_id`는 여기서 한 번만 만듭니다.** `living_support`면 `handoff_context`를 만듭니다 | 없음 (A는 읽기 전용) |
| `b_node` | `b_runner`가 없으면 미연결 안내(3.8)를 보여 줍니다. 있으면 `b_runner({"user_message", "context"}, device=b_device, index=b_index, conversation_state=b_state)`를 호출합니다. `status=="needs_input"`이고 `route=="needs_clarification"`이면 `active_feature="B"`, 아니면 `None`입니다. 제안은 검사만 하고 B가 만든 봉투를 사본으로 그대로 씁니다 | 없음 |
| `approval` | **첫 줄에서 `interrupt(approval_card(pending))`**. 그 앞에는 읽기만 있습니다. 재개 값을 `read_decision()`으로 정리해 `approval_decision`에 남깁니다. 무시한 입력은 `ignored_response`로 기록합니다 | 없음 |
| `execute` | 실행 직전에 다시 검사합니다(3.4). 봉투 **사본**에 `approval={"status": "approved", "at": KST}`를 붙여 type별 실행 함수를 부르고, 성공 판정을 합니다 | **여기만 있음** |
| `close` | 실행하지 않습니다. 거절이면 "알겠어요, 그대로 둘게요."를 보여 주고, A 제안이면 "기존 외출 시각을 유지합니다."와 이번 턴에 이미 인용한 공식 안내 한 줄을 덧붙입니다. 새 질문으로 취소되면 기록만 남깁니다 | 없음 |

**조건부 분기 함수** (LLM 없이 State만 봄)

| 함수 | 반환 |
|---|---|
| `after_intake` | `feature=="system"` → `END`, `active_feature=="B"` → `b_node`, 그 밖 → `a_node` |
| `after_feature` (a_node·b_node 공용) | `pending_action` 있음 → `approval`, A 결과 `route=="living_support"` → `b_node`, 그 밖 → `END` |
| `after_approval` | `approve` → `execute`, `ignored`이고 제안이 남아 있음 → `approval`, 그 밖 → `close` |

- `add_conditional_edges`에 대상 노드 목록을 모두 적었습니다.
- B는 A로 되돌아가지 않으므로, 정상 경로는 한 번의 invoke에 노드 4개 이하입니다.

## 3.3 결과 계약 검사 `check_feature_result()` [구현됨]

A·B 결과를 받을 때마다 검사합니다. **문제가 하나라도 있으면 승인 카드를 만들지 않습니다.** 정상 `answer`는 보여 주고, 아니면 안전 문장을 보여 줍니다. `action_log`에 `contract_violation`과 문제 목록을 남깁니다.

| 검사 | 대상 |
|---|---|
| `answer`는 비어 있지 않은 문자열, `conversation_state`는 dict, `status`·`route`는 문자열 | A·B |
| `status=="needs_input"` ⇔ `route=="needs_clarification"` | A·B |
| `requires_user_confirmation`은 bool이고, `proposed_action is not None`과 같음 | A·B |
| 제안이 있으면 `route`는 `error`·`needs_clarification`이 아님. `requires_approval is True`, `applied is False`, `approval` 키 없음 | A·B |
| 허용 표 `ALLOWED_ACTIONS`: `propose_outing_time_change` → source `outing_decision`, A만 / `request_caregiver_help` → source `living_support`, B만. 문자열이 아닌 type은 허용하지 않음 | A·B |
| A: `original_outing_at`·`proposed_outing_at`·`reason`이 문자열. `source`는 없거나 `outing_decision` | A |
| B: `source=="living_support"`, `status=="ok"`, `route=="living_support"`, `action_id`·`summary`·`payload.message`가 비어 있지 않은 문자열 | B (모두 2.2·2.4 규칙과 같음) |

리스트·딕셔너리처럼 잘못된 자료형이 와도 예외 없이 문제로 처리합니다. (G14 테스트)

## 3.4 승인·거절·재개와 중복 실행 방지 [구현됨]

**재개 값**: 화면 버튼만 만듭니다. 형식은 `Command(resume={"decision": "approve"|"reject"|"cancel", "action_id": str})`입니다.

- `cancel`은 앱 함수만 보냅니다. 승인 대기 중에 새 질문이 왔을 때입니다.
- "응", "그래" 같은 자유 문장은 승인으로 보지 않습니다.

| 경우 | 처리 |
|---|---|
| 승인 대기가 아닌데 버튼을 누름 | 앱 함수가 재개하지 않습니다. "지금은 승인할 제안이 없어요." |
| 버튼 값이 approve·reject가 아님 | 재개하지 않습니다. "승인 또는 거절 버튼으로 답해 주세요." |
| 오래된 `action_id` 버튼 | 재개하지 않고 **현재 승인 대기를 그대로 유지**한 채 현재 카드를 다시 보여 줍니다 |
| Command를 직접 넣었는데 형식 오류이거나 id가 다름 | `approval`이 `ignored`로 처리하고 `approval → approval`로 다시 대기합니다 (2차 방어) |
| 승인 대기 중 새 질문 | 앱 함수가 먼저 `cancel`로 재개합니다 → `close`가 `cancelled`를 기록 → 새 질문을 처음부터 처리합니다 |
| 재개 시 노드 재실행 | LangGraph는 interrupt가 있던 `approval` 노드만 다시 실행합니다. 상담 함수 호출과 `action_id` 생성은 이미 끝난 노드에 있으므로 다시 일어나지 않습니다 (G4b) |

**`execute` 실행 직전 재검사 `check_before_execute()`**: 아래 중 하나라도 해당하면 실행하지 않습니다(`execution_blocked`).

- 승인 대기 제안이 없음
- 결정이 `approve`가 아님
- `action_id`가 일치하지 않음
- type·source가 허용 표와 다름
- `requires_approval`·`applied`가 승인 전 값이 아님
- 이미 `approval` 키가 있음

**중복 실행 방지 (3중)**

1. 앱 함수가 승인 대기 여부와 `action_id`를 확인합니다. [구현됨]
2. `execute`가 `executed_action_ids`에 있는 id면 다시 부르지 않습니다(`duplicate_skipped`). [구현됨]
3. 실행 함수 자체의 `action_id` 멱등성 (2.3의 3번, 3.7). **실제 구현에서 따로 검증해야 합니다.** 4단계에서는 가짜 실행 함수로 호출 횟수만 확인했습니다.

**성공 판정 `is_execution_success()`**: 아래를 모두 만족해야 성공입니다.

- 실행 함수가 dict를 돌려줬고 예외가 없음
- `status`가 `"ok"` 또는 `"already_executed"`
- `applied is True`
- 반환 `action_id`가 같음

| 결과 | 화면 문장 | 기록 |
|---|---|---|
| 성공 | 실행 함수의 `answer` (없으면 "승인하신 요청을 처리했어요.") | `executed`, `executed_action_ids`에 추가 |
| 실행 함수 미연결 | "이 요청을 처리하는 기능이 아직 연결되지 않아 바뀐 것이 없어요." | `not_connected` |
| 예외·형식 오류·`applied=False`·id 불일치 | "요청을 처리하지 못했어요. 성공으로 기록하지 않았습니다." | `execution_failed` |
| 재검사 실패 | "승인 정보가 맞지 않아 실행하지 않았어요." | `execution_blocked` |

어느 경우든 실행 뒤에는 `pending_action=None`입니다.

## 3.5 A·B 독립 설정과 B 상담 끝내기 [구현됨]

```python
build_graph(*, a_runner=run_outing_decision, a_device=None, a_index=None,
            b_runner=None, b_device=None, b_index=None,
            a_executor=None, b_executor=None, checkpointer=None)   # checkpointer 기본값: InMemorySaver()
```

| 항목 | A | B | 규칙 |
|---|---|---|---|
| 상담 함수 | `run_outing_decision` (기본값) | `b_runner` (기본값 `None`) | 함수 주입. **B 모듈 이름은 정하지 않습니다** |
| 실행 함수 | `a_executor` (5단계 `apply_outing_change`, 기본값 `None`) | `b_executor` (`execute_living_action`, 기본값 `None`) | type별로 고릅니다 |
| 색인·장치 | `a_index`·`a_device` | `b_index`·`b_device` | 각자 자기 함수에만 전달합니다 (G8c) |
| LLM 모델 | A가 `OLLAMA_MODEL`(기본 `qwen2.5:3b`)을 읽음 | B가 자기 설정을 읽음 | **그래프는 `os.environ`을 읽지도 바꾸지도 않습니다** |
| 대화 상태 | `a_conversation_state` | `b_conversation_state` | 서로 넘기지 않습니다 (G8a) |

- `A_OLLAMA_MODEL`·`B_OLLAMA_MODEL` 변수는 아직 구현하지 않았습니다. A는 여전히 `OLLAMA_MODEL`만 읽습니다(3단계 코드 변경 없음).
- **B 상담 끝내기**: 아래 경우 `b_conversation_state`·`active_feature`만 비우고 A 상태는 그대로 둡니다. 안내 문장은 "생활 지원 상담을 마쳤어요. 새로 궁금한 것을 말씀해 주세요."입니다.
  - 화면 버튼(`end_consultation()`)을 누른 경우
  - B 상담 중에 발화가 `그만`·`취소`·`처음으로`와 정확히 같은 경우 (앞뒤 문장부호 무시)
  - 이때 이 말은 B에 전달되지 않습니다. 이 단어를 포함한 문장(예: "그만 우유도 필요해")은 B로 전달됩니다.
- 자연어로 화제를 바꾸는 것(B가 묻는 중에 날씨를 묻는 경우 등)은 처리하지 않습니다. 이런 말은 B로 전달됩니다. (4.2)

## 3.6 앱 함수와 Gradio 연결

**앱 함수** [구현됨]: 화면은 그래프 내부를 몰라도 이 함수들만 부르면 됩니다.

| 함수 | 하는 일 |
|---|---|
| `new_thread_id()` | 세션마다 새 대화방 id (Gradio `gr.State`에 보관) |
| `send_message(graph, thread_id, text)` | 발화 처리. 승인 대기 중이면 이전 제안을 먼저 취소 |
| `press_button(graph, thread_id, decision, action_id)` | 승인·거절 버튼. 3.4의 1차 검사 후 재개 |
| `end_consultation(graph, thread_id)` | '상담 끝내기' 버튼 |
| `waiting_action(graph, thread_id)` | 승인 대기 중이면 제안, 아니면 `None` |
| `turn_view(graph, thread_id)` | 화면 값 |

모든 앱 함수는 아래 화면 값을 돌려줍니다.

```python
{"answer": str, "notice": str | None,
 "pending": {"action_id", "type", "source", "summary", "preview"} | None,   # 승인 대기 중일 때만
 "active_feature": "B" | None, "evidence": list[dict], "warning": bool,
 "debug": {"feature", "route", "status", "detail_code", "answer_source", "tool_trace", "execution_result", "action_log"}}
```

- `preview`: B는 `payload.message`(보호자 요청 문장 미리보기), A는 `"14:00 → 16:00 / " + payload.reason`입니다.
- `evidence`: `to_common_view()`가 A의 `official_guidance`(키 `text`)와 B의 `living_evidence`(키 `content`)를 같은 모양 `{content, title, publisher, url, pdf_page}`로 맞춥니다.
- `GraphRecursionError`나 예상하지 못한 오류가 나면 예외를 밖으로 던지지 않고 안내 문장을 돌려줍니다.

**Gradio 화면** [제안, 6단계]

- 화면 구성: `gr.Chatbot` + 입력창 + `승인`·`거절` 버튼(`pending`이 있을 때만) + `상담 끝내기` 버튼(`active_feature=="B"`일 때만)
- 버튼 위에는 `summary`와 `preview`를 보여 줍니다.
- 화면 본문에는 `answer`(+`notice`)와 `evidence` 출처를 보여 줍니다. `debug`는 개발자용 접이식 패널에만 둡니다.
- 앱 시작 시 `prepare_safety_index("xpu")`로 A 색인을 한 번 준비해 `build_graph(a_device="xpu", a_index=...)`에 넘깁니다. 그래프 객체 하나를 모든 세션이 함께 씁니다.
- 버튼 연타 같은 동시 요청은 Gradio queue 설정과 함께 6단계에서 검증합니다.

## 3.7 5단계 `apply_outing_change()` 계약 [제안]

4단계 그래프는 `build_graph(a_executor=apply_outing_change)`로 꽂기만 하면 됩니다. 그래프 쪽 수정은 필요 없습니다.

```python
def apply_outing_change(approved_action: dict) -> dict:
    ...
```

**호출 조건** (그래프가 보장, [구현됨])

- 사용자가 화면의 `승인` 버튼을 눌렀고, 그 버튼의 `action_id`가 현재 승인 대기 제안과 같을 때만 `execute` 노드에서 1회 호출합니다.
- 거절·취소·오래된 버튼·형식 오류일 때는 호출하지 않습니다.

**입력**: `wrap_outing_proposal()`이 만든 봉투에 그래프가 `approval`을 붙인 **사본**입니다.

```json
{"type": "propose_outing_time_change",
 "action_id": "32자리 hex",
 "summary": "외출 시각을 16:00로 바꾸는 것을 제안합니다.",
 "requires_approval": true, "applied": false,
 "created_at": "2026-11-25T15:00+09:00", "source": "outing_decision",
 "payload": {"original_outing_at": "2026-11-25T14:00+09:00",
             "proposed_outing_at": "2026-11-25T16:00+09:00",
             "reason": "두 예보의 차이입니다. 한 시간 강수량 2 → 0mm, 강수확률 80 → 10%"},
 "approval": {"status": "approved", "at": "2026-11-25T15:01+09:00"}}
```

**처리 순서** (B의 2.3과 같은 방식)

1. 검증: 하나라도 틀리면 저장하지 않고 `invalid_action`을 돌려줍니다.
   - `type=="propose_outing_time_change"`, `source=="outing_decision"`
   - `requires_approval is True`, `applied is False`, `action_id`가 비어 있지 않은 문자열
   - `payload`의 두 시각이 시간대가 있는 ISO 문자열이고 서로 다름
2. 승인 확인: `approval.status != "approved"`이면 `rejected`를 돌려주고 저장하지 않습니다.
3. 멱등성: 같은 `action_id`가 이미 저장돼 있으면 새로 저장하지 않고 `already_executed`와 기존 기록을 돌려줍니다.
4. 가상 일정 저장
   - 파일 기본 경로 예: `.safemate/virtual_schedule.json` (기록), `.safemate/schedule_change_log.jsonl` (변경 이력). `.safemate/`는 Git에 올리지 않습니다.
   - 기록 항목: `action_id`, `original_outing_at`, `proposed_outing_at`, `reason`, `approved_at`, `executed_at`
   - LLM을 부르지 않습니다. 개인정보를 저장하지 않습니다. 실제 캘린더 API를 부르지 않습니다.
5. 예외를 밖으로 던지지 않고 `status`로 알립니다.

**반환** (B의 `execute_living_action`과 같은 키)

| 키 | 형식 | 규칙 |
|---|---|---|
| `status` | str | `ok`, `already_executed`, `rejected`, `invalid_action`, `error` |
| `message` | str | 개발자용 |
| `action_id` | str \| None | 입력의 `action_id` |
| `applied` | bool | `ok`, `already_executed`일 때만 `True` |
| `mock` | bool | `True` (실제 일정 서비스가 아닌 가상 일정) |
| `record` | dict \| None | 저장한(또는 이미 있던) 기록 |
| `answer` | str | 예: "외출 시각을 16:00로 바꿔 가상 일정에 저장했어요." |

그래프는 `status in ("ok", "already_executed")`, `applied is True`, `action_id`가 같은 경우에만 성공으로 표시합니다(3.4).

**5단계에서 정할 것**

- A는 '기존 일정'을 따로 저장하지 않습니다. `original_outing_at`은 사용자가 말한 시각입니다. 저장된 일정에 같은 시각이 없을 때 새 항목을 만들지, 변경 이력만 남길지 정해야 합니다.
- 프로젝트 기획서의 '저녁: 승인한 일정·요청 내역 확인' 시나리오를 위해 이 기록을 읽는 함수가 필요한지 정해야 합니다.

## 3.8 B기능 담당자 확인 사항 (4단계 구현과 2장 계약 대조)

결론: **2장 계약대로 만들면 그래프 쪽 수정 없이 연결됩니다.** 2장 규격은 이번 개정에서 한 글자도 바꾸지 않았습니다.

| 2장 규칙 | 4단계 그래프 동작 [구현됨] |
|---|---|
| `run_living_support(request, *, device, index, conversation_state)` | 같은 모양으로 호출합니다. `request`는 항상 dict `{"user_message", "context"}`이고, `context`는 A에서 넘어온 턴에만 dict, 그 밖에는 `None`입니다 |
| 필수 키·일관성 규칙(2.2) | 3.3 검사가 같은 규칙을 확인합니다. 어기면 승인 카드를 만들지 않습니다 |
| 오류 턴 상태 규칙 | 돌려준 `conversation_state`(dict)를 그대로 저장합니다 |
| `needs_clarification`이면 B가 다시 질문 | 다음 턴 발화가 A를 거치지 않고 B로 갑니다 (단, 3.5의 '그만'·'취소'·'처음으로'만 단독으로 말하면 B에 전달되지 않음) |
| `approval`은 그래프만 붙임 | 승인 사본에만 붙여 `execute_living_action(action)`을 1회 호출합니다. B가 `approval` 키를 만들면 제안을 거부합니다 |
| 2.3 반환 키 | 성공 판정에 `status`·`applied`·`action_id`·`answer`를 씁니다 |
| `living_evidence` | 화면 `evidence`로 그대로 보여 줍니다 |

B 쪽 참고 (규격 변경 아님)

- 예외를 던져도 그래프는 멈추지 않지만, 2.1의 4번처럼 `status`로 알리는 것이 맞습니다.
- 실행 함수의 `action_id` 멱등성은 그래프에도 중복 방지가 있지만, 2.3의 3번대로 B 테스트에서 따로 확인해야 합니다.

## 3.9 화면 흐름 예시

| 사용자 | 처리 | 화면 | 상태 |
|---|---|---|---|
| "이따 저녁에 밥 먹으러 가려는데 비 와?" (오후 3시) | A: `needs_clarification`, `python_question` | "오늘 저녁 날씨를 확인하려면 어느 지역인지 알려 주세요." | [구현됨] |
| "인천이야" | 그래프가 저장한 A 상태로 이어짐. 18·19·20시 조회 | 답변 + `[확인한 예보]` | [구현됨] (I2·L1) |
| "내일 오후 2시에 인천 가는데 비 오면 시간 바꿀까?" (14시 비 예보) | A 제안 → `a_node`가 봉투 생성 → `approval`에서 멈춤 | 답변 + "외출 시각을 16:00로 바꾸는 것을 제안합니다." + `14:00 → 16:00 / …` + 승인·거절 | [구현됨] (I1, 가상 예보·가상 LLM) |
| [승인] | `execute` → `a_executor` | 5단계 전: "…아직 연결되지 않아 바뀐 것이 없어요." / 5단계 후: 실행 함수 `answer` | 미연결 안내 [구현됨], 저장 [제안] |
| [거절] | `close` | "알겠어요, 그대로 둘게요. 기존 외출 시각을 유지합니다." + 공식 안내 한 줄 | [구현됨] |
| (승인 대기 중) "그냥 오늘 날씨 알려 줘" | 이전 제안 `cancelled` → 새 질문 처리 | 새 답변 + "이전 제안은 실행하지 않고 취소했어요." | [구현됨] |
| "장보기가 힘들어" | A: `route="living_support"` → `b_node` | B 미연결: "생활 지원(장보기 도움 요청) 기능은 아직 준비 중이에요. 지금은 요청이 실행되지 않습니다." / B 연결 후: B의 질문 | 미연결 안내 [구현됨], B [제안] |
| "우유 하나, 계란 한 판" → [승인] | B 제안 → `approval` → `execute_living_action` | 요청 문장 미리보기 → "…기록했어요. (모의 실행)" | Mock B로만 확인 (G7d) |

---

# 4. 검증 상태와 남은 확인 사항

## 4.1 실제로 실행한 테스트 (2026-11-25, 사용자 Windows `.venv-xpu`)

| 범위 | 결과 | 비고 |
|---|---|---|
| 4단계 `test_safemate_graph.py` 기본 실행 | 48개 실행: **45개 통과**, 3개 건너뜀(L층) | M층 42개(G1~G14, 가짜 A·B·실행 함수), I층 3개(실제 `run_outing_decision` + 3단계 테스트의 가상 예보·가상 LLM), R층 1개 |
| 4단계 L층 `--live-ollama --device xpu` | **3개 통과** (118초) | 실제 qwen2.5:3b + 실제 Open-Meteo. 결과는 4.2 |
| 3단계 `test_outing_decision.py` 기본 | 146개 실행, 통과, 건너뜀 3 | 4단계 구현 후 회귀 |
| 3단계 `--device xpu` | 147개 실행, 통과, 건너뜀 2 | MiniLM 임베딩 Intel Arc 140T(XPU) 실행 확인 |
| 1·2·3단계 파일 4개 | 커밋 `c4e2ca0`과 SHA-256 동일 (R1) | 4단계는 이 파일들을 고치지 않음 |
| 타입 검사 | Pyright 1.1.414, `basic` 모드 (VS Code 설정과 같음): 오류 0 | VS Code Pylance 2026.4.1 화면 확인은 사용자 몫 |
| Python 문법 검사 | `py_compile` 통과 | |

중요한 Mock 확인 항목

- interrupt 재개 후에도 `action_id`가 바뀌지 않고 A 함수는 1회만 호출됨 (G4b)
- 오래된 버튼이 현재 제안을 취소하지 않음 (G10)
- 버튼을 두 번 눌러도 실행은 1회 (G5b)
- source·type 불일치 차단 (G11)
- 실행 실패는 성공으로 표시하지 않음 (G12)
- B 상담 끝내기 (G13)
- 잘못된 자료형 입력 방어 (G14)
- A·B 상태·색인 분리 (G8)

## 4.2 실제 모델에서 확인한 한계

| 항목 | 관찰 내용 |
|---|---|
| **일정 변경 의도 분류 (실제 Qwen)** | L2에서 가상 예보(내일 14시 강수 2mm·80%)와 "내일 오후 2시에 인천 가는데 비 오면 시간 바꿀까?"를 두 번 실행했습니다. 두 번 모두 qwen2.5:3b가 의도를 `schedule_change`가 아닌 `weather_question`으로 분류했습니다. 비 예보는 올바르게 설명했지만 **일정 변경 제안이 만들어지지 않았습니다.** 그래프 실행은 정상이었고, 모델의 의사결정이 기대와 달랐습니다. |
| 일정 변경 흐름 검증 범위 | 제안 → 승인·거절 흐름은 **가상 LLM과 가상 예보(I1)에서만** 확인했습니다. 실제 Qwen으로는 아직 이 흐름까지 가지 못했습니다. 시연은 I1 방식이나 모델 분류 개선(3단계 수정, 별도 결정)이 필요합니다. |
| 실제 날씨 (L3) | 내일 14시 인천에 비 예보가 없어 제안이 없었습니다. 이 경우 제안이 없는 것이 맞습니다. |
| Python 대체 답변 | L층 실제 4턴 중 1~2턴이 `python_fallback`이었습니다(두 번 실행). 이때도 답변·A 상태·예보 캐시는 보존되었습니다 (L1, I3). 3단계에 원래 있던 한계입니다. |
| 실제 Open-Meteo 조회 | L1 둘째 턴에서 실제 조회했습니다. |

## 4.3 아직 구현·검증하지 않은 것

| 항목 | 현재 | 확인할 일 |
|---|---|---|
| **B기능** | **코드 없음, 미연결.** 운영 기본값 `b_runner=None`이라 생활 지원 요청에는 준비 중 안내만 나옵니다. B 흐름은 2장 계약을 지키는 가짜 B로만 확인했습니다 | B 구현 후 2.5 계약 검사 + `build_graph(b_runner=…, b_executor=…, b_index=…)`로 연결해 실제 통합 테스트 |
| B 실행 함수 멱등성 | 가짜 함수로 호출 횟수만 확인 | B 테스트에서 같은 `action_id` 두 번 실행 시 기록 1회 |
| 5단계 `apply_outing_change` | 없음. 승인 시 미연결 안내 | 3.7 계약대로 구현 + 멱등성·검증·저장 테스트 |
| Gradio 화면 | 없음 (Gradio 미설치) | 6단계. 설치는 별도 승인 |
| 동시 요청(버튼 연타) | 순차 호출만 확인 | Gradio queue와 함께 검증 |
| A→B 연속 LLM 지연 | A 이해·답변 후 B LLM 호출 (실측 안 함) | B 연결 후 지연 측정, 별도 라우터 필요 여부 판단 |
| 실제 Qwen의 생활 지원 의도 분류 | 실측 안 함 | B 연결 후 확인 |
| B 상담 중 자연어 화제 전환 | 처리하지 않음 (명시적 끝내기만 있음) | B 개발 후 필요하면 재검토 |
| `A_OLLAMA_MODEL`·`B_OLLAMA_MODEL` | 미구현 | B 모델이 정해지면 결정 |
| 체크포인터 | `InMemorySaver`, 프로그램 종료 시 대화 상태 사라짐 | 영구 저장이 필요한 것은 5단계 기록뿐 |
| Ollama 자체의 GPU 실행 | 확인하지 않음 | `outing_decision.py --check`의 `size_vram` |
| 날씨 설명 검사 범위 | 비 여부·하늘 상태·특별·광역시 이름·보장 표현만 검사 | 실측에서 문제가 되면 추가 |
| 비 예보 기준 50% | SafeMate 프로젝트 설정 | 발표 자료에 공식 기준이 아님을 표시 |
| LangGraph 참고 | `Command(resume=None)`은 LangGraph 1.2.24 내부에서 오류가 남 | 버튼은 `None`을 보내지 않으므로 영향 없음 |
