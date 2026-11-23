"""SafeMate A기능 1단계. 한국 시각으로 외출 출발 시각이 속한 정시 구간의 예보를 조회한다."""

import json     # 데이터 분석
import math
import re   # 정규표현식은 표현 처리 -> 문자열에서 특정한 형태의 문자나 숫자를 찾기 위한 규칙
from datetime import datetime, time, timedelta, timezone # 날짜와 시각 / 하루 중 시각 / 시간 간격 / 시간대 설정
from http.client import HTTPException # 예외들을 처리하기 위한 기반 예외 클래스
from urllib import request # SafeMate에서 Open-Meteo 기상 API 서버에 요청을 보내고,
                           # 서버가 반환하는 날씨 데이터를 받아올 때 사용한다.
from urllib.error import HTTPError  # HTTP 요청에서 발생하는 오류를 처리합니다.
from urllib.parse import urlencode  # Python의 딕셔너리를 URL에 들어가는 요청 파라미터 문자열로 변환


KST = timezone(timedelta(hours=9), "KST") # 한국 표준시 설정 // 9시간의 시간 차이
FORECAST_DAYS = 16  # 날씨 예보를 조회할 기간을 나타내는 변수 // 최대 16일 범위의 예보 요청 가능
API_URL = "https://api.open-meteo.com/v1/forecast"
CITY_COORDINATES = {"인천": (37.4563, 126.7052), "서울": (37.5665, 126.9780)} # 도시별 위도와 경도
HOURLY_FIELDS = ( # 조회할 시간별 기상 항목
    "temperature_2m",   # 지상 2m 높이의 기온
    "apparent_temperature", # 사람이 느끼는 체감온도 
    "wind_speed_10m", # 지상 10m 높이의 풍속
    "precipitation", # 	해당 시간 구간의 강수량
    "precipitation_probability", # 	강수 확률
)

# SafeMate 날씨 조회 과정에서 발생하는 오류를 관리하는 사용자 정의 예외 클래스
class WeatherError(Exception): # Exception은 Python에서 이미 만들어 놓은 내장 클래스
    """조회 중단 이유와 사용자에게 보여 줄 안내를 함께 전달한다."""

    # 오류 상태(status)와 안내 메시지(message)를 전달받아 초기화
    def __init__(self, status: str, message: str):

        # 부모 클래스 Exception에 오류 메시지를 전달
        super().__init__(message)
        # 오류 종류를 객체의 status 속성에 저장
        self.status = status

# 외출 시간을 해석하고 변환하는 함수
# 사용자가 입력한 외출 날짜와 시간을 검사한뒤, 파이썬이 처리 할 수있는 한국 표준시로(KST)의 날짜, 시간, 데이터로 변환하는 함수
def parse_outing_time(  # 1. 외출 날짜, 외출 시간, 현재 시간을 전달받는 함수 생성
    outing_date: str | None, outing_time: str | None, now: datetime
) -> datetime: # 정상 처리되면 날짜, 시간 객체를 반환.
    """오늘·내일·모레 또는 달력 날짜와 24시간제 시각을 한국 시각으로 바꾼다."""


    # 2. 날짜와 시간이 제대로 입력되었는지 검사.
    if not isinstance(outing_date, str) or not outing_date.strip():
        raise WeatherError("needs_input", "외출 날짜를 입력해 주세요. 예) 내일 또는 YYYY-MM-DD")
    
    if not isinstance(outing_time, str) or not outing_time.strip():
        raise WeatherError("needs_input", "외출 시간을 24시간제로 입력해 주세요. 예) 14:30")


    # 3. 날짜와 시간 변환
    day_text = outing_date.strip()  # 불필요한 공백을 제거 -> '오늘', '내일', '모레'를 실제 날짜로 바꿀 준비를 함
    clock_text = outing_time.strip()
    offsets = {"오늘": 0, "내일": 1, "모레": 2}

    if day_text in offsets:
        day_text = (now.date() + timedelta(days=offsets[day_text])).isoformat()
    if not re.fullmatch(r"[0-9]{1,2}:[0-9]{2}", clock_text):    # 시간이 14:30 이와 같은 형식인지 확인
        raise WeatherError("needs_input", "시와 분을 함께 입력해 주세요. 예) 오후 2시 반은 14:30")
    try:
        local_time = datetime.strptime(f"{day_text} {clock_text}", "%Y-%m-%d %H:%M") # 문자열을 날짜·시간 객체로 변환
        outing = local_time.replace(tzinfo=KST) # 한국 표준시 설정
    except ValueError:
        raise WeatherError("needs_input", "달력 날짜와 시간을 확인해 주세요. 예) 내일, 14:30") from None


    # 4. 외출 시간이 유효한지 확인 // 과거 시간 or 16일 예보 범위를 벗어났는지 확인
    if outing < now.replace(second=0, microsecond=0): # 이미 지난 외출 시간인지 확인합니다.
        raise WeatherError("out_of_range", "이미 지난 외출 시각입니다. 현재 또는 이후 시각을 입력해 주세요.")
    
    if outing.date() >= now.date() + timedelta(days=FORECAST_DAYS): # 설정된 예보 조회 기간인 16일의 범위를 벗어나는 날짜인지 확인
        last_day = now.date() + timedelta(days=FORECAST_DAYS - 1)
        raise WeatherError("out_of_range", f"한국 시간 기준 최대 {last_day}까지 조회할 수 있습니다.")

    # 5. 최종 결과 반환
    return outing


def departure_hour_window(outing: datetime) -> tuple[datetime, datetime]:
    """출발 시각이 속한 정시 구간 [시작, 끝)을 구한다. 예) 14:30 → 14:00~15:00."""
    start = outing.replace(minute=0, second=0, microsecond=0)
    return start, start + timedelta(hours=1)


def api_hour(moment: datetime) -> str:
    """API에서 Asia/Seoul을 별도로 지정하므로 현지 시각 문자열만 사용한다."""
    return moment.strftime("%Y-%m-%dT%H:%M")


def fetch_hourly(city: str, start: datetime, end: datetime) -> dict:
    """필요한 시각과 항목만 지정해 Open-Meteo를 한 번 호출한다."""
    latitude, longitude = CITY_COORDINATES[city]
    # 시작·끝 시각으로 조회할 때 forecast_days를 함께 보내면 HTTP 400이 발생한다.
    query = urlencode({
        "latitude": latitude, "longitude": longitude, "timezone": "Asia/Seoul",
        "start_hour": api_hour(start), "end_hour": api_hour(end),
        "hourly": ",".join(HOURLY_FIELDS),
        "temperature_unit": "celsius", "wind_speed_unit": "ms", "precipitation_unit": "mm",
    })
    try:
        with request.urlopen(f"{API_URL}?{query}", timeout=8) as response:
            payload = json.load(response)
    except HTTPError as error:
        error.close()
        if error.code == 429:
            message = "호출 한도에 도달했습니다(HTTP 429). 잠시 후 다시 시도해 주세요."
        else:
            message = f"기상 API가 요청을 처리하지 못했습니다(HTTP {error.code})."
        raise WeatherError("api_error", message) from error
    except (OSError, HTTPException) as error:
        message = "기상 API에 연결하지 못했습니다. 인터넷 연결을 확인한 뒤 다시 시도해 주세요."
        raise WeatherError("network_error", message) from error
    except ValueError as error:
        raise WeatherError("data_unavailable", "기상 API의 응답을 JSON으로 읽을 수 없습니다.") from error

    if not isinstance(payload, dict):
        raise WeatherError("data_unavailable", "기상 API 응답 형식을 확인할 수 없습니다.")
    if payload.get("error"):
        raise WeatherError("api_error", "기상 API가 요청을 거절했습니다.")
    return payload


def hourly_value(
    hourly: dict, units: dict, field: str, index: int | None, unit: str,
    minimum: float = -math.inf, maximum: float = math.inf,
) -> float | None:
    """시각·단위·숫자가 확인되는 값만 읽는다. 0은 유효하며 누락은 None이다."""
    values = hourly.get(field)
    if index is None or units.get(field) != unit or not isinstance(values, list):
        return None
    if len(values) != len(hourly["time"]):
        return None
    value = values[index]
    if type(value) not in (int, float) or not math.isfinite(value):
        return None
    return value if minimum <= value <= maximum else None


def select_weather(payload: dict, start: datetime, end: datetime) -> dict:
    """기온·풍속은 구간의 시작 행, 직전 1시간 누적 강수는 끝 행에서 읽는다."""
    hourly = payload.get("hourly")
    units = payload.get("hourly_units")
    if payload.get("utc_offset_seconds") != 9 * 3600:
        raise WeatherError("data_unavailable", "응답이 한국 시각 기준인지 확인할 수 없습니다.")
    if not isinstance(hourly, dict) or not isinstance(units, dict):
        raise WeatherError("data_unavailable", "시간별 예보 또는 단위 정보가 없습니다.")
    times = hourly.get("time")
    if not isinstance(times, list) or not all(isinstance(stamp, str) for stamp in times):
        raise WeatherError("data_unavailable", "예보 시각을 식별할 수 없습니다.")
    indices = {timestamp: index for index, timestamp in enumerate(times)}
    if len(indices) != len(times):
        raise WeatherError("data_unavailable", "중복된 예보 시각이 있어 구간을 식별할 수 없습니다.")
    start_index = indices.get(api_hour(start))
    end_index = indices.get(api_hour(end))
    readings = {
        "temperature_c": hourly_value(hourly, units, "temperature_2m", start_index, "°C"),
        "apparent_temperature_c": hourly_value(hourly, units, "apparent_temperature", start_index, "°C"),
        "wind_speed_mps": hourly_value(hourly, units, "wind_speed_10m", start_index, "m/s", 0),
        "precipitation_mm": hourly_value(hourly, units, "precipitation", end_index, "mm", 0),
        "precipitation_probability_pct": hourly_value(
            hourly, units, "precipitation_probability", end_index, "%", 0, 100
        ),
    }
    if all(value is None for value in readings.values()):
        raise WeatherError("data_unavailable", "해당 시간대에 사용할 수 있는 예보 값이 없습니다.")
    return readings


def get_outing_weather(
    outing_date: str | None = None, outing_time: str | None = None, city: str | None = None
) -> dict:
    """공개 진입점. 항상 status, message, weather를 담은 딕셔너리를 반환한다."""
    now = datetime.now(KST)
    try:
        outing = parse_outing_time(outing_date, outing_time, now)
        if not isinstance(city, str) or city.strip() not in CITY_COORDINATES:
            choices = ", ".join(CITY_COORDINATES)
            raise WeatherError("needs_input", f"외출 지역을 선택해 주세요. 가능한 지역은 {choices}입니다.")
        city = city.strip()
        start, end = departure_hour_window(outing)
        # 마지막 예보일 23시대에는 다음 날 00시 누적값을 요청할 수 없다.
        last_day = now.date() + timedelta(days=FORECAST_DAYS - 1)
        last_hour = datetime.combine(last_day, time(23), tzinfo=KST)
        payload = fetch_hourly(city, start, min(end, last_hour))
        readings = select_weather(payload, start, end)
    except WeatherError as error:
        return {"status": error.status, "message": str(error), "weather": None}

    missing = [field for field, value in readings.items() if value is None]
    weather = {
        "data_kind": "forecast",
        "source": "Open-Meteo (CC BY 4.0) · https://open-meteo.com/",
        "city": city,
        "outing_at": outing.isoformat(timespec="minutes"),
        "hour_start": start.isoformat(timespec="minutes"),
        "hour_end": end.isoformat(timespec="minutes"),
        **readings,
        "missing_fields": missing,
    }
    if missing:
        message = "일부 예보만 확인했습니다. 누락 항목은 missing_fields에 표시했습니다."
        return {"status": "partial", "message": message, "weather": weather}
    return {"status": "ok", "message": "외출 시간대의 예보를 조회했습니다.", "weather": weather}


def main() -> None:
    """터미널에서 직접 실행할 때만 입력을 받고 결과를 표시한다."""
    city = input(f"외출 지역 ({' / '.join(CITY_COORDINATES)}): ")
    outing_date = input("외출 날짜 (오늘 / 내일 / 모레 / YYYY-MM-DD): ")
    outing_time = input("외출 시간 (24시간제 HH:MM): ")
    result = get_outing_weather(outing_date, outing_time, city)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
