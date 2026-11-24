"""SafeMate A기능 1단계. 한국 시각으로 외출 출발 시각이 속한 정시 구간의 예보를 조회한다."""

import json     # 데이터 분석
import math
import re   # 정규표현식은 표현 처리 -> 문자열에서 특정한 형태의 문자나 숫자를 찾기 위한 규칙
from datetime import datetime, time, timedelta, timezone # 날짜와 시각 / 하루 중 시각 / 시간 간격 / 시간대 설정
from functools import lru_cache # 같은 지역을 반복 조회하지 않도록 지역 검색 결과 캐싱
from http.client import HTTPException # 예외들을 처리하기 위한 기반 예외 클래스
from urllib import request # SafeMate에서 Open-Meteo 기상 API 서버에 요청을 보내고,
                           # 서버가 반환하는 날씨 데이터를 받아올 때 사용한다.
from urllib.error import HTTPError  # HTTP 요청에서 발생하는 오류를 처리합니다.
from urllib.parse import urlencode  # Python의 딕셔너리를 URL에 들어가는 요청 파라미터 문자열로 변환


KST = timezone(timedelta(hours=9), "KST") # 한국 표준시 설정 // 9시간의 시간 차이
FORECAST_DAYS = 16  # 날씨 예보를 조회할 기간을 나타내는 변수 // 최대 16일 범위의 예보 요청 가능
API_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search" # 지명을 위도·경도로 변환하는 API
CITY_COORDINATES = {"인천": (37.4563, 126.7052), "서울": (37.5665, 126.9780)} # 도시별 위도와 경도
# 기존 두 도시는 이전과 같은 좌표를 사용해 1단계 동작을 유지한다.
# 일부 한국어 지명이 검색되지 않을 때는 표준 영문 지명으로 한 번 더 시도한다. (좌표를 임의로 만들지 않음)
CITY_SEARCH_ALIASES = {"부산": "Busan", "대구": "Daegu", "대전": "Daejeon", "광주": "Gwangju", "수원": "Suwon", "안산": "Ansan"}

# (추가) 시·도를 함께 적지 않으면 어느 지역인지 정할 수 없는 지명
# 예: '광주'만 입력하면 광주광역시인지 경기도 광주시인지 알 수 없다.
# 이전 실제 API 확인에서 '광주' 단독 검색이 충청남도의 다른 좌표(36.9122, 127.1279)를 돌려준 적이 있으므로
# 이런 이름은 API 결과를 고르지 않고 처음부터 사용자에게 다시 물어본다.
# 키는 _city_name_key()로 정리한 비교용 이름이고, 값은 사용자에게 보여 줄 입력 예시이다.
AMBIGUOUS_CITY_NAMES = {
    "광주": "'광주광역시' 또는 '광주, 경기도'",
    "gwangju": "'광주광역시' 또는 '광주, 경기도'",
}

# (추가) 도시 후보로 인정할 GeoNames 지역 분류 코드
# - CITY_FEATURE_CODES: 이름만 맞으면 바로 시·군급 도시로 인정하는 코드
#   PPLC(수도), PPLA(시·도청 소재지), PPLA2(시·군청 소재지), ADM1(시·도), ADM2(시·군·구)
# - POPULATED_PLACE_CODES: 사람이 사는 장소이지만 마을·읍·면·동일 수도 있어서
#   인구 조건(1만 명 이상)을 함께 만족할 때만 도시로 인정하는 코드
# - 위 두 목록에 없는 코드(MT=산, HLL=언덕, ISL=섬, STM=하천 등)는 인구와 관계없이 도시로 인정하지 않는다.
CITY_FEATURE_CODES = {"PPLC", "PPLA", "PPLA2", "ADM1", "ADM2"}
POPULATED_PLACE_CODES = {"PPL", "PPLA3", "PPLA4"}
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




# 전국 지역 조회를 위한 지명 → 좌표 변환 함수들 (기존 날씨 조회와 독립적으로 동작)
# (주의) _city_name_key()는 '도시 이름'만 비교하므로 '광주광역시'와 '광주시'가 모두 '광주'가 된다.
# 그래서 이름만으로는 광주광역시와 경기도 광주시를 구분할 수 없다.
# 두 광주의 구분은 resolve_city_coordinates()에서 시·도(admin1)를 함께 확인하여 처리한다.
# (이 함수를 바꾸면 '대전'과 '대전광역시'를 같은 도시로 보는 기존 동작이 깨지므로 그대로 둔다.)
def _city_name_key(text: str) -> str:
    """'대전'과 '대전광역시'처럼 같은 지명을 비교하기 위한 간단한 표준화."""
    name = re.sub(r"[\s-]+", "", text).casefold()
    for ending in ("특별자치시", "특별시", "광역시", "자치시", "시", "군"):
        # (수정) '군'을 추가했다. 예: '양평'과 '양평군', '고성'과 '고성군'을 같은 이름으로 비교
        # 이제 이름이 정확히 같아야만 후보로 인정하므로, '군'을 빼면 군 지역을 찾지 못하게 된다.
        # ('구'는 추가하지 않는다. '대구'가 '대'로 잘리는 문제가 생기기 때문이다.)
        if name.endswith(ending) and len(name) > len(ending):
            return name[:-len(ending)]
    return name







# 사용자가 입력한 시·도 이름과 API의 시·도 이름을 같은 형태로 비교하는 함수
# 예: "경기도", "경기", "Gyeonggi-do" → 모두 "gyeonggi"로 통일
# _province_name_key(): 시·도 이름을 비교할 때만 사용하며 기존 날씨 조회에 영향을 주지 않음
def _province_name_key(name: str) -> str:
    """시·도의 한국어·영어 표기를 통일한다. 등록되지 않은 표기는 기호만 정리하여 그대로 비교한다."""
    # (수정) 원래 설명은 "경기도의 한국어·영어 표기를 통일하고 다른 시·도는 기호를 정리한다."였으나
    #        17개 시·도의 표기를 등록하면서 설명을 바꿨다.

    # 1. 이름의 공백·하이픈 등을 제거하고 영문 대소문자를 통일
    key = re.sub(r"[\s._-]+", "", name).casefold()

    # 2. 경기도의 한국어·영어 표기를 같은 값으로 변환
    # API가 'Gyeonggi-do'로 반환해도 사용자가 입력한 '경기도'와 일치시킴
    aliases = {
        "경기": "gyeonggi",
        "경기도": "gyeonggi",
        "gyeonggi": "gyeonggi",
        "gyeonggido": "gyeonggi",
        "gyeonggiprovince": "gyeonggi",
    }

    # 2-1. (추가) 다른 시·도의 한국어·영어 표기도 같은 값으로 변환
    # 등록한 표기는 행정안전부의 공식 시·도 명칭, 흔히 쓰는 줄임말(경남·충북 등),
    # 국어의 로마자 표기법에 따른 영문 명칭(Gyeongsangnam-do 등)이다.
    # 실제 Open-Meteo 응답 표기로 직접 확인한 것은 '경기도'뿐이므로,
    # 여기에 없는 표기는 '같은 시·도'라고 추측하지 않고 정리한 이름 그대로 비교한다.
    # (예: '경상도'는 경상남도·경상북도 어느 것과도 같다고 보지 않음)
    # ※ '광주시'는 경기도 광주시와 헷갈리므로 광주광역시의 별칭으로 등록하지 않는다.
    province_forms = {
        "seoul": ("서울", "서울시", "서울특별시", "seoul"),
        "busan": ("부산", "부산시", "부산광역시", "busan"),
        "daegu": ("대구", "대구시", "대구광역시", "daegu"),
        "incheon": ("인천", "인천시", "인천광역시", "incheon"),
        "gwangju": ("광주", "광주광역시", "gwangju"),
        "daejeon": ("대전", "대전시", "대전광역시", "daejeon"),
        "ulsan": ("울산", "울산시", "울산광역시", "ulsan"),
        "sejong": ("세종", "세종시", "세종특별자치시", "sejong"),
        "gangwon": ("강원", "강원도", "강원특별자치도", "gangwon", "gangwondo", "gangwonstate"),
        "chungbuk": ("충북", "충청북도", "chungcheongbukdo", "northchungcheong"),
        "chungnam": ("충남", "충청남도", "chungcheongnamdo", "southchungcheong"),
        "jeonbuk": ("전북", "전라북도", "전북특별자치도", "jeollabukdo", "northjeolla", "jeonbukstate"),
        "jeonnam": ("전남", "전라남도", "jeollanamdo", "southjeolla"),
        "gyeongbuk": ("경북", "경상북도", "gyeongsangbukdo", "northgyeongsang"),
        "gyeongnam": ("경남", "경상남도", "gyeongsangnamdo", "southgyeongsang"),
        "jeju": ("제주", "제주도", "제주특별자치도", "jeju", "jejudo"),
    }
    for standard, forms in province_forms.items():
        for form in forms:
            aliases[form] = standard
            # 예: aliases["경상남도"] = "gyeongnam", aliases["gyeongsangnamdo"] = "gyeongnam"

    # 3. 등록된 별칭이 있으면 통일된 이름을, 없으면 원래 정리한 이름을 반환
    return aliases.get(key, key)




# 사용자가 입력한 지역 이름을 검색하여 한국 내 지역 정보(위도, 경도 등)를 가져오는 함수
def _fetch_geocoding_results(name: str) -> list[dict]:
    """한국(KR)으로 한정하여 Open-Meteo 지역 검색 결과를 불러온다."""
    # name: 검색할 지역 이름 / list[dict]: 딕셔너리들이 담긴 리스트를 반환
    # 함수 이름 앞의 _는 내부에서 사용하는 함수라는 관례적 표시

    # 1. 지역 검색 API에 전달할 요청 정보 설정
    query = urlencode({
        "name": name,          # 검색할 지역 이름
        "count": 20,           # 최대 20개의 검색 결과 요청
        "language": "ko",      # 검색 결과를 한국어로 요청
        "countryCode": "KR",   # 대한민국 지역으로 검색 범위 제한
        "format": "json",      # 서버 응답을 JSON 형식으로 요청
    })
    # urlencode(): 딕셔너리를 URL에 붙일 수 있는 요청 문자열로 변환


    # 2. Open-Meteo 지역 검색 API에 요청을 보내고 결과 가져오기
    try:
        with request.urlopen(f"{GEOCODING_API_URL}?{query}", timeout=8) as response:
            # urlopen(): API 서버에 요청 / timeout=8: 최대 8초 대기
            payload = json.load(response)
            # json.load(): 서버의 JSON 응답을 Python 데이터로 변환


    # 3. API 요청 과정에서 발생하는 오류 처리
    except HTTPError as error:
        error.close()

        if error.code == 429:
            # HTTP 429: API 요청 횟수가 허용 한도를 초과한 경우
            message = "지역 검색 API 호출 한도에 도달했습니다(HTTP 429). 잠시 뒤 재시도해 주세요."
        else:
            # 그 외 HTTP 오류가 발생한 경우
            message = f"지역 검색 API 요청이 실패했습니다(HTTP {error.code})."
        raise WeatherError("api_error", message) from error
        # WeatherError 클래스를 사용하여 API 오류 전달

    except (OSError, HTTPException) as error:
        # 인터넷 연결 문제 또는 서버 통신 오류 처리
        raise WeatherError("network_error", "지역 검색 API에 연결하지 못했습니다. 인터넷을 확인해 주세요.") from error
    except ValueError as error:
        # 서버에서 받은 응답을 JSON으로 변환하지 못한 경우
        raise WeatherError("data_unavailable", "지역 검색 결과를 JSON으로 읽을 수 없습니다.") from error


    # 4. 서버에서 받아온 검색 결과의 형식 검사
    if not isinstance(payload, dict):
        # 응답 데이터가 딕셔너리 형식이 아니면 오류 발생
        raise WeatherError("data_unavailable", "지역 검색 API의 응답 형식이 올바르지 않습니다.")

    if payload.get("error"):
        # API 서버가 오류를 응답한 경우
        raise WeatherError("api_error", "지역 검색 API가 요청을 거절했습니다.")
    results = payload.get("results", [])
    # 검색 결과 목록 추출 / 결과가 없으면 빈 리스트([]) 사용

    if not isinstance(results, list):
        # 검색 결과가 리스트 형식이 아니면 오류 발생
        raise WeatherError("data_unavailable", "지역 검색 결과 목록의 형식이 올바르지 않습니다.")


    # 5. 정상적으로 조회한 지역 검색 결과 반환
    return results







# 검색된 지역이 시·군급 도시 후보인지 판별하는 함수
def _is_city_level_place(place: dict) -> bool:
    """검색 결과가 산·마을 이름이 아니라 시·군급 도시 후보인지 판별한다."""
    # place: 검색된 지역 정보가 담긴 딕셔너리
    # -> bool: 조건에 맞으면 True, 아니면 False 반환


    # 1. 입력된 지역 데이터의 형식 검사
    if not isinstance(place, dict):
        # 지역 정보가 딕셔너리 형식이 아니면 False 반환
        return False


    # 2. 지역의 분류 코드와 인구 정보 가져오기
    feature = str(place.get("feature_code", "")).upper()
    # feature_code: 지역의 종류를 나타내는 GeoNames 분류 코드
    # upper(): 영문 코드를 대문자로 통일

    population = place.get("population")
    # population: 해당 지역의 인구 정보


    # 3. 지역 분류 코드를 이용해 도시 후보인지 판별
    # GeoNames의 거주지 및 행정구역 분류 코드를 이용
    is_city = feature in CITY_FEATURE_CODES
    # (수정) 원래 코드는 아래 조건이었다.
    #     feature.startswith("PPLA")  # 행정 중심지 관련 코드
    #     or feature == "PPLC"         # 국가의 수도
    #     or feature in {"ADM1", "ADM2"}  # 1·2단계 행정구역
    # 그런데 startswith("PPLA")는 PPLA3·PPLA4(읍·면·동급 중심지)까지 모두 도시로 인정한다.
    # 그래서 위에서 만든 CITY_FEATURE_CODES(PPLC·PPLA·PPLA2·ADM1·ADM2)에 들어 있는 코드만 바로 인정한다.
    # 조건 중 하나라도 만족하면 is_city = True


    # 4. 인구 규모를 기준으로 추가 판별
    # 인구 데이터가 숫자인 경우에만 비교
    # (수정) 인구 조건은 '사람이 사는 장소(PPL·PPLA3·PPLA4)'에만 적용한다.
    # 원래는 산(MT)이라도 인구 값이 1만 이상으로 들어 있으면 도시로 통과할 수 있었다.
    # ※ 1만 명 기준은 SafeMate가 정한 보조 판단 기준이며, 한국의 법정 행정구역 기준이 아니다.
    if (feature in POPULATED_PLACE_CODES
            and isinstance(population, (int, float)) and not isinstance(population, bool)):
        # 인구가 1만 명 이상이면 도시 후보로 인정
        if population >= 10000:
            return True


    # 5. 최종 판별 결과 반환
    # 인구 조건을 만족하지 않았다면 지역 분류 코드의 판별 결과 반환
    return is_city





# (추가) API에서 받은 값이 정상적인 숫자인지 검사하는 함수 (좌표와 날씨 값에 함께 사용)
def _safe_number(value, minimum: float = -math.inf, maximum: float = math.inf):
    """정상적인 int·float이고 범위 안에 있으면 원래 값을, 아니면 None을 반환한다."""
    # value: 검사할 값 / minimum, maximum: 허용 범위 (기본값은 범위 제한 없음)

    # 1. 자료형 검사
    # type()으로 비교하면 True·False(bool)는 숫자로 취급하지 않는다.
    # (Python에서 bool은 int의 하위 클래스라서 isinstance(True, int)는 True가 되기 때문)
    if type(value) not in (int, float):
        return None

    # 2. 실수(float)로 바꿀 수 있는 크기인지 검사
    # 10**400처럼 너무 큰 정수는 float()나 math.isfinite()에서 OverflowError가 발생한다.
    # 정상적인 좌표나 날씨 값이 아니므로 프로그램을 멈추지 않고 None으로 처리한다.
    try:
        number = float(value)
    except OverflowError:
        return None

    # 3. NaN(숫자가 아님)·무한대이거나 허용 범위를 벗어나면 None
    if not math.isfinite(number) or not (minimum <= number <= maximum):
        return None

    # 4. 정상 값은 원래 자료형 그대로 반환 (예: API가 0을 정수로 주면 0 그대로)
    return value




# (추가) 검색된 지역의 위도·경도를 꺼내고 정상 좌표인지 검사하는 함수
# 원래 resolve_city_coordinates()의 6단계 안에 있던 좌표 검사를 함수로 옮겼다.
def _place_coordinates(place: dict) -> tuple[float, float] | None:
    """정상적인 위도·경도면 (위도, 경도)를, 아니면 None을 반환한다."""

    # API 응답에서 위도와 경도 추출
    latitude_raw = place.get("latitude")
    longitude_raw = place.get("longitude")

    # 위도가 정상적인 숫자인지 검사
    # 좌표가 정상적인 숫자이며 허용 범위 안에 있는지 검사
    latitude = _safe_number(latitude_raw, -90, 90)

    # 경도가 정상적인 숫자인지 검사
    longitude = _safe_number(longitude_raw, -180, 180)

    if latitude is None or longitude is None:
        # 위도·경도가 없거나, 문자열이거나, NaN·무한대·너무 큰 수·범위 밖이면 사용하지 않음
        return None

    # 위도와 경도를 실수(float)로 변환
    return float(latitude), float(longitude)




# (추가) 검색된 지역이 속한 시·도를 비교용 이름으로 돌려주는 함수
def _place_province_key(place: dict) -> str:
    """검색 결과의 admin1(시·도)을 _province_name_key()로 정리한다."""
    province = place.get("admin1")
    if not province and str(place.get("feature_code", "")).upper() == "ADM1":
        # 광역시 자체(ADM1)는 상위 행정구역 칸이 비어 있을 수 있어서 자기 이름을 시·도로 본다.
        province = place.get("name")
    return _province_name_key(str(province or ""))
    # admin1이 없으면 빈 문자열("")이 되어 어떤 시·도와도 일치하지 않는다.




# (추가) 검색된 지역 하나가 사용자가 요청한 지역과 맞는지 판별하는 함수
# 최초 검색 결과 검사, 재검색 여부 판단, 최종 좌표 선택에서 모두 이 함수 하나만 사용한다.
# 단계마다 서로 다른 기준을 쓰면 '재검색은 하지 않았는데 최종 선택에서는 탈락'하는 식의 오류가 생기기 때문이다.
def _is_matching_place(place: dict, valid_names: set[str], requested_province: str | None) -> bool:
    """국가·지역명·시·도·지역 종류·좌표를 모두 만족하는 후보만 True."""
    # place: 검색된 지역 정보 / valid_names: 같은 도시로 인정할 이름들
    # requested_province: 사용자가 정한 시·도 (정하지 않았으면 None)

    # 1. 한국 지역인지 확인
    # 검색할 때 countryCode=KR을 보내지만, 응답도 한 번 더 확인한다.
    if not isinstance(place, dict) or place.get("country_code") != "KR":
        return False

    # 2. 이름이 사용자가 입력한 도시와 같은지 확인
    # 왜 이름을 검증하는가? → 검색 API는 비슷한 이름이나 다른 장소도 함께 돌려줄 수 있다.
    # '검색 결과가 하나뿐이니 맞을 것이다'라고 가정하지 않고 이름이 같은 후보만 남긴다.
    if _city_name_key(str(place.get("name", ""))) not in valid_names:
        return False

    # 3. 사용자가 시·도를 정했다면 그 시·도에 속하는지 확인
    # 왜 시·도를 확인하는가? → 광주광역시와 경기도 광주시처럼 이름이 같은 도시가 있기 때문이다.
    # 시·도가 다르거나 API에 행정구역 정보가 없는 후보는 안전하게 제외한다.
    if requested_province is not None and _place_province_key(place) != requested_province:
        return False

    # 4. 산·마을이 아니라 시·군급 도시인지 확인
    # 왜 도시와 산을 구분하는가? → '안산'처럼 산(MT) 이름과 도시 이름이 같으면
    # 산의 좌표로 날씨를 조회하게 되어 실제 생활 지역의 날씨와 달라질 수 있다.
    if not _is_city_level_place(place):
        # 앞서 만든 _is_city_level_place() 함수를 이용해 도시 후보 판별
        return False

    # 5. 좌표가 정상인지 확인 (좌표가 없으면 날씨를 조회할 수 없음)
    return _place_coordinates(place) is not None




# 동일한 지역을 다시 검색할 때 이전 결과를 재사용하는 캐시 설정
@lru_cache(maxsize=128)
# 최대 128개의 서로 다른 입력에 대한 결과를 저장하여 중복 API 호출을 줄임

# 사용자가 입력한 지역명을 검색하여 해당 지역의 위도와 경도를 반환하는 함수
def resolve_city_coordinates(city: str) -> tuple[float, float]:
    """지역명으로 좌표를 가져온다. 동명이인 지역은 임의 선택하지 않고 재입력을 요청한다."""
    # city: 사용자가 입력한 지역명
    # tuple[float, float]: 위도와 경도를 실수형 튜플로 반환


    # 1. 사용자가 입력한 지역명의 유효성 검사
    if not isinstance(city, str) or not city.strip():
        raise WeatherError("needs_input", "외출 지역을 입력해 주세요. 예) 대전, 수원, 광주광역시 또는 광주, 경기도")
        # (수정) 예시의 '광주'를 '광주광역시'로 바꿨다. 이제 '광주'만 입력하면 다시 묻기 때문이다.

    city = city.strip()
    # 지역명 앞뒤의 불필요한 공백 제거


    # 2. 기존에 등록된 지역인지 먼저 확인
    if city in CITY_COORDINATES:
        # 서울·인천처럼 기존에 좌표가 등록된 지역은 API 검색 없이 바로 반환
        return CITY_COORDINATES[city]


    # 3. 지역명 입력 형식 검사
    parts = [part.strip() for part in city.split(",")]
    # split(","): 쉼표를 기준으로 지역명을 분리
    # 예: "광주, 경기도" → ["광주", "경기도"]

    if len(parts) > 2 or any(not part for part in parts) or len(parts[0]) < 2:
        # 지역명이 너무 짧거나, 쉼표 구분이 잘못된 경우 오류 발생
        raise WeatherError("needs_input", "지역명을 확인해 주세요. 예) 수원 또는 광주, 경기도")


    # 3-1. (추가) 검색 결과와 비교할 기준 만들기
    # 사용자가 입력한 지역명과 등록된 별칭을 비교 대상으로 설정
    requested_city = _city_name_key(parts[0])
    # _city_name_key(): 지역명을 비교하기 위한 형태로 정리하는 별도 함수

    english_alias = CITY_SEARCH_ALIASES.get(parts[0]) or CITY_SEARCH_ALIASES.get(requested_city, parts[0])
    # '대전광역시'처럼 접미사가 붙어 있어도 '대전'의 영어 별칭(Daejeon)을 찾을 수 있게 정리한 이름으로도 찾는다.

    valid_names = {
        requested_city,
        _city_name_key(english_alias),
        # API가 영어 이름 뒤에 '-si'를 붙여 반환하는 경우도 같은 도시로 인식
        _city_name_key(f"{english_alias}-si"),
    }

    # 사용자가 정한 시·도 확인
    # - '광주, 경기도'처럼 쉼표 뒤에 적은 경우 → 그 시·도
    # - '광주광역시'처럼 광역시·특별시 이름을 적은 경우 → 그 광역시 자체가 시·도이다.
    #   (광역시는 행정구역상 시·도와 같은 단계이므로 API의 admin1도 '광주광역시'가 된다.)
    #   이렇게 해야 '광주광역시'를 입력했을 때 이름이 같은 '경기도 광주시'를 고르지 않는다.
    metro_endings = ("특별시", "광역시", "특별자치시")
    requested_province = None
    if parts[0].endswith(metro_endings):
        requested_province = _province_name_key(parts[0])
    if len(parts) == 2:
        written_province = _province_name_key(parts[1])
        if requested_province is not None and requested_province != written_province:
            # 예: '광주광역시, 경기도'처럼 서로 맞지 않는 입력은 어느 쪽인지 다시 묻는다.
            raise WeatherError("needs_input", f"'{city}'의 도시와 시·도가 서로 맞지 않습니다. 다시 확인해 주세요.")
        requested_province = written_province

    # 시·도를 적지 않으면 정할 수 없는 이름(예: '광주')은 API 결과를 고르지 않고 바로 다시 묻는다.
    # 이전에는 '광주'만 입력하면 검색 결과의 좌표 하나를 그대로 사용했다.
    if requested_province is None and requested_city in AMBIGUOUS_CITY_NAMES:
        raise WeatherError("needs_input",
                           f"'{city}'(이)라는 지역은 여러 곳입니다. "
                           f"{AMBIGUOUS_CITY_NAMES[requested_city]}처럼 입력해 주세요.")


    # 4. Open-Meteo 지역 검색 API를 이용한 지역 검색
    # Geocoding API는 한국어를 지원하지만 모든 국내 지명이 등록된 것은 아님
    # '광주, 경기도'를 통째로 보내지 않고 도시명 '광주'만 먼저 검색한다.
    # 시·도인 '경기도'는 API 검색 결과의 admin1에서 별도로 확인한다.
    search_name = parts[0] if len(parts) == 2 else city

    results = []
    searched_names = set()
    # (추가) 이미 검색한 검색어를 기록하여 같은 검색어로 API를 두 번 호출하지 않는다.
    # 예전에는 '안산' → 'Ansan' 검색 후 마지막 단계에서 'Ansan'을 한 번 더 호출하는 경우가 있었다.

    def search(name: str) -> None:
        if name in searched_names:
            return  # 같은 영어 별칭을 불필요하게 두 번 검색하지 않음
        searched_names.add(name)
        results.extend(_fetch_geocoding_results(name))
        # extend(): 추가 검색 결과를 기존 리스트에 이어 붙임

    # 이름과 시·도가 동시에 맞는 후보가 없을 때만 검색어를 바꾸어 다시 시도한다.
    # (수정) 재검색 여부를 '검색 결과 개수'가 아니라 _is_matching_place() 기준으로 판단한다.
    # 예전에는 '안산' 검색 결과가 산(MT) 하나뿐이면 결과가 1개라는 이유로 재검색하지 않았다.
    def has_matching_place() -> bool:
        return any(_is_matching_place(place, valid_names, requested_province) for place in results)

    search(search_name)
    # 시·도까지 입력했다면 그 시·도에 속하는 지역이 실제로 검색되었는지 확인한다.
    # 예: '광주, 경기도'에서는 광주광역시만 발견되어도 검색 성공으로 취급하지 않는다.

    # 4-1. 맞는 후보가 없을 때만 차례대로 보완 검색 (왜 별칭 검색이 필요한가?)
    # API에는 한국어 이름이 '안산시'처럼 행정구역 명칭으로만 있거나, 영어 이름(Ansan)으로만 있는 경우가 있다.
    # 그래서 같은 도시를 다른 표기로 다시 찾아본다. (좌표를 임의로 만들지 않음)
    fallback_names = []
    if parts[0].endswith(metro_endings):
        # 예: '광주광역시'가 검색되지 않으면 '광주'로 검색 (시·도는 3-1에서 이미 광주광역시로 정함)
        fallback_names.append(requested_city)
    elif re.search(r"[가-힣]", search_name) and not search_name.endswith(("시", "군", "구")):
        # 지역명 끝에 시·군·구가 없다면 '시'를 붙여 다시 검색
        # 예: '안산' 검색에서 산만 나오면 '안산시'로 다시 검색
        # 예: '광주' 검색에서 경기도 광주시가 없다면 '광주시'로 다시 검색
        fallback_names.append(f"{search_name}시")
    if english_alias != parts[0]:
        # 검색 결과가 없고 등록된 영어 별칭이 있으면 해당 이름으로 다시 검색
        # 한국어 결과에 없다면 표준 영문 이름도 시도한다.
        fallback_names.append(english_alias)
        # 그래도 도시 후보가 없으면 등록된 영어 지역명으로 다시 검색
        # 영문 지명에 행정구역 접미사 -si를 추가하여 마지막으로 재검색
        fallback_names.append(f"{english_alias}-si")

    for name in fallback_names:
        if has_matching_place():
            break  # 맞는 후보를 찾았으면 더 이상 API를 호출하지 않음
        search(name)


    # 5. 산·마을 등 잘못된 지역이 검색되는 경우 추가 검색
    # 예: "안산" 검색 시 같은 이름의 산이나 마을이 먼저 검색될 수 있음
    # 시·군급 도시 후보가 없으면 "안산시"처럼 행정구역 명칭으로 재검색
    # (수정) 이 재검색은 위 4-1에서 같은 기준(_is_matching_place)으로 함께 처리한다.
    #        산·마을 후보는 _is_city_level_place()에서 걸러지므로 '맞는 후보 없음'으로 판단되어 재검색된다.


    # 6. 검색된 지역 중 정상적인 한국 지역과 좌표만 선별
    candidates = []
    # 유효한 지역 후보들을 저장할 리스트

    seen_coordinates = set()
    # 이미 확인한 좌표를 저장하여 중복 지역을 제거
    # set(): 중복된 값을 허용하지 않는 자료구조

    for place in results:
        # 검색 결과를 하나씩 검사

        if not _is_matching_place(place, valid_names, requested_province):
            continue
            # 한국이 아니거나 데이터 형식이 잘못된 후보는 건너뜀
            # (수정) 이름·시·도·도시 여부·좌표가 맞지 않는 후보도 여기서 모두 건너뜀
            # continue: 현재 반복을 중단하고 다음 후보로 넘어감

        coordinates = _place_coordinates(place)
        if coordinates is None:
            continue
            # (추가) _is_matching_place()에서 이미 좌표를 검사했으므로 실제로는 여기에 오지 않는다.
            # 하지만 _place_coordinates()는 None을 반환할 수 있다고 선언되어 있어서,
            # None 검사를 해 두어야 VS Code(Pylance)가 아래에서 튜플로 안전하게 쓴다고 판단한다. (타입 좁히기)
        latitude, longitude = coordinates

        # 왜 좌표 중복 제거를 나중에 수행하는가?
        # 예전에는 이름·시·도를 확인하기 전에 좌표 중복을 먼저 제거했다.
        # 그러면 잘못된 시·도의 후보가 먼저 좌표를 차지하여, 같은 좌표의 올바른 후보가 '중복'으로 삭제될 수 있었다.
        # 그래서 조건을 모두 통과한 후보끼리만 중복을 비교하고, 시·도가 다르면 좌표가 같아도 다른 지역으로 본다.
        # 좌표를 소수점 다섯 자리까지 반올림하여 중복 여부 확인
        coordinate_key = (_place_province_key(place), round(latitude, 5), round(longitude, 5))

        if coordinate_key not in seen_coordinates:
            # 처음 등장한 좌표라면 후보 목록에 추가
            candidates.append(place)
            seen_coordinates.add(coordinate_key)


    # 7. 사용 가능한 지역 후보가 하나도 없는 경우 오류 처리
    # 왜 조건에 맞는 후보가 없으면 오류를 반환하는가?
    # → 다른 지역의 날씨를 '정상(ok)'으로 넘기면 3단계 LLM이 잘못된 날씨로 외출을 판단하게 된다.
    #   틀린 위치의 날씨를 주는 것보다 "다시 입력해 주세요"라고 묻는 것이 어르신에게 더 안전하다.
    if not candidates:
        if len(parts) == 2:
            # 일치하는 지역이 없을 때 다른 광주를 임의로 선택하지 않는다.
            raise WeatherError(
                "needs_input",
                f"'{city}'와 일치하는 시·도 정보를 확인하지 못했습니다. "
                "도시 이름과 시·도를 확인해 주세요."
            )
        raise WeatherError("needs_input",
                           f"'{city}' 지역을 확인하지 못했습니다. 도시명이나 '광주, 경기도'처럼 행정구역을 입력해 주세요.")


    # 7-1. 사용자가 시·도를 함께 적었다면 같은 시·도에 속한 지역만 남긴다.
    # 예: '광주, 경기도' → admin1이 경기도인 후보만 선택 (광주광역시는 제외)
    # 8. 검색 결과 중 사용자가 입력한 지역명과 정확히 일치하는 후보 우선
    # 시·도만 맞는 다른 도시를 잘못 선택하지 않도록 지명도 반드시 확인한다.
    # 9. 검색 결과에서 산·마을보다 실제 도시 후보를 우선
    # (수정) 7-1·8·9단계의 검사는 6단계의 _is_matching_place()로 옮겼다.
    #        예전에는 정확히 일치하는 이름이 없어도 후보가 남아 있으면 그 좌표를 사용했다.
    #        (예: '가람'을 검색했는데 '다른도시' 하나만 나와도 그 좌표를 반환)
    #        이제는 이름·시·도·도시 여부를 모두 만족한 후보만 6단계를 통과한다.


    # 9-1. (추가) 같은 도시가 두 가지 기록으로 함께 검색된 경우 하나로 합치기
    # GeoNames에는 한 도시가 '행정구역(ADM1·ADM2)' 기록과 '시청 소재지(PPLA·PPLA2)' 기록으로 따로 있을 수 있다.
    # 예: '대전' 검색 → 대전(PPLA, 시청 소재지)과 대전광역시(ADM1, 행정구역)
    # (수정) 처음에는 '시·도가 같고 30 km 안'이기만 하면 합쳤는데, 그러면 같은 시·도 안에서
    #        하위 행정구역(admin2)이 서로 다른 두 장소도 첫 번째 좌표로 합쳐질 수 있었다.
    #        그래서 아래 네 조건을 '모두' 만족할 때만 같은 도시의 중복 기록으로 보고 합친다.
    #   ① 시·도가 모두 같다.
    #   ② 시청 소재지(PPLC·PPLA·PPLA2) 기록이 정확히 1개이고, 나머지는 모두 행정구역(ADM1·ADM2) 기록이다.
    #      → 사람이 사는 장소가 2곳 이상이면 서로 다른 장소일 수 있으므로 합치지 않는다.
    #   ③ admin2(시·군·구)가 적혀 있는 기록끼리는 admin2가 모두 같다.
    #   ④ 서로 30 km 안에 있다. (행정구역의 중심점과 시청 위치는 조금 떨어져 있을 수 있음)
    # 하나라도 만족하지 않으면 합치지 않고, 아래 10단계에서 사용자에게 다시 묻는다. (확신할 수 없으면 묻기)
    if len(candidates) > 1:
        centers = [place for place in candidates
                   if str(place.get("feature_code", "")).upper() in {"PPLC", "PPLA", "PPLA2"}]
        regions = [place for place in candidates
                   if str(place.get("feature_code", "")).upper() in {"ADM1", "ADM2"}]
        admin2_names = {_city_name_key(str(place["admin2"])) for place in candidates if place.get("admin2")}

        points = []
        for place in candidates:
            coordinates = _place_coordinates(place)
            if coordinates is not None:  # 타입 좁히기 (6단계를 통과한 후보라 실제로는 항상 좌표가 있음)
                points.append(coordinates)
        # 위도 1도 ≈ 111 km, 한국 위도(약 33~38도)에서 경도 1도 ≈ 88 km로 계산한 간단한 거리
        farthest_km = max(
            math.hypot((a[0] - b[0]) * 111, (a[1] - b[1]) * 88)
            for a in points for b in points
        )

        same_city_records = (
            len({_place_province_key(place) for place in candidates}) == 1   # ①
            and len(centers) == 1 and len(centers) + len(regions) == len(candidates)  # ②
            and len(admin2_names) <= 1                                       # ③
            and len(points) == len(candidates) and farthest_km <= 30         # ④
        )
        if same_city_records:
            candidates = centers
            # 시청 소재지 기록 1개만 남긴다. (날씨는 실제 생활 중심지인 시청 위치 기준으로 조회)


    # 10. 동일한 이름의 지역이 여러 개 남아 있는 경우 처리
    if len(candidates) > 1:
        choices = []

        # 최대 5개의 지역 후보에 대한 상세 위치 안내 생성
        for place in candidates[:5]:
            # [:5]: 리스트에서 앞의 5개 항목까지만 사용

            label = ", ".join(
                str(place[key])
                for key in ("name", "admin1", "admin2")
                if place.get(key)
            )
            # 지역명, 상위 행정구역, 하위 행정구역을 조합하여 표시

            if label not in choices:
                choices.append(label)
                # 동일한 안내 문구가 중복되지 않도록 저장

        # 여러 지역 중 하나를 임의로 선택하지 않고 사용자에게 재입력 요청
        raise WeatherError("needs_input",
                           f"'{city}'와 일치하는 지역이 여러 곳입니다: {' / '.join(choices)}. "
                           "'도시, 시·도' 형식으로 더 구체적으로 입력해 주세요.")


    # 11. 최종적으로 선택된 지역의 위도와 경도 반환
    place = candidates[0]
    # 후보가 하나만 남았으므로 첫 번째 지역 선택
    # (이 후보는 6단계에서 이름·시·도·도시 여부·좌표 검사를 모두 통과한 후보이다.)

    coordinates = _place_coordinates(place)
    if coordinates is None:
        # (추가) 6단계에서 좌표를 검사했으므로 실제로는 오지 않지만, None을 그대로 반환하지 않도록 막는다.
        # 이 검사 덕분에 함수의 반환 자료형이 항상 tuple[float, float]가 된다. (Pylance 타입 좁히기)
        raise WeatherError("needs_input", f"'{city}' 지역의 좌표를 확인하지 못했습니다. 다시 입력해 주세요.")
    return coordinates
    # 최종 결과: (위도, 경도) 형태의 튜플 반환



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




#외출 출발 시간의 구간을 구하는 함수
# 사용자가 입력한 외출 시간을 기준으로, 해당 시간이 포함된 1시간 구간을 계산하는 함수
def departure_hour_window(outing: datetime) -> tuple[datetime, datetime]:
    # 1. 외출 출발 시간의 구간을 구하는 함수
    # outing : 사용자가 입력한 외출 날짜와 시간 // 시작 시간과 종료 시간 두 개를 반환한다는 의미

    """출발 시각이 속한 정시 구간 [시작, 끝)을 구한다. 예) 14:30 → 14:00~15:00."""
    # 2. 외출 시간의 분과 초를 0으로 바꿔 시작 시각 계산
    start = outing.replace(minute=0, second=0, microsecond=0)

    # 3. 시작 시각과 1시간 뒤의 종료 시각을 반환
    return start, start + timedelta(hours=1)

    # 요약 : 외출 시간이 14시 30분이라면 14시부터 15시까지의 시간 구간을 계산해 날씨 조회에 활용하는 함수




# API 요청용 시간 변환 함수
# 날짜와 시간 데이터를 Open-Meteo API 요청에 사용할 문자열 형식으로 변환하는 함수
def api_hour(moment: datetime) -> str:
    # api_hour : API에 전달할 날짜, 시간 형식을 만드는 함수
    # moment : 변환할 날짜와 시간 (datetime 객체)
    # -> str : 변환된 결과를 문자열(String)로 반환한다는 의미

    """API에서 Asia/Seoul을 별도로 지정하므로 현지 시각 문자열만 사용한다."""
    
    # datetime 객체를 YYYY-MM-DDTHH:MM 형식의 문자열로 변환
    return moment.strftime("%Y-%m-%dT%H:%M")




# 시간별 날씨 예보 조회 함수
# 지정한 지역과 시간대의 날씨 예보를 Open-Meteo API에서 조회하는 함수
def fetch_hourly(city: str, start: datetime, end: datetime) -> dict: # 1. 지역과 조회 시작·종료 시간을 입력받음
    """필요한 시각과 항목만 지정해 Open-Meteo를 한 번 호출한다."""


    # 2. 지역의 위도와 경도 가져오기
    # 도시 이름에 해당하는 위도와 경도 가져옴
    latitude, longitude = resolve_city_coordinates(city) # 기존 지역 또는 Geocoding API로 좌표 확인

    # 시작·끝 시각으로 조회할 때 forecast_days를 함께 보내면 HTTP 400이 발생한다.


    # 3. API 요청 정보 생성 // 날씨 서버에 어떤 데이터를 요청 할 것인지 설정.
    # 조회 지역, 시간대, 기상 항목, 단위를 API 요청 형식으로 변환
    query = urlencode({
        "latitude": latitude, "longitude": longitude, "timezone": "Asia/Seoul",
        "start_hour": api_hour(start), "end_hour": api_hour(end),
        "hourly": ",".join(HOURLY_FIELDS),
        "temperature_unit": "celsius", "wind_speed_unit": "ms", "precipitation_unit": "mm",
    })


    # 4. 실제 API 호출 및 오류 처리
    try:
        with request.urlopen(f"{API_URL}?{query}", timeout=8) as response:
        # request.urlopen() : 날씨 API 서버에 요청을 보냄
        # timeout=8 : 서버 응답을 기다리는 시간 제한을 8초로 설정
        # json.load(response) : 서버가 보내온 JSON 데이터를 Python 객체로 변환
            payload = json.load(response)

    # 이부분 부터 오류 처리 
    # API 요청 실패 또는 호출 한도 초과
    except HTTPError as error:
        error.close()
        if error.code == 429:
            message = "호출 한도에 도달했습니다(HTTP 429). 잠시 후 다시 시도해 주세요."
        else:
            message = f"기상 API가 요청을 처리하지 못했습니다(HTTP {error.code})."
        raise WeatherError("api_error", message) from error
    # 인터넷 연결이나 서버 통신 문제
    except (OSError, HTTPException) as error:
        message = "기상 API에 연결하지 못했습니다. 인터넷 연결을 확인한 뒤 다시 시도해 주세요."
        raise WeatherError("network_error", message) from error
    # 서버 응답을 정상적으로 해석하지 못한 경우
    except ValueError as error:
        raise WeatherError("data_unavailable", "기상 API의 응답을 JSON으로 읽을 수 없습니다.") from error


    # 5. 응답 데이터 확인 및 반환
    # 서버에서 받아온 데이터가 정상인지 확인함.
    # 데이터가 딕셔너리 인지 검사 // API가 오류를 반환했는지 검사 
    if not isinstance(payload, dict):
        raise WeatherError("data_unavailable", "기상 API 응답 형식을 확인할 수 없습니다.")
    if payload.get("error"):
        raise WeatherError("api_error", "기상 API가 요청을 거절했습니다.")
    
    return payload  # 문제가 없으면 return payload로 결과 반환




# 시간별 날씨 데이터 검증 및 추출 함수
# 날씨 API에서 받아온 데이터 중 필요한 기상 값을 꺼내고, 그 값이 정상적인 데이터인지 검사하는 함수
def hourly_value(   # 1. 조회할 기상 항복과 검사 조건을 전달 받음.
    hourly: dict, units: dict, field: str, index: int | None, unit: str,
    minimum: float = -math.inf, maximum: float = math.inf,
) -> float | None: # 정상적인 숫자 또는 값이 없음을 의미하는 None을 반환할 수 있다.
    
    """시각·단위·숫자가 확인되는 값만 읽는다. 0은 유효하며 누락은 None이다."""


    # 2. 필요한 기상 데이터 가져오기
    values = hourly.get(field)

    # 3. 가져온 데이터가 정상적인 구조인지 확인하는 부분
    # 조회할 시간의 위치가 있는지 확인 // 기상 데이터의 단위가 올바른지 확인 // 데이터가 리스트 형태인지 확인 // 시간 목록과 기상 데이터 목록의 길이가 같은지 확인
    # 조건 만족 못하면 None 반환
    if index is None or units.get(field) != unit or not isinstance(values, list):
        return None
    if len(values) != len(hourly["time"]):
        return None
    

    # 4. 숫자가 정상적인지 검사
    # index에 해당하는 시간의 실제 기상 값을 가져옴
    value = values[index]
    # (수정) 원래 코드: if type(value) not in (int, float) or not math.isfinite(value): return None
    # 10**400처럼 너무 큰 정수는 math.isfinite()에서 OverflowError가 발생하므로
    # 자료형·NaN·무한대·너무 큰 수를 _safe_number()에서 한 번에 검사한다. (bool도 숫자로 보지 않음)
    value = _safe_number(value)
    if value is None:
        return None


    # 5. 정상적인 기상 값 반환
    # 허용 범위에 포함되면 값을 반환하고, 아니면 None 반환
    return value if minimum <= value <= maximum else None




# 외출 시간대에 필요한 날씨 정보를 추출하는 함수입니다.
# API에서 받은 전체 날씨 데이터 중 외출 시간대의 기상 정보만 추출하는 함수
def select_weather(payload: dict, start: datetime, end: datetime) -> dict:
    """기온·풍속은 구간의 시작 행, 직전 1시간 누적 강수는 끝 행에서 읽는다."""
    # payload : 날씨 API에서 받아온 전체 응답 데이터
    # start : 조회할 시간대의 시작 시각
    # end : 조회할 시간대의 종료 시각
    # -> dict : 선택한 날씨 정보를 딕셔너리 형태로 반환


    # 2. 날씨 데이터의 구조 검사
    # 잘못된 날씨 데이터가 SafeMate에 사용되는 것을 방지하는 역할
    hourly = payload.get("hourly") # 시간별 기상 예보 데이터
    units = payload.get("hourly_units") # 기온, 풍속 등의 단위 정보

    if payload.get("utc_offset_seconds") != 9 * 3600: # 한국 표준시인 UTC+9 기준인지 확인
        raise WeatherError("data_unavailable", "응답이 한국 시각 기준인지 확인할 수 없습니다.")
    # 데이터가 올바른 형식인지 검사
    if not isinstance(hourly, dict) or not isinstance(units, dict):
        raise WeatherError("data_unavailable", "시간별 예보 또는 단위 정보가 없습니다.")

    
    # 3. 외출 시간대에 해당하는 데이터 찾기
    times = hourly.get("time")
    if not isinstance(times, list) or not all(isinstance(stamp, str) for stamp in times):
        raise WeatherError("data_unavailable", "예보 시각을 식별할 수 없습니다.")
    
    indices = {timestamp: index for index, timestamp in enumerate(times)}
    if len(indices) != len(times):
        raise WeatherError("data_unavailable", "중복된 예보 시각이 있어 구간을 식별할 수 없습니다.")
    
    start_index = indices.get(api_hour(start))
    end_index = indices.get(api_hour(end))


    # 4. 필요한 기상 정보 추출
    # 기온 // 체감온도 // 풍속 // 강수량 // 강수 확률
    readings = {
        "temperature_c": hourly_value(hourly, units, "temperature_2m", start_index, "°C"),
        "apparent_temperature_c": hourly_value(hourly, units, "apparent_temperature", start_index, "°C"),
        "wind_speed_mps": hourly_value(hourly, units, "wind_speed_10m", start_index, "m/s", 0),
        "precipitation_mm": hourly_value(hourly, units, "precipitation", end_index, "mm", 0),
        "precipitation_probability_pct": hourly_value(
            hourly, units, "precipitation_probability", end_index, "%", 0, 100
        ),
    }

    # 5. 최종 날씨 데이터 반환
    # 추출한 날씨 데이터가 모두 None이라면 오류를 발생시킴.
    # 하나라도 사용할 수 있는 값이 있다면 readings 딕셔너리를 반환
    # 일부 값만 누락된 경우에는 오류를 발생시키지 않고 해당 항목을 None으로 남김
    if all(value is None for value in readings.values()):
        raise WeatherError("data_unavailable", 
                           "해당 시간대에 사용할 수 있는 예보 값이 없습니다.")
    return readings




# 외출 정보를 입력받아 날씨를 조회하고, 그 결과를 알려주는 함수
# 앞에서 만든 날씨 관련 함수들을 하나로 연결하여,
# 어르신의 외출 시간대에 해당하는 날씨 정보를 최종적으로 조회하고 결과를 반환하는 함수
def get_outing_weather( # 1. 외출 날짜, 외출 시간, 지역을 입력받음
    outing_date: str | None = None, 
    outing_time: str | None = None, 
    city: str | None = None
    # = None : 해당 정보가 전달되지 않아도 함수를 호출할 수 있도록 설정한 기본값
) -> dict: # 결과를 딕셔너리 형태로 반환
    """공개 진입점. 항상 status, message, weather를 담은 딕셔너리를 반환한다."""
    

    # 2. 사용자 외출 정보 확인
    # 어르신이 입력한 외출 날짜, 시간, 지역이 올바른지 확인하는 부분
    now = datetime.now(KST) # 현재 한국 시간 가져옴
    try:
        outing = parse_outing_time(outing_date, outing_time, now)   # 외출 날짜와 시간을 변환하고 검사
        
        if not isinstance(city, str) or not city.strip():
            # 지역 이름이 입력되었는지 확인 (지원 지역은 Geocoding API에서 추가 확인)
            raise WeatherError("needs_input", "외출 지역을 입력해 주세요. 예) 인천, 서울, 대전, 수원")
        
        city = city.strip() # 지역 이름의 앞뒤 공백 제거


        # 3. 실제 날씨 조회 실행
        start, end = departure_hour_window(outing)

        
        last_day = now.date() + timedelta(days=FORECAST_DAYS - 1)
        last_hour = datetime.combine(last_day, time(23), tzinfo=KST)

        payload = fetch_hourly(city, start, min(end, last_hour))
        readings = select_weather(payload, start, end)


    # 4. 오류 처리
    # 오류가 발생하더라도 일정한 형식으로 결과를 반환하도록 하는 부분
    except WeatherError as error:
        return {"status": error.status, # 오류 종류
                "message": str(error),  # 사용자에게 보여줄 안내 메시지
                "weather": None         # 오류가 발생했으므로 None
                }


    # 5. 조회된 기상 정보와 누락된 항목을 딕셔너리로 정리
    # 예보 데이터라는 표시 // 날씨 정보 출처 // 외출 지역 // 어르신이 입력한 외출 시각
    # 날씨 조회 시간 구간 // 기온, 체감온도, 풍속, 강수량 등 조회 결과 // 조회하지 못한 기상 항목 목록
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


    # 6. 전체 조회 성공 또는 일부 누락 여부에 따라 결과 반환
    if missing:
        message = "일부 예보만 확인했습니다. 누락 항목은 missing_fields에 표시했습니다."
        return {"status": "partial", "message": message, "weather": weather}
    return {"status": "ok", "message": "외출 시간대의 예보를 조회했습니다.", "weather": weather}


def main() -> None:
    """터미널에서 직접 실행할 때만 입력을 받고 결과를 표시한다."""
    city = input("외출 지역 (예: 인천 / 서울 / 대전 / 부산 / 수원): ")
    outing_date = input("외출 날짜 (오늘 / 내일 / 모레 / YYYY-MM-DD): ")
    outing_time = input("외출 시간 (24시간제 HH:MM): ")
    result = get_outing_weather(outing_date, outing_time, city)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
