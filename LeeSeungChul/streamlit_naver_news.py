# --- 필요한 모듈 불러오기 ---
import streamlit as st
import mysql.connector
from kor_location_map import location_map

# --- 데이터베이스 연결 함수 ---
def connect_to_db():
    config = {
        "host": "localhost",
        "user": "team5",
        "password": "teamteam5",
        "database": "team5_database"
    }
    return mysql.connector.connect(**config)

# --- 특정 지역 뉴스 검색 함수 ---
def get_news(location1, location2, limit=3):
    """
    시/도, 구/군 조건에 맞는 뉴스를 최신순으로 최대 limit개까지 가져온다
    """
    conn = connect_to_db()
    cursor = conn.cursor(dictionary=True)

    sql = """
        SELECT title, link, accident_type, accident_date
        FROM car_accident_naver_news
        WHERE accident_location1 = %s
          AND accident_location2 = %s
        ORDER BY accident_date DESC
        LIMIT %s
    """
    cursor.execute(sql, (location1, location2, limit))
    news = cursor.fetchall()

    cursor.close()
    conn.close()

    return news

# --- 뉴스 출력 함수 (중복 제거 + 줄간격 줄이기) ---
def display_news(title, news_list):
    """
    중복 제목 없이 뉴스 리스트를 출력하고
    줄 간격을 줄여서 빽빽하게 보여주는 함수
    """
    st.write(f"### {title}")

    if not news_list:
        st.info("😥 관련 뉴스가 없습니다.")
        return

    shown_titles = set()  # 중복 방지용

    for news in news_list:
        title_text = news["title"]
        link = news["link"]
        accident_type = news["accident_type"]
        accident_date = news["accident_date"]

        if title_text in shown_titles:
            continue  # 이미 보여준 뉴스는 스킵

        shown_titles.add(title_text)  # 출력한 제목 기억

        # 사고유형, 사고날짜와 함께 뉴스 제목을 표시 (줄간격 최소화)
        st.markdown(
            f"<span style='font-size:15px;'>"
            f"<strong>[{accident_type}] ({accident_date})</strong> "
            f"<a href='{link}' target='_blank'>{title_text}</a>"
            f"</span>",
            unsafe_allow_html=True
        )

# --- Streamlit 앱 시작 ---
st.title("🚗 출발지 - 경유지 - 도착지 뉴스 검색")

# --- 출발지 선택 ---
st.header("출발지 선택")
start_city = st.selectbox("출발 시/도 선택", list(location_map.keys()), key="start_city")
start_district = st.selectbox(f"{start_city}의 구/군 선택", location_map[start_city], key="start_district")

# --- 경유지 선택 (없어도 됨) ---
st.header("경유지 선택 (선택사항)")
use_waypoint = not st.checkbox("경유지 없음", key="no_waypoint_checkbox")

if use_waypoint:
    waypoint_city = st.selectbox("경유 시/도 선택", list(location_map.keys()), key="waypoint_city")
    waypoint_district = st.selectbox(f"{waypoint_city}의 구/군 선택", location_map[waypoint_city], key="waypoint_district")
else:
    waypoint_city = None
    waypoint_district = None

# --- 도착지 선택 ---
st.header("도착지 선택")
end_city = st.selectbox("도착 시/도 선택", list(location_map.keys()), key="end_city")
end_district = st.selectbox(f"{end_city}의 구/군 선택", location_map[end_city], key="end_district")

# --- "확인" 버튼 누르면 뉴스 검색 시작 ---
if st.button("확인"):
    st.subheader("📰 뉴스 결과")

    # 출발지 뉴스
    start_news = get_news(start_city, start_district, limit=3)

    # 경유지 뉴스
    if use_waypoint:
        waypoint_news = get_news(waypoint_city, waypoint_district, limit=3)

    # 도착지 뉴스
    end_news = get_news(end_city, end_district, limit=3)

    # 뉴스 출력
    display_news(f"출발지 [{start_city} {start_district}] 관련 뉴스", start_news)

    if use_waypoint:
        display_news(f"경유지 [{waypoint_city} {waypoint_district}] 관련 뉴스", waypoint_news)

    display_news(f"도착지 [{end_city} {end_district}] 관련 뉴스", end_news)