import streamlit as st
import pandas as pd
import requests
import os
import mysql.connector
from sqlalchemy import create_engine
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import openai
from gtts import gTTS
from io import BytesIO

# ── 1. .env 파일 로드 ──
load_dotenv()

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI client 생성 (⭐ 신버전 스타일)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ── 2. DB 연결 ──
engine = create_engine('mysql+pymysql://skn14:skn14@localhost:3306/accidentdb')

# ── 3. Streamlit 페이지 상태 관리 ──
if 'page' not in st.session_state:
    st.session_state['page'] = 'main'

# ── 4. 사고유형 설명 딕셔너리 ──
type_main_explanation = {
    '차대사람': '자동차와 보행자가 충돌해 발생한 사고입니다.',
    '차대차': '두 대 이상의 차량 간 충돌로 발생한 사고입니다.',
    '차량단독': '차량이 단독으로 미끄러지거나 시설물과 충돌하여 발생한 사고입니다.'
}

# ── 5. 메인 페이지 ──
if st.session_state['page'] == 'main':
    st.title("🚗 교통사고 정보 조회")

    df = pd.read_sql("select distinct region1, region2 from car_accident", con=engine)
    region1_list = sorted(df['region1'].unique())

    # 출발지 입력
    st.header("출발지 선택")
    selected_region1_departure = st.selectbox("출발지 시/도 선택", region1_list, key="departure_region1")
    filtered_region2_departure = df[df['region1'] == selected_region1_departure]
    region2_list_departure = sorted(filtered_region2_departure['region2'].unique())
    selected_region2_departure = st.selectbox("출발지 시/군/구 선택", region2_list_departure, key="departure_region2")

    # 경유지 입력
    st.header("경유지 선택")
    selected_region1_waypoint = st.selectbox("경유지 시/도 선택", region1_list, key="waypoint_region1")
    filtered_region2_waypoint = df[df['region1'] == selected_region1_waypoint]
    region2_list_waypoint = sorted(filtered_region2_waypoint['region2'].unique())
    selected_region2_waypoint = st.selectbox("경유지 시/군/구 선택", region2_list_waypoint, key="waypoint_region2")

    # 도착지 입력
    st.header("도착지 선택")
    selected_region1_destination = st.selectbox("도착지 시/도 선택", region1_list, key="destination_region1")
    filtered_region2_destination = df[df['region1'] == selected_region1_destination]
    region2_list_destination = sorted(filtered_region2_destination['region2'].unique())
    selected_region2_destination = st.selectbox("도착지 시/군/구 선택", region2_list_destination, key="destination_region2")

    # 사고 유형 조회 버튼
    if st.button("사고 유형 조회"):
        query_departure = f"""
        select type_main, type_sub, law
        from car_accident
        where region1 = '{selected_region1_departure}' and region2 = '{selected_region2_departure}'
        """
        query_waypoint = f"""
        select type_main, type_sub, law
        from car_accident
        where region1 = '{selected_region1_waypoint}' and region2 = '{selected_region2_waypoint}'
        """
        query_destination = f"""
        select type_main, type_sub, law
        from car_accident
        where region1 = '{selected_region1_destination}' and region2 = '{selected_region2_destination}'
        """

        result_departure = pd.read_sql(query_departure, con=engine)
        result_waypoint = pd.read_sql(query_waypoint, con=engine)
        result_destination = pd.read_sql(query_destination, con=engine)

        st.session_state['result_departure'] = result_departure
        st.session_state['result_waypoint'] = result_waypoint
        st.session_state['result_destination'] = result_destination

        st.session_state['selected_departure'] = (selected_region1_departure, selected_region2_departure)
        st.session_state['selected_waypoint'] = (selected_region1_waypoint, selected_region2_waypoint)
        st.session_state['selected_destination'] = (selected_region1_destination, selected_region2_destination)

    # 사고 유형 출력
    for label, result_key, region1, region2 in [
        ("출발지", 'result_departure', selected_region1_departure, selected_region2_departure),
        ("경유지", 'result_waypoint', selected_region1_waypoint, selected_region2_waypoint),
        ("도착지", 'result_destination', selected_region1_destination, selected_region2_destination)
    ]:
        if result_key in st.session_state:
            result = st.session_state[result_key]
            st.subheader(f"📍 {label} ({region1} {region2}) 교통사고 유형 정보")
            st.subheader(f"🗺️ {label} 지도")
            st.write("여기에 지도 출력 코드 들어갈 예정")

            if not result.empty:
                total_pages = (len(result) - 1) // 5 + 1
                page = st.number_input(
                    label=f'{label} 페이지 선택',
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    step=1,
                    key=f'{label}_page'
                )
                start_idx = (page - 1) * 5
                end_idx = start_idx + 5
                paginated_result = result.iloc[start_idx:end_idx]

                for i, row in paginated_result.iterrows():
                    type_main = row['type_main']
                    explanation = type_main_explanation.get(type_main, "해당 사고 유형에 대한 설명이 없습니다.")
                    type_sub = row['type_sub']
                    law = row['law']

                    with st.container():
                        st.markdown("---")
                        st.markdown(f"### 🚦 사고 유형 {i + 1}")
                        st.markdown(f"👉 {explanation}")
                        st.markdown(f"**🔎 상세 사고 유형:** {type_sub}")
                        st.markdown(f"**📜 관련 법령:** {law}")
                        st.markdown("---")
            else:
                st.warning(f"{label} ({region1} {region2}) 지역에 대한 사고 유형 정보가 없습니다.")
            st.info("❗안전 운전을 위해 해당 사고 유형에 각별히 주의하시기 바랍니다.❗")

    # 뉴스 페이지 이동 버튼
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([6, 2, 2])
    with col3:
        if st.button("📰 뉴스 보기"):
            st.session_state['page'] = 'news'

# ── 6. 뉴스 페이지 ──
elif st.session_state['page'] == 'news':
    st.title("📰 지역별 교통사고 뉴스 요약 & TTS")

    departure_region1, departure_region2 = st.session_state.get('selected_departure', ("", ""))
    transit_region1, transit_region2 = st.session_state.get('selected_waypoint', ("", ""))
    destination_region1, destination_region2 = st.session_state.get('selected_destination', ("", ""))

    departure_keyword = f"{departure_region1} {departure_region2}"
    transit_keyword = f"{transit_region1} {transit_region2}"
    destination_keyword = f"{destination_region1} {destination_region2}"

    # 뉴스 검색 함수
    def search_news(keyword: str) -> list:
        url = 'https://openapi.naver.com/v1/search/news.json'
        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
        }
        params = {
            'query': f'{keyword} 교통사고',
            'display': 3,
            'start': 1,
            'sort': 'sim',
        }
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                return data.get('items', [])
            else:
                st.error(f"뉴스 검색 실패: {response.status_code}")
                return []
        except Exception as e:
            st.error(f"뉴스 검색 오류: {str(e)}")
            return []

    # HTML 전처리
    def clean_html(soup: BeautifulSoup):
        for tag in soup(['script', 'style', 'iframe', 'footer', 'nav', 'header', '.ad', '.related-article']):
            tag.decompose()
        for tag in soup():
            tag.attrs = {}

    # 뉴스 본문 가져오기
    def get_html_text(url: str) -> str:
        try:
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            clean_html(soup)
            return soup.get_text(separator="\n")
        except Exception as e:
            st.error(f"본문 가져오기 실패: {str(e)}")
            return ""

    # 뉴스 요약
    def summarize_text(text: str) -> str:
        try:
            lines = text.splitlines()
            cleaned_text = "\n".join([line.strip() for line in lines if len(line.strip()) > 30])

            if len(cleaned_text) > 8000:
                cleaned_text = cleaned_text[:8000]

            prompt = f"""
            다음 웹페이지 텍스트를 읽고 중요 기사 내용만 3줄로 요약해줘.
            메뉴, 광고, 댓글, 저작권 문구는 무시해줘.

            {cleaned_text}
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"요약 실패: {e}")
            return "요약 실패"

    # 뉴스 출력 함수
    def handle_news_selection(keyword, label):
        if keyword.strip():
            with st.expander(f"🗺️ {label} ({keyword}) 뉴스 보기", expanded=True):
                with st.spinner("뉴스 검색 중입니다..."):
                    news_list = search_news(keyword)
                    if news_list:
                        for idx, news in enumerate(news_list, start=1):
                            title = news.get('title')
                            link = news.get('originallink')

                            with st.container():
                                st.markdown(f"### 🔗 뉴스 #{idx}")
                                st.markdown(f"[{title}]({link})")

                                article = get_html_text(link)
                                if not article:
                                    st.error("❗ 본문 가져오기 실패")
                                    continue

                                summary = summarize_text(article)
                                st.success("✨ 요약 결과")
                                st.write(summary)
                            st.markdown("---")
                    else:
                        st.warning(f"{label} 뉴스가 없습니다.")

    handle_news_selection(departure_keyword, "출발지")
    handle_news_selection(transit_keyword, "경유지")
    handle_news_selection(destination_keyword, "도착지")

    if st.button("⬅️ 뒤로 가기"):
        st.session_state['page'] = 'main'

