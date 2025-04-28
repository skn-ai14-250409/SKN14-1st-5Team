import streamlit as st
import mysql.connector
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from gtts import gTTS
from io import BytesIO
from datetime import datetime
import time

# ── 1) .env 로드 ─────────────────────────────────
load_dotenv()
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPEN_AI_API")

client = OpenAI(api_key=OPENAI_API_KEY)  # ⭐ 신버전 스타일로 클라이언트 생성

db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}

if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
    raise ValueError("NAVER_CLIENT_ID and NAVER_CLIENT_SECRET 이 설정되지 않았습니다")

# ── 2) DB 연결 함수 ───────────────────────────────
def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except mysql.connector.Error as err:
        st.error(f"DB 연결 실패: {err}")
        return None

# ── 3) NaverNews 클래스 정의 ─────────────────────
class NaverNews:
    def __init__(self, title, link, originallink, description, pubDate):
        self.title = title
        self.link = link
        self.originallink = originallink
        self.description = description
        self.pubDate = pubDate

# ── 4) 뉴스 저장 함수 ─────────────────────────────
def save_news_to_db(news_list: list, table_name: str):
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cursor:
                cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
                row_count = cursor.fetchone()[0]

                # 오래된 뉴스 삭제
                if row_count > 3:
                    to_delete = row_count - 3
                    cursor.execute(f'SELECT id FROM {table_name} ORDER BY pub_date ASC LIMIT %s', (to_delete,))
                    old_ids = cursor.fetchall()

                    for (news_id,) in old_ids:
                        cursor.execute(f'DELETE FROM {table_name} WHERE id = %s', (news_id,))
                    conn.commit()
                    st.info(f"✅ 뉴스가 3개를 넘어 {to_delete}개의 오래된 뉴스를 삭제했습니다!")

                # 중복된 뉴스 URL 확인하고 저장
                for news in news_list:
                    # 중복된 originallink가 있는지 확인
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE originallink = %s", (news.originallink,))
                    existing_news_count = cursor.fetchone()[0]

                    if existing_news_count == 0:  # 중복이 아니면 새로운 뉴스 추가
                        cursor.execute(f'''
                            INSERT INTO {table_name} (title, originallink, link, description, pub_date)
                            VALUES (%s, %s, %s, %s, %s)
                        ''', (news.title, news.originallink, news.link, news.description, news.pubDate))
                conn.commit()
    except mysql.connector.Error as e:
        st.error(f"DB 오류: {e}")


# ── 5) 뉴스 검색 함수 ────────────────────────────
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
        naver_news_list = []

        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            for item in items:
                naver_news_list.append(NaverNews(**item))
        else:
            st.error(f"뉴스 검색 실패: {response.status_code}")

        if not naver_news_list:
            st.warning("해당 키워드로 검색된 뉴스가 없습니다.")

        return naver_news_list
    except Exception as e:
        st.error(f"뉴스 검색 중 오류 발생: {str(e)}")
        return []

# ── 6) HTML 전처리 함수 ──────────────────────────
def clean_html(soup: BeautifulSoup):
    for tag in soup(['script', 'style', 'iframe', 'footer', 'nav', 'header', '.ad', '.related-article']):
        tag.decompose()

    for tag in soup():
        del tag['style']
        del tag['class']

# ── 7) 뉴스 텍스트 가져오기 ───────────────────────
def get_html_text(url: str) -> str:
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        clean_html(soup)
        return soup.get_text(separator="\n")
    except Exception as e:
        st.error(f"웹페이지 가져오기 실패: {str(e)}")
        return ""

# ── 8) 뉴스 요약 기능 (신버전) ─────────────────────
def summarize_text(text: str) -> str:
    try:
        lines = text.splitlines()

        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 30:
                cleaned_lines.append(line)

        cleaned_text = "\n".join(cleaned_lines)

        max_chars = 8000
        if len(cleaned_text) > max_chars:
            cleaned_text = cleaned_text[:max_chars]

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

        summary = response.choices[0].message.content.strip()
        return summary

    except Exception as e:
        print(f"요약 실패: {e}")
        return "요약에 실패했습니다."

# ── 9) Streamlit 앱 ──────────────────────────────
st.title("📰 지역별 교통사고 뉴스 요약 & TTS")

# 드롭다운
departure = st.selectbox("출발지를 선택하세요", ["지역을 선택해주세요", "서울", "부산", "대구", "인천"])
transit = st.selectbox("경유지를 선택하세요", ["지역을 선택해주세요", "서울", "부산", "대구", "인천"])
destination = st.selectbox("목적지를 선택하세요", ["지역을 선택해주세요", "서울", "부산", "대구", "인천"])

# 드롭다운 선택 시 뉴스 처리
def handle_news_selection(selected_location, table_name, label):
    if selected_location != "지역을 선택해주세요":
        with st.expander(f"🗺️ {label} ({selected_location}) 뉴스 보기", expanded=True):
            with st.spinner("뉴스를 불러오고 요약하는 중입니다... 잠시만 기다려주세요!"):
                news_list = search_news(selected_location)
                if news_list:
                    save_news_to_db(news_list, table_name)
                    process_news_from_db(table_name)


# 뉴스 처리 함수
def process_news_from_db(table_name: str):
    urls = []
    conn = None

    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT originallink FROM {table_name}")
                urls = [r[0] for r in cur.fetchall()]
    except Exception as e:
        st.error(f"DB 조회 오류: {e}")
    finally:
        if conn:
            conn.close()

    if not urls:
        st.warning(f"{table_name}에 저장된 뉴스가 없습니다.")
    else:
        # 중복 제거
        urls = list(dict.fromkeys(urls))

        for idx, url in enumerate(urls, start=1):
            with st.container():
                st.markdown(f"### 🔗 뉴스 #{idx}")
                st.markdown(f"[{url}]({url})")

                article = get_html_text(url)
                if not article:
                    st.error("❗ 전체 텍스트 가져오기 실패")
                    continue

                summary = summarize_text(article)

                if "요약에 실패했습니다." in summary:
                    st.warning(summary)
                    continue

                st.success("✨ 요약 결과")
                st.write(summary)

                try:
                    tts = gTTS(text=summary, lang="ko")
                    buf = BytesIO()
                    tts.write_to_fp(buf)
                    buf.seek(0)
                    st.audio(buf, format="audio/mp3")
                except Exception as e:
                    st.error(f"TTS 생성 실패: {str(e)}")

            st.markdown("---")

# 드롭다운별 뉴스 실행
handle_news_selection(departure, "departure_news", "출발지")
handle_news_selection(transit, "transit_news", "경유지")
handle_news_selection(destination, "destination_news", "목적지")

