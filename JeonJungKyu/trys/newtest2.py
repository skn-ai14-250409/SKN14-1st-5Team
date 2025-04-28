import streamlit as st
import mysql.connector
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
import openai
from gtts import gTTS
from io import BytesIO

# ── 1) .env 로드 ─────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPEN_AI_API")

# ── 2) 설정 ─────────────────────────────────────
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/89.0.4389.114 Safari/537.36"
    )
}


# ── 3) 유틸: HTML 전처리 ─────────────────────────
def clean_html(soup: BeautifulSoup):
    for tag in soup.select("script, style, aside, .ad, .related-article"):
        tag.decompose()


# ── 4) 전체 텍스트 가져오기 ─────────────────────
def get_html_text(url: str) -> str:
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        clean_html(soup)
        return soup.get_text(separator="\n")
    except Exception:
        return ""


# ── 5) 요약하기 (공백 정리 추가) ─────────────────
def summarize_text(text: str) -> str:
    cleaned_text = "\n".join(
        line.strip() for line in text.splitlines() if line.strip()
    )

    prompt = f"""
    다음 웹페이지 텍스트를 읽고 중요 기사 내용만 3줄로 요약해줘.
    메뉴, 광고, 댓글, 저작권 문구는 무시해줘.

    {cleaned_text}
    """
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return resp.choices[0].message.content.strip()


# ── 6) Streamlit 앱 ─────────────────────────────
st.title("📰 뉴스 요약 & TTS 변환")

st.caption("네이버 뉴스 원문 링크를 요약하고 음성으로 변환합니다.")

if st.button("📝 뉴스 요약 시작"):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT originallink FROM naver_news")
        urls = [r[0] for r in cur.fetchall()]
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"DB 오류: {e}")
        urls = []

    for idx, url in enumerate(urls, start=1):
        with st.container():
            st.markdown(f"### 🔗 뉴스 #{idx}")
            st.markdown(f"[{url}]({url})")

            article = get_html_text(url)
            if not article:
                st.error("❗ 전체 텍스트 가져오기 실패")
                continue

            summary = summarize_text(article)

            if "무단 전재" in summary or "재배포 금지" in summary:
                st.warning("🚫 해당 기사는 TTS가 불가합니다. (무단 전재 문구 포함)")
                continue

            st.success("✨ 요약 결과")
            st.write(summary)

            tts = gTTS(text=summary, lang="ko")
            buf = BytesIO()
            tts.write_to_fp(buf)

            st.audio(buf.getvalue(), format="audio/mp3")

        st.markdown("---")  # 구분선 추가
