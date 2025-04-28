import streamlit as st
import mysql.connector
from newspaper import Article
import openai
from gtts import gTTS
import os
from dotenv import load_dotenv
# OpenAI API 키 설정

load_dotenv()
openai.api_key = os.getenv("OPEN_AI_API")
# MySQL DB 연결 정보
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}
# 기사 본문 가져오기
def get_article_text(url):
    article = Article(url, language='ko')
    article.download()
    article.parse()
    return article.text

# GPT로 요약 (OpenAI SDK ≥ 1.0.0 대응)

def summarize_text(text):
    prompt = f"다음 기사를 3줄로 요약해줘:\n\n{text}"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    # SDK 1.x 이후 응답 파싱
    summary = response.choices[0].message.content
    return summary

st.title("📰 뉴스 요약 + TTS 서비스")

if st.button("뉴스 요약 시작하기"):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT originallink FROM naver_news")
        urls = cursor.fetchall()

        for idx, (url,) in enumerate(urls):
            if not url:
                continue

            st.subheader(f"원본 링크 #{idx+1}")
            st.write(url)

            try:
                # 1) 본문 가져오기
                article_text = get_article_text(url)

                # 2) 요약
                summary = summarize_text(article_text)

                st.subheader("요약 결과")
                st.write(summary)

                # 3) TTS로 음성 생성
                tts = gTTS(text=summary, lang='ko')
                audio_path = f"temp_{idx}.mp3"
                tts.save(audio_path)

                # 4) 스트림릿에 오디오 재생
                with open(audio_path, "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/mp3")

                # 5) 임시 파일 삭제
                os.remove(audio_path)

            except Exception as e:
                st.error(f"링크 처리 중 오류 발생: {e}")

        cursor.close()
        conn.close()

    except mysql.connector.Error as e:
        st.error(f"DB 연결 오류: {e}")