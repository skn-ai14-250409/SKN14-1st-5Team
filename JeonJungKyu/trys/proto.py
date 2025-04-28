
import mysql.connector
import streamlit as st
import pandas as pd
import mysql.connector
from gtts import gTTS
import tempfile
import os
# MySQL 연결 설정
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}
# 데이터 불러오기 함수
def load_data():
    conn = mysql.connector.connect(**config)
    query = "SELECT * FROM naver_news"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Streamlit 페이지 구성
st.title("📊 네이버 뉴스 데이터 보기")

# 데이터 로딩
df = load_data()

# 표 출력
st.dataframe(df)

# TTS 음성 생성 함수
def text_to_speech(text):
    tts = gTTS(text=text, lang='ko')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# Streamlit UI
st.title("🗞️ 네이버 뉴스 - 제목 음성 읽기")

df = load_data()

# title 컬럼만 보여주고 버튼 추가
for i, row in df.iterrows():
    st.write(f"**{row['title']}**")
    if st.button(f"🔊 읽기 {i}", key=i):
        audio_file = text_to_speech(row['title'])
        audio_bytes = open(audio_file, 'rb').read()
        st.audio(audio_bytes, format='audio/mp3')
        os.remove(audio_file)  # 임시파일 삭제