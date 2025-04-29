import streamlit as st
import pandas as pd
import folium
from dotenv import load_dotenv
import os
import mysql.connector
from datetime import datetime

# 환경 변수 로드
load_dotenv()

# DB 연결 함수
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

# region1에 해당하는 region2 값을 불러오는 함수
def get_region2_options(region1):
    conn = get_db_connection()
    query = f"""
    SELECT DISTINCT region2
    FROM car_accident a
    JOIN tbl_spot s ON a.region1 = s.사고다발지역시도
    WHERE a.region1 = '{region1}'
    """
    region2_list = pd.read_sql(query, conn)
    conn.close()
    return region2_list['region2'].tolist()

# 사고 데이터 불러오기 함수 (2012년 5월 한정)
def get_seoul_accident_data(region1, region2, start_date, end_date):
    conn = get_db_connection()
    query = f"""
    SELECT a.id, a.datetime, a.daynight, a.weekday, a.dead, a.hurt, a.region1, a.region2,
           s.위도 AS lat, s.경도 AS lon
    FROM car_accident a
    JOIN tbl_spot s ON a.region1 = s.사고다발지역시도
    WHERE a.region1 = '{region1}' AND a.region2 = '{region2}'
    AND a.datetime BETWEEN '{start_date}' AND '{end_date}'
    LIMIT 100
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Streamlit 제목
st.title('🚗 교통사고 다발지역 지도 (folium 버전)')
st.caption("2012년 5월에 발생한 사고를 기반으로 사고 규모가 시각화됩니다. 안전운전하세요!")

# 지역1 선택
region1_options = ['서울', '부산', '대구', '인천']  # 예시로 서울, 부산, 대구, 인천 선택지 추가
region1 = st.selectbox('지역을 선택하세요', region1_options)

# 지역2 선택 (region1에 따라 동적으로 변경)
if region1:
    region2_options = get_region2_options(region1)
    region2 = st.selectbox('세부 지역을 선택하세요', region2_options)

    # 고정된 날짜 범위 (2012년 5월)
    start_date = '2012-05-01'
    end_date = '2012-05-31'

    # 데이터 로드 (스피너 추가)
    with st.spinner("🚦 사고 데이터를 불러오는 중입니다..."):
        df = get_seoul_accident_data(region1, region2, start_date, end_date)
    st.success("✅ 데이터 불러오기 완료!")

    # folium 지도 생성
    my_map = folium.Map(location=[37.5665, 126.9780], zoom_start=11)

    # 사고 데이터 지도에 추가
    for _, row in df.iterrows():
        size = 5 + row['dead'] * 5 + row['hurt']  # 반지름 설정

        # 원형 마커
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=size,
            color='crimson',
            fill=True,
            fill_opacity=0.8
        ).add_to(my_map)

        # 숫자만 (사망자 수, 부상자 수) 표시하는 텍스트 마커
        folium.Marker(
            location=[row['lat'], row['lon']],
            icon=folium.DivIcon(
                html=f"""<div style='font-size:10pt; color:black'>
                        ({row['dead']},{row['hurt']})
                        </div>"""
            )
        ).add_to(my_map)

    # Streamlit에 지도 표시
    st.components.v1.html(my_map._repr_html_(), height=600)
