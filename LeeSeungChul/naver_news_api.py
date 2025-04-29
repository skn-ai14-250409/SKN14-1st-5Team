# --- 필요한 모듈 불러오기 ---
import os
from dotenv import load_dotenv
import requests
import mysql.connector
from kor_location_map import find_location
from datetime import datetime
import time

# --- .env 파일에서 API 키 불러오기 ---
load_dotenv()

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# --- 사고유형 키워드 목록 ---
accident_keywords = ["음주", "과속", "신호위반", "보복운전", "무면허", "무단횡단"]

# --- 수집할 키워드 목록 ---
keywords = [
    "교통사고",
    "음주운전",
    "신호위반",
    "과속사고",
    "보복운전",
    "무면허운전",
    "무단횡단 사고"
]

# --- 네이버 뉴스 API 호출 함수 ---
def crawl_naver_news(keyword, start_num, max_count=100):
    url = "https://openapi.naver.com/v1/search/news.json"

    params = {
        "query": keyword,
        "display": max_count,
        "start": start_num,
        "sort": "date"
    }

    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        result = response.json()
        items = result.get("items", [])

        news_list = []
        for item in items:
            title = item.get("title", "").replace("<b>", "").replace("</b>", "")
            link = item.get("link", "")
            description = item.get("description", "").replace("<b>", "").replace("</b>", "")
            pubDate = item.get("pubDate", "")  # 발행일

            # pubDate를 accident_date로 변환
            try:
                pub_date_obj = datetime.strptime(pubDate, '%a, %d %b %Y %H:%M:%S %z')
                accident_date = pub_date_obj.strftime('%Y-%m-%d')
            except Exception as e:
                print(f"❗ 날짜 변환 실패: {e}")
                accident_date = None

            news_list.append({
                "title": title,
                "link": link,
                "description": description,
                "accident_date": accident_date
            })

        return news_list
    else:
        print(f"❗ 에러 발생: {response.status_code}")
        return []

# --- 사고 유형 찾기 함수 ---
def find_accident_type(text):
    for keyword in accident_keywords:
        if keyword in text:
            return keyword
    return None

# --- 완성된 뉴스만 데이터베이스에 저장하는 함수 ---
def save_news_to_db(news_list):
    config = {
        "host": "localhost",
        "user": "team5",
        "password": "teamteam5",
        "database": "team5_database"
    }

    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()

    inserted_count = 0

    for news in news_list:
        title = news["title"]
        link = news["link"]
        description = news["description"]
        accident_date = news["accident_date"]

        combined_text = title + " " + description

        loc1, loc2 = find_location(combined_text)
        accident_type = find_accident_type(combined_text)

        # 🔥 location1, location2, accident_type, accident_date가 모두 있을 때만 저장
        if loc1 and loc2 and accident_type and accident_date:
            sql = """
                INSERT INTO car_accident_naver_news 
                (title, link, description, accident_location1, accident_location2, accident_type, accident_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            values = (title, link, description, loc1, loc2, accident_type, accident_date)

            try:
                cursor.execute(sql, values)
                inserted_count += 1
            except Exception as e:
                print(f"❗ 저장 실패: {title} 에러: {e}")

    conn.commit()
    print(f"✅ DB에 저장된 뉴스 수: {inserted_count}개")

    cursor.close()
    conn.close()

# --- 메인 실행 ---
if __name__ == "__main__":
    all_news = []

    for keyword in keywords:
        for page in range(10):  # 키워드당 5페이지 (100개 * 5 = 최대 500개)
            start_num = page * 100 + 1
            news_list = crawl_naver_news(keyword, start_num, max_count=100)
            if not news_list:
                break
            all_news.extend(news_list)

    print(f"총 수집한 뉴스 개수(가공 전): {len(all_news)}개")

    # 가공해서 저장
    save_news_to_db(all_news)
