-- 1. 기존 team5_database가 있으면 삭제합니다.
DROP DATABASE IF EXISTS team5_database;

-- 2. 새로 team5_database를 생성합니다.
CREATE DATABASE team5_database;

-- 3. team5 사용자 계정을 생성합니다. (비밀번호는 'teamteam5')
CREATE USER IF NOT EXISTS 'team5'@'%' IDENTIFIED BY 'teamteam5';

-- 4. team5 사용자에게 team5_database에 대한 모든 권한을 부여합니다.
GRANT ALL PRIVILEGES ON team5_database.* TO 'team5'@'%';

-- 5. 권한 설정을 적용합니다.
FLUSH PRIVILEGES;

-- 6. team5_database를 사용하겠다고 선언합니다.
USE team5_database;

-- 7. car_accident_naver_news 테이블을 생성합니다.
CREATE TABLE IF NOT EXISTS car_accident_naver_news (
    id INT AUTO_INCREMENT PRIMARY KEY,        --고유번호 (자동 증가)
    accident_date VARCHAR(50),                 --사고 날짜 (문자열로 저장, 예: 2025-04-25)
    accident_location1 VARCHAR(50),            -- 시/도 (예: 서울특별시)
    accident_location2 VARCHAR(50),            -- 구/군 (예: 강남구)
    title TEXT,                                --  뉴스 제목
    description TEXT,                          --  뉴스 요약 (API에서 받아오는 요약글)
    link TEXT,                                 -- 네이버 뉴스 링크
    content LONGTEXT,                          -- 뉴스 전체 본문 (크롤링해서 가져올 경우 저장)
    accident_type VARCHAR(50)                  -- 사고 유형 (예: 음주운전, 과속, 신호위반 등)
);

delete  from car_accident_naver_news; # id값은 살아있음

TRUNCATE TABLE car_accident_naver_news; # id까지 모두 지우고 리셋

# 지역과 유형이 있는 컬럼조회
SELECT count(*)
FROM car_accident_naver_news
WHERE accident_location1 IS NOT NULL     -- 지역(accident_location1)이 NULL이 아니고 (값이 존재)
  AND accident_location1 != ''
  and accident_location2 is not null
  and accident_location2 !=''-- 지역(accident_location1)이 빈 문자열이 아니고 ("" 아닌)
  AND accident_type IS NOT NULL            -- 사고유형(accident_type)이 NULL이 아니고 (값이 존재)
  AND accident_type != '';

# 갯수도 조회해보고
select count(*)
from car_accident_naver_news;

# 내용도 보고
select *
from car_accident_naver_news;

