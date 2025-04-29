# 폴리곤 테이블 생성

# use car_accident;
show tables;

# ALTER TABLE accident_info ADD COLUMN 사고다발지역시도 VARCHAR(20);
# ALTER TABLE accident_info ADD COLUMN 사고다발지역시군구 VARCHAR(50);
# ALTER TABLE accident_info ADD COLUMN 사고다발지역폴리곤정보 TEXT;

# insert into accident_info
# select 사고다발지역시도, 사고다발지역시군구, 사고다발지역폴리곤정보
# from accident_info;

# 폴리곤 테이블
# drop table tbl_spot;

create table tbl_spot(
  사고다발지역시도 varchar(10),
  사고다발지역시군구 varchar(20),
  위도 text,
  경도 text,
  사고다발지역폴리곤정보 text
);
alter table tbl_spot
add column 지역코드 bigint;

-- 1. Safe Updates 끄기
SET SQL_SAFE_UPDATES = 0;

-- 2. 지역코드 UPDATE
UPDATE tbl_spot s
JOIN tbl_region
ON s.시도 = 시도명
AND s.시군구 = 시군구명
SET s.지역코드 = 지역코드;

-- 3. 다시 Safe Updates 켜기 (선택)
SET SQL_SAFE_UPDATES = 1;

UPDATE tbl_spot
SET 시도 = '경기'
WHERE 시도 = '경기도';







