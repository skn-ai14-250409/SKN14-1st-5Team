# 지역테이블(지역코드 포함된)
use accidentdb;
# drop table tbl_region;

create table tbl_region (
    지역코드 int primary key,
    시도코드 int not null,
    시군구코드 int not null,
    시도명 varchar(50) not null,
    시군구명 varchar(50) not null
);
# 지역코드 bigint로 수정
ALTER TABLE tbl_region
MODIFY COLUMN `지역코드` BIGINT;

# 외래키 참조
update tbl_location l
JOIN tbl_region r
ON l.region1 = r.시도명 AND l.region2 = r.시군구명
SET l.지역코드 = r.지역코드;

update tbl_spot s
JOIN tbl_region r
ON s.사고다발지역시도 = r.시도명 AND s.사고다발지역시군구 = r.시군구명
SET s.지역코드 = r.지역코드;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              = r.region_code;


# 중복값 조회
select 시도, 시군구, count(*)
from tbl_region
group by 시도, 시군구;


# 세종시 추가 3611000000
replace into
    tbl_region
values (
        	3611000000, 36, 110,'세종','세종시'
       );


UPDATE tbl_location l
JOIN tbl_region
ON l.region1 = 시도명
AND l.region2 = 시군구명
SET 지역코드 = 지역코드;

# 하나만 남기고 삭제 (12개)
# 경기,수원시
# delete from tbl_region
# where 시도 = '경기' and 시군구 = '수원시'
# limit 1;
# # 경기,성남시
# delete from tbl_region
# where 시도 = '경기' and 시군구 = '성남시'
# limit 1;
# # 경기,안양시
# delete from tbl_region
# where 시도 = '경기' and 시군구 = '안양시'
# limit 1;
# # 경기,부천시
# delete from tbl_region
# where 시도 = '경기' and 시군구 = '부천시'
# limit 1;
# # 경기,안산시
# delete from tbl_region
# where 시도 = '경기' and 시군구 = '안산시'
# limit 1;
# # 경기,고양시
# delete from tbl_region
# where 시도 = '경기' and 시군구 = '고양시'
# limit 1;
# # 경기,용인시
# delete from tbl_region
# where 시도 = '경기' and 시군구 = '용인시'
# limit 1;
# # 충북,청주시
# delete from tbl_region
# where 시도 = '충북' and 시군구 = '청주시'
# limit 1;
# # 충남,천안시
# delete from tbl_region
# where 시도 = '충남' and 시군구 = '천안시'
# limit 1;
# # 경북,포항시
# delete from tbl_region
# where 시도 = '경북' and 시군구 = '포항시'
# limit 1;
# # 경남,창원시
# delete from tbl_region
# where 시도 = '경남' and 시군구 = '창원시'
# limit 1;
# # 전북,전주시
# delete from tbl_region
# where 시도 = '전북' and 시군구 = '전주시'
# limit 1;

# 중복값 허용안함
alter table tbl_region
add constraint unique (시도, 시군구);
