use accidentdb;
show tables;


# tbl_date
create table tbl_date(
    id int primary key, # auto 사용x
    datetime datetime not null,
    daynight varchar(10) not null,
    weekday varchar(10) not null
);

insert into tbl_date
select id, datetime, daynight, weekday
from car_accident;


# tbl_location
# drop table tbl_location;
create table tbl_location(
  id int primary key,
  region1 varchar(10),
  region2 varchar(20)
);

insert into tbl_location
select id, region1, region2
from car_accident;

alter table tbl_location
add column `지역코드` bigint;

# 충남,연기군-> 폐지 경북,군위군-> 지역코드없음 충북,청원군->청주시 인천,남구->미추홀구
update tbl_location
set region2 = '청주시'
where region1 = '충북' and region2 = '청원군';

update tbl_location
set region2 = '미추홀구'
where region1 = '인천' and region2 = '남구';

delete from tbl_location
where 지역코드 is null;



# tbl_type
create table tbl_type(
    id int primary key,
    type_main varchar(50),
    type_sub varchar(50),
    law varchar(100)
);

insert into tbl_type
select id, type_main, type_sub, law
from car_accident;



# tbl_damage
create table tbl_damage(
    id int primary key,
    dead int default 0,
    hurt int default 0
);

insert into tbl_damage
select id, dead, hurt
from car_accident;

UPDATE tbl_location
SET region2 = REPLACE(region2, '(통합)', '')
WHERE region2 LIKE '%(통합)%';





