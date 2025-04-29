create table if not exists car_accident_naver_news
(
    id                 int auto_increment
        primary key,
    accident_date      varchar(50) null,
    accident_location1 varchar(50) null,
    accident_location2 varchar(50) null,
    title              text        null,
    description        text        null,
    link               text        null,
    content            longtext    null,
    accident_type      varchar(50) null
);


