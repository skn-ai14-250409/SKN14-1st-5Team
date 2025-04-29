# 출력 값 클래스 변수와 연결
import requests
import mysql.connector

# 시도명 변환
def clean_sido_name(sido_name):
    if sido_name.startswith('충청남도'):
        return '충남'
    elif sido_name.startswith('충청북도'):
        return '충북'
    elif sido_name.startswith('전라남도'):
        return '전남'
    elif sido_name.startswith('전라북도'):
        return '전북'
    elif sido_name.startswith('경상남도'):
        return '경남'
    elif sido_name.startswith('경상북도'):
        return '경북'
    # 소괄호로 튜플로 묶고 해당 값을 보내야 함
    elif sido_name.endswith(('특별자치시', '특별자치도', '특별시', '광역시', '도')):
        return sido_name[:2]
    else:
        return sido_name


# 테이블 엔티티클래스
class Region:
    def __init__(self, region_code, sido_code, sigungu_code, sido_name, sigungu_name):
        self.region_code = region_code
        self.sido_code = sido_code
        self.sigungu_code = sigungu_code
        self.sido_name = sido_name
        self.sigungu_name = sigungu_name

    def __repr__(self):
        return f'Region({self.region_code}, {self.sido_code}, {self.sigungu_code}, {self.sido_name}, {self.sigungu_name})'

url = 'http://apis.data.go.kr/1741000/StanReginCd/getStanReginCdList'

region_list = []

for page in range(1, 208):
    params = {
        'serviceKey' : 'chHDedlO/C/aI5p034BuUga+/D96VJiehvTkS3N5WoxtsKrXuaCHXumBLFLUXz9k2rrMUNJrNz/DAkN/yJu/2A==',
        'pageNo' : page,
        'numOfRows' : '100',
        'type' : 'json'
        }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        item = data.get('StanReginCd', [])

        # 데이터가 제대로 있는지 먼저 체크
        if len(item) >= 2 and 'row' in item[1]:
            rows = item[1]['row'] # head 다음

            for row in rows:
                시도코드 = row.get('sido_cd')
                시군구코드 = row.get('sgg_cd')
                읍면동코드 = row.get('umd_cd')
                지역명 = row.get('locatadd_nm')
                지역코드 = row.get('region_cd')

                if 읍면동코드 == '000':
                    parts = 지역명.split(' ')
                    if len(parts) >= 2:
                        시도명 = clean_sido_name(parts[0])
                        시군구명 = parts[1]
                        region = Region(
                            region_code=지역코드,
                            sido_code=시도코드,
                            sigungu_code=시군구코드,
                            sido_name=시도명,
                            sigungu_name=시군구명
                        )
                        region_list.append(region)
                        print(region)

        else:
            print(f"{page} 페이지: 데이터 없음. 종료합니다.")
            # 207 페이지: 데이터 없음. 종료합니다.

    else:
        print(f'요청 실패: {response.status_code}')
        response.raise_for_status()
print(len(region_list)) # 263개 출력

# 데이터 삽입
config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'skn14',
    'password': 'skn14',
    'database': 'accidentdb',
}

try:
    with mysql.connector.connect(**config) as conn:
        with conn.cursor() as cursor:
            for region in region_list:
                cursor.execute('''
                    insert into tbl_region(지역코드, 시도코드, 시군구코드, 시도명, 시군구명)
                    values (%s, %s, %s, %s, %s)
            ''', (region.region_code, region.sido_code, region.sigungu_code, region.sido_name, region.sigungu_name))
            conn.commit()
except mysql.connector.Error as e:
    print('DB 오류: ', e)
