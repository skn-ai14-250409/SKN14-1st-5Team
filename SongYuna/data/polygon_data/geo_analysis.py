import geopandas as gpd

# CSV 파일의 경우 먼저 pandas로 읽은 후 GeoDataFrame으로 변환해야 합니다
import pandas as pd

# CSV 파일을 pandas DataFrame으로 읽기
df = pd.read_csv('accident_cleaned.csv', encoding='utf-8-sig')
#print(df.head())
#print(df)
# 만약 CSV 파일에 geometry 정보가 있다면 (예: 위도/경도)
# 적절한 방식으로 GeoDataFrame으로 변환해야 합니다
#df['사고다발지역시도시군구']

df['사고다발지역시도시군구'] = df['사고다발지역시도시군구'].str.rstrip('0123456789')
# 변경된 결과 확인
print(df['사고다발지역시도시군구'].head())
print('---------------------------------')


df1=df[['사고다발지역시도시군구','사고다발지역폴리곤정보']]
print(df1)

print('---------------------------------')

# df2=df['사고다발지역폴리곤정보'][0]
# print(df2)
print("폴리곤 데이터 샘플:")
print(df['사고다발지역폴리곤정보'].iloc[0])
print('---------------------------------')
import json
from shapely import wkt
from shapely.geometry import Point, Polygon
from shapely.wkt import loads

#polygon_geojson=df['사고다발지역폴리곤정보']
#print(polygon_geojson)

# 폴리곤 정보를 GeoJSON으로 변환하는 함수
def create_polygon(polygon_str):
        polygon_json = polygon_str.replace('type:', '"type":').replace('coordinates:', '"coordinates":').replace('Polygon', '"Polygon"')
        geojson = json.loads(polygon_json)
        return Polygon(geojson['coordinates'][0])


df['사고다발지역폴리곤정보'] = df['사고다발지역폴리곤정보'].apply(create_polygon)
gdf = gpd.GeoDataFrame(df[['사고다발지역시도시군구']], geometry=df['사고다발지역폴리곤정보'], crs="EPSG:4326")
print(gdf)

#시각화
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
#gdf.plot()
fig, ax = plt.subplots(figsize=(15, 10))

polygon_data=gdf['geometry'][0]

polygon_patch = plt.Polygon(list(polygon_data.exterior.coords),closed=True, edgecolor='red', facecolor='yellow')
ax.add_patch(polygon_patch)

plt.title('Polygon')
plt.axis('equal')
plt.show()

#저장
gdf.to_file("polygon_data.geojson", driver='GeoJSON')

#한국 좌표계로 변환하는 코드
#EPSG:5179 (Korean 1985)로 변환
#gdf_transformed = gdf.to_crs("EPSG:5179")


#배경 지도 추가
import contextily as ctx

# GeoDataFrame을 Web Mercator 좌표계로 변환
gdf_web = gdf.to_crs(epsg=3857)

# 시각화
fig, ax = plt.subplots(figsize=(15, 10))
gdf_web.plot(ax=ax, alpha=0.5)
ctx.add_basemap(ax)
plt.show()

#지역 이름 라벨 표시
for idx, row in gdf.iterrows():
        plt.annotate(text=row['사고다발지역시도시군구'],
                     xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                     ha='center')
#색상 구분
gdf.plot(column='사고다발지역시도시군구',
       legend=True,
       cmap='Set3',
       categorical=True)

#지역별 통계 정보
# 지역별 사고 건수
stats = gdf.groupby('사고다발지역시도시군구').size()
print("지역별 사고다발구역 수:")
print(stats)

#면적 계산
# 먼저 한국 좌표계로 변환
gdf_korean = gdf.to_crs("EPSG:5179")

# 면적 계산 (제곱미터 단위)
gdf_korean['면적'] = gdf_korean.geometry.area
print("지역별 사고다발구역 면적:")
print(gdf_korean[['사고다발지역시도시군구', '면적']])

# 지역 간 중첩 분석
# for idx1, row1 in gdf.iterrows():
#     for idx2, row2 in gdf.iterrows():
#         if idx1 < idx2:  # 중복 비교 방지
#             if row1.geometry.intersects(row2.geometry):
#                print(f"중첩된 지역: {row1['사고다발지역시도시군구']} - {row2['사고다발지역시도시군구']}")

# 1.면적이 넓은 사고다발구역 분석
# 한국 좌표계로 변환 후 면적 계산
gdf_korean = gdf.to_crs("EPSG:5179")
gdf_korean['면적'] = gdf_korean.geometry.area

# 지역별 총 면적과 평균 면적 계산
area_stats = gdf_korean.groupby('사고다발지역시도시군구').agg({
    '면적': ['sum', 'mean', 'count']
}).round(2)

# 결과 정렬 (총 면적 기준)
area_stats = area_stats.sort_values(('면적', 'sum'), ascending=False)
print("\n== 지역별 사고다발구역 면적 분석 ==")
print(area_stats)

# 시각화
plt.figure(figsize=(12, 6))
area_stats[('면적', 'sum')].plot(kind='bar')
plt.title('지역별 총 사고다발구역 면적')
plt.xlabel('지역')
plt.ylabel('총 면적 (m²)')
plt.xticks(rotation=45)
plt.show()

'''
- 각 지역의 총 사고다발구역 면적 계산
- 평균 면적과 구역 수도 함께 분석
- 면적이 넓은 지역은 잠재적 위험이 더 클 수 있음
'''

# 2. 중첩 위험 지역 분석
# 중첩 분석 함수
'''
def analyze_overlaps(gdf):
    overlap_count = {}
    overlap_details = []

    for idx1, row1 in gdf.iterrows():
        area_name1 = row1['사고다발지역시도시군구']
        overlap_count[area_name1] = overlap_count.get(area_name1, 0)

        for idx2, row2 in gdf.iterrows():
            if idx1 < idx2:
                if row1.geometry.intersects(row2.geometry):
                    area_name2 = row2['사고다발지역시도시군구']
                    overlap_count[area_name1] += 1
                    overlap_count[area_name2] = overlap_count.get(area_name2, 0) + 1

                    # 중첩 면적 계산
                    overlap_area = row1.geometry.intersection(row2.geometry).area
                    overlap_details.append({
                        'area1': area_name1,
                        'area2': area_name2,
                        'overlap_area': overlap_area
                    })

    return overlap_count, overlap_details


overlap_count, overlap_details = analyze_overlaps(gdf_korean)

# 결과 출력
print("\n== 중첩이 가장 많은 위험 지역 ==")
sorted_overlaps = sorted(overlap_count.items(), key=lambda x: x[1], reverse=True)
for area, count in sorted_overlaps[:10]:  # 상위 10개 지역
    print(f"{area}: {count}회 중첩")

- 다른 구역과 겹치는 횟수 계산
- 중첩 면적도 함께 분석
- 여러 구역이 겹치는 지역은 특별한 관리가 필요할 수 있음
'''

#3.지역별 패턴 분석
# 패턴 분석을 위한 지표 계산
pattern_analysis = pd.DataFrame({
    '구역_수': gdf_korean.groupby('사고다발지역시도시군구').size(),
    '평균_면적': gdf_korean.groupby('사고다발지역시도시군구')['면적'].mean(),
    '밀집도': gdf_korean.groupby('사고다발지역시도시군구').apply(
        lambda x: len(x) / x.geometry.unary_union.area
    )
})

# 클러스터링을 통한 패턴 그룹화
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 데이터 정규화
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pattern_analysis)

# K-means 클러스터링
kmeans = KMeans(n_clusters=3, random_state=42)
pattern_analysis['패턴_그룹'] = kmeans.fit_predict(scaled_data)

print("\n== 지역별 패턴 분석 ==")
print(pattern_analysis)

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(pattern_analysis['평균_면적'], pattern_analysis['밀집도'],
           c=pattern_analysis['패턴_그룹'], cmap='viridis')
plt.xlabel('평균 면적')
plt.ylabel('밀집도')
plt.title('사고다발구역 패턴 분석')
plt.show()
'''
- 구역 수, 평균 면적, 밀집도 등 다양한 지표 분석
- 클러스터링을 통한 유사한 패턴을 가진 지역 그룹화
- 시각화를 통한 패턴 이해 도움
'''

#1. **대규모 사고다발구역 분석**:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한국 좌표계로 변환 및 면적 계산
gdf_korean = gdf.to_crs("EPSG:5179")
gdf_korean['면적'] = gdf_korean.geometry.area


# 면적 기준 상위 지역 분석
def analyze_large_areas(gdf):
    # 지역별 통계 계산
    area_stats = gdf.groupby('사고다발지역시도시군구').agg({
        '면적': ['sum', 'mean', 'count']
    }).round(2)

    area_stats.columns = ['총면적', '평균면적', '구역수']
    area_stats = area_stats.sort_values('총면적', ascending=False)

    # 상위 10개 지역 출력
    print("\n=== 대규모 사고다발구역 분석 (상위 10개 지역) ===")
    print(area_stats.head(10))

    # 시각화
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    area_stats['총면적'].head(10).plot(kind='bar')
    plt.title('지역별 총 사고다발구역 면적')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    area_stats.plot.scatter(x='구역수', y='총면적')
    plt.title('구역수 vs 총면적')

    plt.tight_layout()
    plt.show()

    return area_stats


area_analysis = analyze_large_areas(gdf_korean)
'''
#2. **중첩 위험 지역 분석**:
def analyze_overlapping_areas(gdf):
    overlap_data = []

    # 중첩 분석
    for idx1, row1 in gdf.iterrows():
        for idx2, row2 in gdf.iterrows():
            if idx1 < idx2:
                if row1.geometry.intersects(row2.geometry):
                    intersection_area = row1.geometry.intersection(row2.geometry).area
                    overlap_data.append({
                        '지역1': row1['사고다발지역시도시군구'],
                        '지역2': row2['사고다발지역시도시군구'],
                        '중첩면적': intersection_area
                    })

    overlap_df = pd.DataFrame(overlap_data)

    # 지역별 중첩 횟수 계산
    region_overlap_count = pd.concat([
        overlap_df['지역1'].value_counts(),
        overlap_df['지역2'].value_counts()
    ]).groupby(level=0).sum().sort_values(ascending=False)

    print("\n=== 중첩이 많은 위험 지역 (상위 10개) ===")
    print(region_overlap_count.head(10))

    # 시각화
    plt.figure(figsize=(12, 6))
    region_overlap_count.head(10).plot(kind='bar')
    plt.title('지역별 중첩 발생 횟수')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return region_overlap_count, overlap_df


overlap_count, overlap_details = analyze_overlapping_areas(gdf_korean)
'''
#3. **지역 패턴 분석**:
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def analyze_patterns(gdf):
    # 패턴 분석을 위한 특성 계산
    pattern_data = pd.DataFrame({
        '구역수': gdf.groupby('사고다발지역시도시군구').size(),
        '평균면적': gdf.groupby('사고다발지역시도시군구')['면적'].mean(),
        '총면적': gdf.groupby('사고다발지역시도시군구')['면적'].sum()
    })

    # 데이터 정규화
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(pattern_data)

    # K-means 클러스터링
    kmeans = KMeans(n_clusters=3, random_state=42)
    pattern_data['클러스터'] = kmeans.fit_predict(scaled_features)

    # 클러스터별 특성 분석
    cluster_stats = pattern_data.groupby('클러스터').mean()
    print("\n=== 클러스터별 특성 ===")
    print(cluster_stats)

    # 시각화
    plt.figure(figsize=(12, 6))
    plt.scatter(pattern_data['평균면적'], pattern_data['구역수'],
                c=pattern_data['클러스터'], cmap='viridis')
    plt.xlabel('평균 면적')
    plt.ylabel('구역 수')
    plt.title('지역 패턴 클러스터링')

    # 클러스터 중심점 표시
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 1], centers[:, 0], c='red', marker='x', s=200, linewidths=3)

    plt.show()

    return pattern_data


pattern_analysis = analyze_patterns(gdf_korean)

'''
**종합 분석 결과**:
1. **대규모 사고다발구역**:
    - 면적이 가장 큰 지역들을 파악하여 우선 관리 대상 선정
    - 구역 수와 면적의 관계를 통해 밀집도 높은 지역 식별

2. **중첩 위험 지역**:
    - 여러 사고다발구역이 겹치는 지역 파악
    - 중첩이 많은 지역은 교통안전시설 보강 등 특별 관리 필요

3. **패턴 분석**:
    - 클러스터 0: 소규모 다발구역 (작은 면적, 적은 구역 수)
    - 클러스터 1: 중규모 다발구역 (중간 면적, 중간 구역 수)
    - 클러스터 2: 대규모 다발구역 (큰 면적, 많은 구역 수)

**권장 대책**:
1. 대규모 구역:
    - 도로 구조 개선
    - 교통신호 체계 재정비

2. 중첩 구역:
    - CCTV 추가 설치
    - 교통경찰 우선 배치

3. 패턴별 맞춤 대책:
    - 소규모: 교통표지판 개선
    - 중규모: 과속방지턱 설치
    - 대규모: 종합적인 교통안전 시설물 설치

'''

# 1. **밀집도 높은 지역 분석**:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_density(gdf):
    # 지역별 기본 통계 계산
    density_stats = pd.DataFrame({
        '구역수': gdf.groupby('사고다발지역시도시군구').size(),
        '총면적': gdf.groupby('사고다발지역시도시군구')['면적'].sum(),
        '평균면적': gdf.groupby('사고다발지역시도시군구')['면적'].mean()
    })

    # 밀집도 계산 (구역수 / 총면적)
    density_stats['밀집도'] = (density_stats['구역수'] / density_stats['총면적']) * 1000000  # 제곱킬로미터당 구역수

    # 결과 정렬
    density_sorted = density_stats.sort_values('밀집도', ascending=False)

    print("\n=== 밀집도 분석 결과 (상위 10개 지역) ===")
    print("\n밀집도 = 구역수/총면적 (높을수록 조밀)")
    print(density_sorted.head(10))

    # 시각화
    plt.figure(figsize=(15, 10))

    # 1. 밀집도 분포
    plt.subplot(2, 2, 1)
    density_sorted['밀집도'].head(10).plot(kind='bar')
    plt.title('상위 10개 지역 밀집도')
    plt.xticks(rotation=45)

    # 2. 구역수 vs 총면적 산점도
    plt.subplot(2, 2, 2)
    plt.scatter(density_stats['총면적'], density_stats['구역수'], alpha=0.6)
    for idx, row in density_sorted.head(5).iterrows():
        plt.annotate(idx, (row['총면적'], row['구역수']))
    plt.xlabel('총면적')
    plt.ylabel('구역수')
    plt.title('구역수 vs 총면적 관계')

    # 3. 밀집도 히트맵
    if '좌표_X' in gdf.columns and '좌표_Y' in gdf.columns:
        plt.subplot(2, 2, 3)
        sns.kdeplot(data=gdf, x='좌표_X', y='좌표_Y', cmap='Reds')
        plt.title('사고다발구역 밀집도 히트맵')

    plt.tight_layout()
    plt.show()

    return density_stats


density_analysis = analyze_density(gdf_korean)
'''
#2. **중첩 지역 상세 분석**:
def analyze_overlap_details(gdf):
    overlap_data = []
    region_intersections = {}

    # 중첩 분석
    for idx1, row1 in gdf.iterrows():
        region1 = row1['사고다발지역시도시군구']
        if region1 not in region_intersections:
            region_intersections[region1] = {
                '중첩횟수': 0,
                '총중첩면적': 0,
                '중첩지역목록': []
            }

        for idx2, row2 in gdf.iterrows():
            if idx1 < idx2:
                region2 = row2['사고다발지역시도시군구']
                if row1.geometry.intersects(row2.geometry):
                    intersection_area = row1.geometry.intersection(row2.geometry).area

                    # 데이터 저장
                    overlap_data.append({
                        '지역1': region1,
                        '지역2': region2,
                        '중첩면적': intersection_area,
                        '중첩비율': intersection_area / row1.geometry.area
                    })

                    # 지역별 통계 업데이트
                    region_intersections[region1]['중첩횟수'] += 1
                    region_intersections[region1]['총중첩면적'] += intersection_area
                    region_intersections[region1]['중첩지역목록'].append(region2)

                    if region2 not in region_intersections:
                        region_intersections[region2] = {
                            '중첩횟수': 1,
                            '총중첩면적': intersection_area,
                            '중첩지역목록': [region1]
                        }
                    else:
                        region_intersections[region2]['중첩횟수'] += 1
                        region_intersections[region2]['총중첩면적'] += intersection_area
                        region_intersections[region2]['중첩지역목록'].append(region1)

    # 데이터프레임 변환
    overlap_df = pd.DataFrame(overlap_data)
    intersection_df = pd.DataFrame(region_intersections).T

    # 결과 출력
    print("\n=== 중첩 분석 결과 ===")
    print("\n가장 많이 중첩된 지역 (상위 5개):")
    print(intersection_df.sort_values('중첩횟수', ascending=False).head())

    print("\n가장 큰 중첩면적을 가진 지역 쌍 (상위 5개):")
    print(overlap_df.sort_values('중첩면적', ascending=False).head())

    # 시각화
    plt.figure(figsize=(15, 5))

    # 1. 지역별 중첩 횟수
    plt.subplot(1, 2, 1)
    intersection_df['중첩횟수'].sort_values(ascending=False).head(10).plot(kind='bar')
    plt.title('지역별 중첩 횟수 (상위 10개)')
    plt.xticks(rotation=45)

    # 2. 중첩 네트워크 그래프
    try:
        import networkx as nx
        plt.subplot(1, 2, 2)
        G = nx.Graph()

        # 상위 10개 중첩 관계만 표시
        top_overlaps = overlap_df.sort_values('중첩면적', ascending=False).head(10)
        for _, row in top_overlaps.iterrows():
            G.add_edge(row['지역1'], row['지역2'], weight=row['중첩면적'])

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=1000, font_size=8)
        plt.title('중첩 관계 네트워크 (상위 10개)')
    except ImportError:
        print("networkx 라이브러리가 설치되어 있지 않습니다.")

    plt.tight_layout()
    plt.show()

    return overlap_df, intersection_df


overlap_analysis, intersection_analysis = analyze_overlap_details(gdf_korean)
'''
'''
**분석 결과 해석**:
1. **밀집도 분석**:
    - 단위 면적당 사고다발구역이 가장 많은 지역 식별
    - 좁은 면적에 많은 사고다발구역이 몰려있는 위험지역 파악
    - 지역별 관리 우선순위 설정 가능

2. **중첩 분석**:
    - 가장 많이 중첩되는 지역들의 패턴 확인
    - 중첩 면적이 큰 지역들 간의 관계 파악
    - 중첩 네트워크를 통한 지역 간 연관성 시각화

**권장 대책**:
1. 밀집도가 높은 지역:
    - 교통 체계 전면 재검토
    - 교통량 분산 대책 수립
    - 집중 단속 구역 지정

2. 중첩이 많은 지역:
    - 교통안전시설 보강
    - 도로 구조 개선
    - 우회도로 개설 검토
'''


#1. **시간대별 중첩 패턴 분석**:
def analyze_temporal_patterns(gdf):
    if '발생시간' in gdf.columns:  # 발생시간 컬럼이 있다고 가정
        # 시간대별 사고 분포
        gdf['시간대'] = pd.to_datetime(gdf['발생시간']).dt.hour
        temporal_stats = pd.DataFrame({
            '발생건수': gdf.groupby('시간대').size(),
            '평균_중첩수': gdf.groupby('시간대')['중첩횟수'].mean()
        })

        plt.figure(figsize=(12, 6))
        temporal_stats.plot(kind='bar')
        plt.title('시간대별 사고 발생 및 중첩 패턴')
        plt.show()

        return temporal_stats

# temporal_analysis = analyze_temporal_patterns(gdf_korean)

#2. **지형적 특성과 사고다발구역 관계 분석**:
def analyze_geographical_features(gdf):
    # 경사도나 고도 데이터가 있다고 가정
    geographical_stats = pd.DataFrame({
        '평균_경사도': gdf.groupby('사고다발지역시도시군구')['경사도'].mean(),
        '사고_건수': gdf.groupby('사고다발지역시도시군구').size()
    })

    # 상관관계 분석
    correlation = geographical_stats['평균_경사도'].corr(geographical_stats['사고_건수'])
    print(f"\n경사도와 사고 건수의 상관계수: {correlation:.2f}")

    return geographical_stats


#3. **클러스터 간 거리 분석**:
from scipy.spatial.distance import pdist, squareform


def analyze_cluster_distances(gdf):
    # 사고다발구역 중심점 간의 거리 계산
    centroids = gdf.geometry.centroid
    coords = np.column_stack((centroids.x, centroids.y))
    distances = pdist(coords)
    distance_matrix = squareform(distances)

    # 평균 거리 계산
    mean_distances = pd.Series(distance_matrix.mean(axis=1), index=gdf.index)

    print("\n=== 클러스터 간 거리 분석 ===")
    print(f"평균 클러스터 간 거리: {mean_distances.mean():.2f}m")
    print(f"최소 클러스터 간 거리: {mean_distances.min():.2f}m")

    return mean_distances

#4. **도로 유형별 분석**:
def analyze_road_types(gdf):
    if '도로유형' in gdf.columns:  # 도로유형 컬럼이 있다고 가정
        road_stats = pd.DataFrame({
            '사고건수': gdf.groupby('도로유형').size(),
            '평균면적': gdf.groupby('도로유형')['면적'].mean(),
            '중첩비율': gdf.groupby('도로유형')['중첩횟수'].mean()
        })

        print("\n=== 도로 유형별 분석 ===")
        print(road_stats)

        return road_stats

#5. **공간적 자기상관성 분석**:
from pysal.lib import weights
from pysal.explore import esda


def analyze_spatial_autocorrelation(gdf):
    # 공간 가중치 행렬 생성
    w = weights.distance.DistanceBand.from_dataframe(gdf, threshold=1000)

    # Moran's I 통계량 계산
    moran = esda.moran.Moran(gdf['중첩횟수'], w)

    print("\n=== 공간적 자기상관성 분석 ===")
    print(f"Moran's I: {moran.I:.3f}")
    print(f"p-value: {moran.p_sim:.3f}")

'''
**새로운 인사이트**:
1. **시간적 패턴**:
    - 특정 시간대의 중첩 위험도 증가
    - 시간대별 맞춤형 교통관리 전략 수립 가능

2. **지형적 특성**:
    - 경사도와 사고 발생의 상관관계
    - 지형에 따른 위험도 예측 가능

3. **공간적 군집성**:
    - 사고다발구역 간의 거리 패턴
    - 위험지역의 공간적 확산 예측

4. **도로 특성**:
    - 도로 유형별 위험도 차이
    - 도로 설계 개선점 도출

**실무적 제안**:
1. **단기 대책**:
    - 고위험 시간대 집중 관리
    - 중첩구역 우선 개선

2. **중기 대책**:
    - 도로 구조 개선
    - 교통시설물 보강

3. **장기 대책**:
    - 도시계획 반영
    - 교통체계 재설계

4. **예방 전략**:
    - 위험예측 모델 개발
    - 사전 예방 시스템 구축

'''