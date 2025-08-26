
# ==== From comp1.ipynb | code cell 1 ====
import pandas as pd


# ==== From comp1.ipynb | code cell 2 ====
def preprocess_merge(file1_path, file2_path, output_path):
    # 첫 번째 파일 읽기
    df1 = pd.read_csv(file1_path)

    # 두 번째 파일 읽기
    df2 = pd.read_csv(file2_path)

    # 결과를 저장할 데이터프레임 생성
    result_rows = []

    # 첫 번째 파일의 각 행을 순회하며 measure 범위와 feature를 추출
    for _, row in df1.iterrows():
        name = row['name']
        feature = row['important_feature']
        start = row['interval_start']
        end = row['interval_end']

        for measure in range(start, end + 1):
            # 두 번째 파일에서 일치하는 measure와 feature 필터링
            match_df = df2[(df2['measure'] == measure) & (df2['feature'] == feature)]

            if not match_df.empty:
                # 일치하는 데이터가 있는 경우
                for _, match_row in match_df.iterrows():
                    result_rows.append([
                        match_row['measure'],
                        match_row['feature'],
                        match_row['feature_value']
                    ])
            else:
                # 일치하는 데이터가 없는 경우
                result_rows.append([measure, feature, 'null'])

    # 결과 데이터프레임으로 변환
    result_df = pd.DataFrame(result_rows, columns=['measure', 'feature', 'feature_value'])
    
    # critical_score 계산
    critical_score_df = result_df.groupby(['measure', 'feature', 'feature_value']).size().reset_index(name='critical_score')

    # 결과를 CSV 파일로 저장
    critical_score_df.to_csv(output_path, index=False)
    print(f'Result saved to {output_path}')


# ==== From comp1.ipynb | code cell 3 ====
def merge(file1_path, file2_path, output_path):
    # 두 CSV 파일 읽기
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    # 두 번째 파일에서 critical_score가 2 이상인 행만 필터링
    #df2_filtered = df2[df2['critical_score'] >= 2]
    
    # concat 후 groupby로 처리
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # groupby 처리
    def resolve_group(group):
        result = group.iloc[0].copy()
        
        # feature_value: null이 아닌 값 선택
        feature_values = group['feature_value'].dropna()
        result['feature_value'] = feature_values.iloc[0] if not feature_values.empty else None

        # critical_score: 합산
        result['critical_score'] = group['critical_score'].fillna(0).sum()

        return result

    merged_df = combined_df.groupby(['measure', 'feature'], as_index=False).apply(resolve_group)

    # 인덱스 초기화 (groupby 후 생성되는 계층 인덱스 제거)
    merged_df.reset_index(drop=True, inplace=True)

    # 결과 저장
    merged_df.to_csv(output_path, index=False)
    print(f'Result saved to {output_path}')


# ==== From comp1.ipynb | code cell 4 ====
def pid_convert(pid_csv, performance_csv, output_csv):
    # 파일 읽기
    pid_df = pd.read_csv(pid_csv)
    performance_df = pd.read_csv(performance_csv)
    
    # midi_path와 pid를 딕셔너리로 매핑
    path_to_pid = dict(zip(pid_df['midi_path'], pid_df['pid']))
    
    # performance_id를 pid로 변환
    performance_df['pid'] = performance_df['performance_id'].map(path_to_pid)
    
    # 변환 과정: measure, feature, feature_value로 변경
    performance_df['measure'] = performance_df['bar'] + 1
    performance_df['feature'] = performance_df['feature_idx'] + 1
    performance_df['feature_value'] = performance_df['value']
    
    # 필요한 열 선택 및 순서 지정
    result_df = performance_df[['pid', 'measure', 'feature', 'feature_value']]

    
    ###chopin - pid가 45에서 60 사이인 행만 필터링
    filtered_df = result_df[(result_df['pid'] >= 45) & (result_df['pid'] <= 60)]
    ###beethoven - pid가 45미만, 60 초과인 행만 필터링
    #filtered_df = result_df[(result_df['pid'] < 45) | (result_df['pid'] > 60)]
    
    filtered_df = filtered_df.drop_duplicates(subset=['pid', 'measure', 'feature'])
    
    # 결과 CSV 파일 저장
    filtered_df.to_csv(output_csv, index=False)
    print(f'변환된 데이터가 {output_csv}로 저장되었습니다.')


# ==== From comp1.ipynb | code cell 5 ====
def compare(performance_csv, guideline_csv, output_csv):
    # CSV 파일 읽기
    performance_df = pd.read_csv(performance_csv)
    guideline_df = pd.read_csv(guideline_csv)
    
    
    
    performance_df['measure'] = performance_df['measure'].astype(float)
    performance_df['feature'] = performance_df['feature'].astype(float)
    guideline_df['measure'] = guideline_df['measure'].astype(float)
    guideline_df['feature'] = guideline_df['feature'].astype(float)
    

    # measure와 feature를 기준으로 병합 (right join)
    merged_df = pd.merge(performance_df, guideline_df, on=['measure', 'feature'], how='right')

    # difference 계산
    merged_df['difference'] = merged_df['feature_value_x'] - merged_df['feature_value_y']
    
    # 열 이름 변경
    merged_df = merged_df.rename(columns={'feature_value_x': 'feature_value','feature_value_y': 'guideline_value', 'critical_score': 'critical_score'})
    
    # 필요한 열만 선택
    result_df = merged_df[['pid', 'measure', 'feature', 'feature_value', 'guideline_value', 'difference', 'critical_score']]
    
    ### chopin
    result_df = result_df[result_df['measure'] < 80]
    ### beethoven
    #result_df = result_df[result_df['measure'] < 306]
    
    # 결과 저장
    result_df.to_csv(output_csv, index=False)
    print(f'비교 결과가 {output_csv}로 저장되었습니다.')


# ==== From comp1.ipynb | code cell 6 ====
def get_feature(item):
    
    feature_mapping = {
        'feature': [1, 9, 11, 2, 3, 7, 6, 8, 10, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'new_feature': [11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 10, 16, 17, 32],
        'back': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'lv1': [1,2,2,3,3,4,5,5,5,6,6,7,7,7,7,8,8,8,9,1,4,4,9],
    }

    df2 = pd.DataFrame(feature_mapping)
    subset = df2[df2['new_feature'] == item]

    feature_value = subset.iloc[0]['feature']
    back_value = subset.iloc[0]['back']
    return int(feature_value), int(back_value)


# ==== From comp1.ipynb | code cell 7 ====
def get_lv1(item):
    
    feature_mapping = {
        'feature': [1, 9, 11, 2, 3, 7, 6, 8, 10, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'new_feature': [11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 10, 16, 17, 32],
        'back': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'lv1': [1,2,2,3,3,4,5,5,5,6,6,7,7,7,7,8,8,8,9,1,4,4,9],
    }

    df2 = pd.DataFrame(feature_mapping)
    subset = df2[df2['new_feature'] == item]

    feature_value = subset.iloc[0]['lv1']
    return int(feature_value)


# ==== From comp1.ipynb | code cell 8 ====
def overlap(annotation, expid, output_csv):
    # CSV 파일 읽기
    df = pd.read_csv(annotation)  # 병합된 테이블
    df = df[df['pid'] != expid] # expid에 해당하는 pid 제외

    # 결과 저장을 위한 리스트
    results = []

    # 전체 measure 범위를 가져옴
    min_measure = df['start_measure'].min()
    max_measure = df['end_measure'].max()

    # measure 단위로 반복
    for measure in range(min_measure, max_measure + 1):
        df_measure = df[(df['start_measure'] <= measure) & (df['end_measure'] >= measure)]
        df_measure = df_measure.drop_duplicates()
        # Feature별 저장을 위한 딕셔너리
        feature_results = {}

        # 모든 pid (annotation) 반복
        for _, row in df_measure.iterrows():
            level = row['level']
            item = row['item']
            pid = row['pid']  # 현재 row의 pid 저장

            # Level 2만 처리
            if level != 2:
                continue

            suggestion_value = row['suggestion']
            observation_value = row['observation']

            # Feature Mapping 조회
            feature, back = get_feature(item)
            if feature is None:
                continue  # 매핑되지 않은 feature는 스킵

            if feature not in feature_results:
                feature_results[feature] = {"critical_score": 0, "feature_sum": 0}

            # Back 처리
            if back == 1:
                if suggestion_value > 0 : 
                    suggestion_value = 8 - suggestion_value
                if observation_value > 0 : 
                    observation_value = 8 - observation_value

            lv1 = get_lv1(item)

            # Suggestion 값이 있으면 바로 반영
            if suggestion_value > 0:
                
                # -1 ~ 1 정규화
                suggestion_value = (suggestion_value - 4) / 3.0

                feature_results[feature]["critical_score"] += 1
                feature_results[feature]["feature_sum"] += suggestion_value

            # Observation 값이 있으면 처리
            elif observation_value > 0:
                
                # -1 ~ 1 정규화
                observation_value = (observation_value - 4) / 3.0
                
                feature_results[feature]["critical_score"] += 1

                # Level 1 값 찾기 (같은 pid를 가진 것 중에서 찾기)
                level1_rows = df_measure[(df_measure['level'] == 1) & (df_measure['pid'] == pid)]
                matched_lv1 = level1_rows[level1_rows['item'] == lv1]

                if not matched_lv1.empty:
                    # Level 1의 observation 값을 사용
                    level1_value = matched_lv1.iloc[0]['observation']
                    feature_results[feature]["feature_sum"] += observation_value * ((level1_value / 5) - 1)
                else:
                    # Level 0 값 (같은 row의 score 값)
                    level0_value = row['score']
                    if level0_value is None:
                        level0_value = abs(observation_value - 4) * 10 / 3
                    feature_results[feature]["feature_sum"] += observation_value * ((level0_value / 5) - 1)

        # Feature별 Feature Value 계산 및 저장
        for feature, values in feature_results.items():
            feature_value = values["feature_sum"] / values["critical_score"] if values["critical_score"] > 0 else 0
            results.append({
                'measure': measure,
                'feature': feature,
                'critical_score': values["critical_score"],
                'feature_value': feature_value
            })

    # 결과를 데이터프레임으로 변환
    df_results = pd.DataFrame(results)
    df_results = df_results[df_results['feature'] < 20] # feature 19까지만 남기기

    # CSV 저장
    df_results.to_csv(output_csv, index=False)


# ==== From comp1.ipynb | code cell 9 ====
def standard(annotation, output_csv):
    # CSV 파일 읽기
    df = pd.read_csv(annotation)  # 병합된 테이블

    # 결과 저장을 위한 리스트
    results = []

    # 전체 measure 범위를 가져옴
    min_measure = df['start_measure'].min()
    max_measure = df['end_measure'].max()

    # measure 단위로 반복
    for measure in range(min_measure, max_measure + 1):
        df_measure = df[(df['start_measure'] <= measure) & (df['end_measure'] >= measure)]
        df_measure = df_measure.drop_duplicates()

        for _, row in df_measure.iterrows():
            level = row['level']
            item = row['item']
            pid = row['pid']
            suggestion_value = row['suggestion']
            observation_value = row['observation']

            # Level 2만 처리
            if level != 2:
                continue

            # Feature 매핑
            feature, back = get_feature(item)
            if feature is None:
                continue

            # Back 처리
            if back == 1:
                if suggestion_value > 0:
                    suggestion_value = 8 - suggestion_value
                if observation_value > 0:
                    observation_value = 8 - observation_value

            # 정규화
            if suggestion_value > 0:
                suggestion_norm = (suggestion_value - 4) / 3.0
            else:
                suggestion_norm = None

            if observation_value > 0:
                observation_norm = (observation_value - 4) / 3.0
            else:
                observation_norm = None

            results.append({
                'pid': pid,
                'measure': measure,
                'feature': feature,
                'suggestion': suggestion_norm,
                'observation': observation_norm
            })

    # 결과 저장
    df_results = pd.DataFrame(results)
    df_results = df_results[df_results['feature'] < 19]  # feature 18까지만 유지
    df_results.to_csv(output_csv, index=False)


# ==== From comp1.ipynb | code cell 10 ====
def standard2(annotation, output_csv):
    # CSV 파일 읽기
    df = pd.read_csv(annotation)

    # 결과를 저장할 리스트
    results = []

    # 각 row의 measure 범위를 풀어서 저장
    for _, row in df.iterrows():
        pid = row['pid']
        start = int(row['start_measure'])
        end = int(row['end_measure'])

        for measure in range(start, end + 1):
            results.append({'pid': pid, 'measure': measure})

    # DataFrame 생성 및 중복 제거
    df_result = pd.DataFrame(results).drop_duplicates().sort_values(['pid', 'measure'])

    # 저장
    df_result.to_csv(output_csv, index=False)


# ==== From comp1.ipynb | code cell 11 ====
def pid_drop(performance, expid, output_csv):
    # CSV 파일 읽기
    df = pd.read_csv(performance)  # 병합된 테이블
    df = df[df['pid'] == expid] # expid에 해당하는 pid만 남김
    min_val = df['feature_value'].min()
    max_val = df['feature_value'].max()

    # [-1, 1]로 정규화
    df['feature_value'] = 2 * (df['feature_value'] - min_val) / (max_val - min_val) - 1
    df.to_csv(output_csv, index=False)


# ==== From comp1.ipynb | code cell 12 ====
#결과 비교할 annotation 데이터 정리
standard('merged_c.csv', 'standard_c.csv')
standard2('merged_c.csv', 'standard2_c.csv')
standard('merged_b.csv', 'standard_b.csv')
standard2('merged_b.csv', 'standard2_b.csv')

###embedding model training
perf = 'concatenated_v5.csv'
perf2 = 'result/pid_v5.csv'

###chopin
anno = 'merged_c.csv'
crit = 'c_score.csv'

###beethoven
#anno = 'merged_b.csv'
#crit = 'b_score.csv'

### pid_convert에서 chopin/beethoven 곡에 해당하는 pid만 남기기
pid_convert('pidname.csv', perf, perf2)

perf_df = pd.read_csv(perf2)
pids = perf_df['pid'].unique() # pid 열에 있는 고유값 목록 가져오기

for expid in pids:
    
    # 각 expid에 대해 파이프라인 실행
    inter1 = f'result/{expid}_inter1.csv'
    inter2 = f'result/{expid}_inter2.csv'
    inter3 = f'result/{expid}_inter3.csv'
    inter4 = f'result/{expid}_inter4.csv'
    diff = f'result/{expid}_diff.csv'

    pid_drop(perf2, expid, inter1)
    overlap(anno, expid, inter2)
    preprocess_merge(crit, inter2, inter3)
    merge(inter3, inter2, inter4)
    ### chopin measure 79 beethoven measure 305
    compare(inter1, inter4, diff)

