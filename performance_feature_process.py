
# ==== From performance_feature_process.ipynb | code cell 1 ====
import pandas as pd
import os


# ==== From performance_feature_process.ipynb | code cell 2 ====
# 1. 데이터 불러오기
df = pd.read_csv('shap_v5.csv')


# ==== From performance_feature_process.ipynb | code cell 3 ====
# 2. feature_idx를 feature_1 ~ feature_19로 변환, value 7 degree
df['feature_name'] = df['feature'].apply(lambda x: f'feature_{x}')
df['feature_value'] = df['feature_value'] * 7


# ==== From performance_feature_process.ipynb | code cell 4 ====
# 3. (performance_id, bar) 단위로 pivot
pivot_df = df.pivot_table(index=['pid', 'measure'], 
                          columns='feature_name', 
                          values='feature_value').reset_index()


# ==== From performance_feature_process.ipynb | code cell 5 ====
# 4. performance_id별로 나눠서 저장
output_dir = 'shap_v5'
os.makedirs(output_dir, exist_ok=True)


# ==== From performance_feature_process.ipynb | code cell 6 ====
# 5. 열 순서를 feature_1, feature_2, ..., feature_19 순서로 재정렬
feature_cols = [f'feature_{i}' for i in range(1, 20)]
pivot_df = pivot_df[['pid', 'measure'] + feature_cols]


# ==== From performance_feature_process.ipynb | code cell 7 ====
for pid, group in pivot_df.groupby('pid'):
    # performance_id 값에서 파일명에 안 맞는 특수문자 제거하고 싶으면 추가할 수 있음
    filename = os.path.join(output_dir, f'shap_{pid}.csv')
    group.drop(columns='pid').to_csv(filename, index=False)

print("✅ 모든 performance_id 별로 csv 파일 저장 완료!")

