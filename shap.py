
# ==== From shap.ipynb | code cell 1 ====
import pandas as pd 
import numpy as np 

from xgboost import XGBRegressor, plot_importance 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import shap


# ==== From shap.ipynb | code cell 2 ====
filename_input = 'shaptemp.csv'

# CSV 파일을 읽어옵니다.
csv_file = filename_input
df = pd.read_csv(csv_file)


# ==== From shap.ipynb | code cell 3 ====
df


# ==== From shap.ipynb | code cell 4 ====
df = df.dropna(subset=['Unsatisfactory - Convincing Interpretation'])


# ==== From shap.ipynb | code cell 5 ====
filtered_df = df.iloc[:,3:21]


# ==== From shap.ipynb | code cell 6 ====
filtered_df


# ==== From shap.ipynb | code cell 7 ====
y_new = df['Unsatisfactory - Convincing Interpretation'].values


# ==== From shap.ipynb | code cell 8 ====
y_new


# ==== From shap.ipynb | code cell 9 ====
y_new.size


# ==== From shap.ipynb | code cell 10 ====
print("NaN in y_train:", np.isnan(y_new).sum())
print("Inf in y_train:", np.isinf(y_new).sum())
print("Max y_train:", np.max(y_new))


# ==== From shap.ipynb | code cell 11 ====
X,y = filtered_df, y_new


# ==== From shap.ipynb | code cell 12 ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)


# ==== From shap.ipynb | code cell 13 ====
# 실험할 learning_rate 리스트
learning_rates = [0.095, 0.096, 0.097, 0.098, 0.099]

# 결과 저장용 리스트
results = []

# 각 learning_rate마다 학습하고 평가
for lr in learning_rates:
    print(f"Training with learning_rate={lr}...")
    
    model = XGBRegressor(
        learning_rate=lr,
        n_estimators=300,       # 트리 수 충분히 크게
        max_depth=6,
        random_state=1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'learning_rate': lr,
        'MSE': mse,
        'R_squared': r2
    })

# 결과 정리
results_df = pd.DataFrame(results)
print(results_df.sort_values('R_squared', ascending=False))


# ==== From shap.ipynb | code cell 14 ====
# modeling 
model = XGBRegressor(
    learning_rate=0.098,
    n_estimators=300,
    max_depth=6,
    random_state=1
)
model.fit(X_train, y_train)


# ==== From shap.ipynb | code cell 15 ====
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")


# ==== From shap.ipynb | code cell 16 ====
# load js 
shap.initjs()

'''
KernelExplainer() : KNN, SVM, RandomForest, GBM, H2O 
TreeExplainer() : tree-based machine learning model (faster) 
DeepExplainer() : deep learning model 
'''

explainer = shap.TreeExplainer(model)


# ==== From shap.ipynb | code cell 17 ====
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)


# ==== From shap.ipynb | code cell 18 ====
import glob
import os


# ==== From shap.ipynb | code cell 19 ====
input_folder = 'shap_v6'


# ==== From shap.ipynb | code cell 20 ====
output_folder = 'shap_value_v6'

os.makedirs(output_folder, exist_ok=True)

csv_files = glob.glob(os.path.join(input_folder, 'shap_*.csv'))

for file in csv_files:
    # 파일 이름만 추출 (예: shap_53.csv)
    filename = os.path.basename(file)
    
    # CSV 파일 읽기
    df = pd.read_csv(file)
    filtered_df = df.iloc[:, 1:19]  # 두 번째 열부터 18번째 열까지 사용
    
    # SHAP 값 계산
    shap_values = explainer.shap_values(filtered_df)
    
    # SHAP 값을 DataFrame으로 변환
    shap_df = pd.DataFrame(shap_values)
    
    # 결과 저장
    output_path = os.path.join(output_folder, filename)
    shap_df.to_csv(output_path, index=False)

    print(f"Processed: {filename} → {output_path}")


# ==== From shap.ipynb | code cell 21 ====
output_folder = 'overallscore_v6'

os.makedirs(output_folder, exist_ok=True)

csv_files = glob.glob(os.path.join(input_folder, 'shap_*.csv'))

for file in csv_files:
    # 파일 이름만 추출 (예: shap_53.csv)
    filename = os.path.basename(file)
    
    # CSV 파일 읽기
    df = pd.read_csv(file)
    filtered_df = df.iloc[:, 1:19]  # 두 번째 열부터 18번째 열까지 사용
    
    # output 값 추출
    score = model.predict(filtered_df)
    
    # SHAP 값을 DataFrame으로 변환
    score_df = pd.DataFrame(score)
    
    numeric_values = pd.to_numeric(score_df[0][1:], errors='coerce')
    print(numeric_values.describe())
    
    # 결과 저장
    output_path = os.path.join(output_folder, filename)
    score_df.to_csv(output_path, index=False)

    print(f"Processed: {filename} → {output_path}")


# ==== From shap.ipynb | code cell 22 ====
# ✅ pid 목록 정의
pid_list = get2_pid_list_from_result_diff()

for pid in pid_list:
    print(f"🔍 Processing PID: {pid}")

    try:
        # 1. improve.csv 로드
        improve_path = f'result_b/{pid}_improve.csv'
        df = pd.read_csv(improve_path)

        # 2. feature_1 ~ feature_18 컬럼만 선택
        filtered_df = df.iloc[:, 1:19]  # measure 열 제외하고 feature_1 ~ feature_18

        # 3. 모델 예측
        score = model.predict(filtered_df)

        # 4. 결과 저장
        score_df = pd.DataFrame({'score': score})
        score_path = f'result_b/{pid}_score.csv'
        score_df.to_csv(score_path, index=False)

        print(f"✅ Saved: {score_path}")

    except Exception as e:
        print(f"❌ Failed for PID {pid}: {e}")


# ==== From shap.ipynb | code cell 23 ====
import os
import re

def get2_pid_list_from_result_diff(result_dir='result_b'):
    pid_list = []
    pattern = re.compile(r'^(\d+)_diff\.csv$')  # 패턴: 숫자_pid_diff.csv

    for filename in os.listdir(result_dir):
        match = pattern.match(filename)
        if match:
            pid = int(match.group(1))
            pid_list.append(pid)

    return sorted(pid_list)


# ==== From shap.ipynb | code cell 24 ====
import os
import re

def get_pid_list_from_result_diff(result_dir='result'):
    pid_list = []
    pattern = re.compile(r'^(\d+)_diff\.csv$')  # 패턴: 숫자_pid_diff.csv

    for filename in os.listdir(result_dir):
        match = pattern.match(filename)
        if match:
            pid = int(match.group(1))
            pid_list.append(pid)

    return sorted(pid_list)


# ==== From shap.ipynb | code cell 25 ====
filtered_df

