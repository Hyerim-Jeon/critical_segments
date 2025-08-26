
# ==== From comp2.ipynb | code cell 1 ====
import json
import pandas as pd


# ==== From comp2.ipynb | code cell 2 ====
import pandas as pd
import numpy as np
import ruptures as rpt
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os
import seaborn as sns

def segment_single_pid(pid, diff_dir="result", shap_dir="shap_value_v1", smooth_sigma=1, diff_k=1.0, penalty = 0.1, percentile_threshold = 80, shap_threshold = 0.0):
    print(f"Processing PID {pid}...")

    # === 1. Load diff data
    diff_path = os.path.join(diff_dir, f"{pid}_diff.csv")
    diff_df = pd.read_csv(diff_path)
    mask = diff_df['difference'].isna()
    default_guideline = 0.0
    #score에서만 존재하고 annotation에 없어서 guideline value가 없는 경우 중립인 0으로 guideline 값 세팅
    diff_df.loc[mask, 'difference'] = diff_df.loc[mask, 'feature_value'] - default_guideline
    diff_df = diff_df[diff_df['feature'].between(1, 18)]  # ensure valid feature range

    # Initialize full 0-filled DataFrame with shape (79, 18)
    
    ###chopin 79
    #full_index = pd.Index(range(1, 80), name='measure')       # 1~79 measures
        
    ###beethoven 305
    full_index = pd.Index(range(1, 306), name='measure')       # 1~305 measures
    
    full_columns = pd.Index(range(1, 19), name='feature')     # 1~18 features

    # === Create diff_pivot: shape (79, 18), initialized with 0.0
    diff_pivot = pd.DataFrame(0.0, index=full_index, columns=full_columns)
    actual_diff = diff_df.pivot_table(index="measure", columns="feature", values="difference", aggfunc='first')
    diff_pivot.update(actual_diff)
    
    # diff_k 클수록 완만해짐
    scaled_diff = diff_pivot.applymap(lambda x: np.tanh(x / diff_k))
    diff_pivot.update(scaled_diff)
    
    # diff 값 절댓값으로 변환
    diff_pivot = diff_pivot.applymap(np.abs)

    # === Create critical_pivot: shape (79, 18), initialized with 0
    critical_pivot = pd.DataFrame(0, index=full_index, columns=full_columns)
    actual_critical = diff_df.pivot_table(index="measure", columns="feature", values="critical_score", aggfunc='first')
    critical_pivot.update(actual_critical)
    
    # === Scale critical_pivot values to 0 ~ 1
    max_val = critical_pivot.to_numpy().max()
    if max_val > 0:
        critical_pivot = critical_pivot / max_val
    else:
        critical_pivot = pd.DataFrame(0.0, index=full_index, columns=full_columns)  # 모든 값이 0인 경우 처리

    # Ensure all 18 features present
    for pivot in [diff_pivot, critical_pivot]:
        pivot = pivot.reindex(columns=range(1, 19), fill_value=0)
        pivot.sort_index(inplace=True)

    # === 2. Load shap data
    shap_path = os.path.join(shap_dir, f"shap_{pid}.csv")
    shap_df = pd.read_csv(shap_path)
    shap_df["measure"] = range(1, len(shap_df) + 1)

    # Melt shap, then pivot to same shape
    shap_long = shap_df.melt(id_vars="measure", var_name="feature_index", value_name="shap_value")
    shap_long["feature"] = shap_long["feature_index"].astype(int) + 1
    shap_pivot = shap_long.pivot_table(index="measure", columns="feature", values="shap_value")
    shap_pivot = shap_pivot.reindex(columns=range(1, 19), fill_value=0).sort_index()

    common_measures = diff_pivot.index.intersection(shap_pivot.index)
    diff_array = diff_pivot.loc[common_measures].values
    shap_array = shap_pivot.loc[common_measures].values
    critical_array = critical_pivot.loc[common_measures].fillna(0).values
    
    # 안전하게 NaN을 모두 0으로
    diff_array = np.nan_to_num(diff_array, nan=0.0)
    shap_array = np.nan_to_num(shap_array, nan=0.0)
    critical_array = np.nan_to_num(critical_array, nan=0.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    sns.heatmap(diff_pivot.values, cmap="coolwarm", ax=axes[0], cbar=True)
    axes[0].set_title("Difference")
    axes[0].set_xlabel("Feature")
    axes[0].set_ylabel("Measure")

    sns.heatmap(critical_pivot.values, cmap="YlOrRd", ax=axes[1], cbar=True)
    axes[1].set_title("Critical Score")
    axes[1].set_xlabel("Feature")

    sns.heatmap(shap_pivot.values, cmap="Blues", ax=axes[2], cbar=True)
    axes[2].set_title("SHAP Value")
    axes[2].set_xlabel("Feature")

    plt.tight_layout()
    plt.show()
    
    
    # === 1. Calculate weighted_diff for all features
    #weighted_diff = np.abs(diff_pivot.values * shap_pivot.values * critical_pivot.values)  # shape: (79, 18)
    weighted_diff = np.abs(diff_pivot.values * critical_pivot.values)
    
    # 1. 먼저 log scaling -> 차이 극대화 위해
    weighted_diff = np.log1p(weighted_diff)
    
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    sns.heatmap(diff_pivot.values, cmap="coolwarm", ax=axes[0], cbar=True)
    axes[0].set_title("Difference")
    axes[0].set_xlabel("Feature")
    axes[0].set_ylabel("Measure")

    sns.heatmap(critical_pivot.values, cmap="YlOrRd", ax=axes[1], cbar=True)
    axes[1].set_title("Critical Score")
    axes[1].set_xlabel("Feature")

    sns.heatmap(weighted_diff, cmap="Blues", ax=axes[2], cbar=True)
    axes[2].set_title("Weighted Difference Value")
    axes[2].set_xlabel("Feature")

    plt.tight_layout()
    plt.show()
    
    

    # === 2. Segment and plot each feature
    n_features = weighted_diff.shape[1]
    
    change_points_list = []
    score_list = []
    smoothed_score_list = []
    
    
    
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features), sharex=True)

    for feature_idx in range(n_features):
        ax = axes[feature_idx]

        # Score per feature (1D time series)
        score = weighted_diff[:, feature_idx]
        smoothed_score = gaussian_filter1d(score, sigma=smooth_sigma)

        # Save for return
        score_list.append(score)
        smoothed_score_list.append(smoothed_score)

        # Segmentation
        model = rpt.Pelt(model="rbf").fit(smoothed_score.reshape(-1, 1))
        change_points = model.predict(pen=penalty)
        change_points_list.append(change_points)

        # Plot
        ax.plot(smoothed_score, label="Smoothed Score")
        for cp in change_points:
            ax.axvline(cp, color="red", linestyle="--", alpha=0.7)
        ax.set_title(f"Feature {feature_idx + 1}")
        ax.set_ylabel("Score")
        ax.grid(True)
        ax.legend()

    plt.xlabel("Measure")
    plt.tight_layout()
    plt.show()
    
    #각 feature에 대해 segment 별로 smoothed score값 평균을 이용해 상위 몇 percentile에 속하는 segment를 먼저 찾고, 그 segment들에 대해 shap value 평균을 구해서 threshold 이하일 경우 그 segment 정보를 출력하는 코드 이어서 작성
    # 예: 상위 90% percentile 을 기준으로 중요 segment 선정
    

    # shap_array: (measure, feature) 크기 numpy 배열
    # smoothed_score_list: feature별 (measure,) 크기 리스트
    
    selected_segments = []  # 결과를 담을 리스트

    for feature_idx in range(n_features):
        smoothed_scores = smoothed_score_list[feature_idx]
        change_points = change_points_list[feature_idx]

        # segment 경계 정의 (start, end)
        segment_bounds = [(0, change_points[0])]
        for i in range(1, len(change_points)):
            segment_bounds.append((change_points[i-1], change_points[i]))

        # 각 segment별 smoothed_score 평균 계산
        segment_means = []
        for start, end in segment_bounds:
            segment_mean = smoothed_scores[start:end].mean()
            segment_means.append(segment_mean)

        segment_means = np.array(segment_means)

        # 상위 percentile 임계값 계산
        thresh_value = np.percentile(segment_means, percentile_threshold)

        print(f"\nFeature {feature_idx+1} - 상위 {percentile_threshold} percentile threshold: {thresh_value:.4f}")
        

        for idx, (start, end) in enumerate(segment_bounds):
            segment_size = end - start
            if segment_size > 20 or segment_size < 2:
                continue  # 크기 20 초과 or 2 미만인 세그먼트는 무시

            if segment_means[idx] >= thresh_value:
                # 해당 segment 내 shap value 평균 계산
                segment_shap_vals = shap_array[start:end, feature_idx]
                shap_mean = segment_shap_vals.mean()
                print(f"  Segment {idx+1}: Measure {start+1} ~ {end}, "
                          f"Smoothed Score Mean: {segment_means[idx]:.4f}, ")

                
                if shap_mean <= shap_threshold:
                    print(f"  Segment {idx+1}: Measure {start+1} ~ {end}, "
                          f"Smoothed Score Mean: {segment_means[idx]:.4f}, "
                          f"SHAP Mean: {shap_mean:.4f} (<= {shap_threshold})")
                    # 결과 저장
                    segment_info = {
                        "feature": feature_idx + 1,
                        "start_measure": start + 1,
                        "end_measure": end,
                        "smoothed_score_mean": segment_means[idx],
                        "shap_mean": shap_mean
                    }
                    selected_segments.append(segment_info)
    
    return selected_segments, smoothed_score_list, shap_array


# ==== From comp2.ipynb | code cell 3 ====
def update_segment_pid(pid, selected_segments, diff_dir="result",):

    # === 1. Load diff data
    diff_path = os.path.join(diff_dir, f"{pid}_diff.csv")
    diff_df = pd.read_csv(diff_path)
    mask = diff_df['difference'].isna()
    default_guideline = 0.0
    #score에서만 존재하고 annotation에 없어서 guideline value가 없는 경우 중립인 0으로 guideline 값 세팅
    diff_df.loc[mask, 'difference'] = diff_df.loc[mask, 'feature_value'] - default_guideline
    
    
    # === 2. Create empty list to collect filtered rows
    result_rows = []

    # === 3. Iterate over selected segments
    for segment_id, seg in enumerate(selected_segments):
        f = seg['feature']
        start_m = seg['start_measure']
        end_m = seg['end_measure']
        score_mean = seg['smoothed_score_mean']
        shap_mean = seg['shap_mean']

        # Filter diff_df for this segment's feature and measure range
        seg_df = diff_df[
            (diff_df['feature'] == f) &
            (diff_df['measure'] >= start_m) &
            (diff_df['measure'] <= end_m)
        ].copy()

        # Add segment info
        seg_df['segment'] = segment_id
        seg_df['smoothed_score_mean'] = score_mean
        seg_df['shap_mean'] = shap_mean

        result_rows.append(seg_df)

    # === 4. Concatenate all matched rows
    if result_rows:
        result_df = pd.concat(result_rows, ignore_index=True)
    else:
        result_df = pd.DataFrame()  # Empty DataFrame fallback

    # === 5. Save result
    result_path = os.path.join(diff_dir, f"{pid}_result0.csv")
    result_df.to_csv(result_path, index=False)


# ==== From comp2.ipynb | code cell 4 ====
def analyze_segments(change_points_list, score_list):
    all_sorted_segments = []

    for feature_idx in range(len(score_list)):
        cps = change_points_list[feature_idx]
        score = score_list[feature_idx]

        # Plot
        plt.figure(figsize=(15, 3))
        plt.plot(score, label="Weighted Diff Score")
        for cp in cps[:-1]:  # 마지막은 len(score)라 생략
            plt.axvline(cp, color='red', linestyle='--', alpha=0.6)
        plt.xlabel("Measure Index")
        plt.ylabel("Importance Score")
        plt.title(f"Feature {feature_idx + 1}: SHAP-weighted Feature Difference")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Segment scoring
        segment_scores = []
        if len(cps) > 1:  # change_points가 하나 이상 있어야 세그먼트가 존재
            for i in range(len(cps) - 1):
                start, end = cps[i], cps[i+1]
                segment_mean = score[start:end].mean()
                segment_max = score[start:end].max()
                segment_scores.append((i, start, end, segment_mean, segment_max))

            # Sort by mean score
            sorted_segments = sorted(segment_scores, key=lambda x: x[3], reverse=True)
        else:
            # No segments found, return an empty list
            sorted_segments = []

        all_sorted_segments.append(sorted_segments)

        # Optional: print top-1 important segment if available
        if sorted_segments:
            print(f"[Feature {feature_idx + 1}] Top segment (mean): {sorted_segments[0]}")
        else:
            print(f"[Feature {feature_idx + 1}] No segments found")

    return all_sorted_segments


# ==== From comp2.ipynb | code cell 5 ====
def extract_important_indices(score_list, percentile=70):
    important_indices_list = []

    for feature_idx, score in enumerate(score_list):
        if len(score) == 0:
            print(f"[Feature {feature_idx + 1}] No scores available, skipping.")
            important_indices_list.append([])  # 빈 리스트를 추가
            continue

        threshold = np.percentile(score, percentile)
        important_indices = np.where(score > threshold)[0]
        important_indices_list.append(important_indices)

        # Optional print
        print(f"[Feature {feature_idx + 1}] Threshold: {threshold:.4f}, Important Indices: {important_indices}")

    return important_indices_list


# ==== From comp2.ipynb | code cell 6 ====
def extract_important_segments(all_sorted_segments, percentile=70):
    important_segments_list = []

    for feature_idx, sorted_segments in enumerate(all_sorted_segments):
        if not sorted_segments:
            print(f"[Feature {feature_idx + 1}] No segments to evaluate.")
            important_segments_list.append([])
            continue

        # Extract mean scores for thresholding
        mean_scores = [seg[3] for seg in sorted_segments]
        threshold = np.percentile(mean_scores, percentile)

        # Filter segments above the threshold
        important_segments = [seg for seg in sorted_segments if seg[3] > threshold]
        important_segments_list.append(important_segments)

        # Optional debug print
        print(f"[Feature {feature_idx + 1}] Threshold: {threshold:.4f}, "
              f"Selected Segments: {[(s[1], s[2]) for s in important_segments]}")

    return important_segments_list


# ==== From comp2.ipynb | code cell 7 ====
def compare_result_pid(pid, base_csv='standard_c.csv', result_dir='result'):
    # === Load CSVs
    standard_df = pd.read_csv(base_csv)
    result_df = pd.read_csv(os.path.join(result_dir, f"{pid}_result0.csv"))

    # === Filter by pid
    std_pid_df = standard_df[standard_df['pid'] == pid]
    res_pid_df = result_df[result_df['pid'] == pid]

    # === Get unique features in either standard or result
    features = set(std_pid_df['feature'].unique()).union(res_pid_df['feature'].unique())

    results = []

    for feature in sorted(features):
        std_measures = set(std_pid_df[std_pid_df['feature'] == feature]['measure'])
        res_measures = set(res_pid_df[res_pid_df['feature'] == feature]['measure'])

        intersection = std_measures & res_measures
        union = std_measures | res_measures

        iou = len(intersection) / len(union) if union else 0
        precision = len(intersection) / len(res_measures) if res_measures else 0
        recall = len(intersection) / len(std_measures) if std_measures else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
        coverage = len(intersection) / len(std_measures) if std_measures else 0

        results.append({
            'pid': pid,
            'feature': feature,
            'standard_measures': sorted(std_measures),
            'result_measures': sorted(res_measures),
            'intersection': sorted(intersection),
            'IoU': round(iou, 3),
            'Precision': round(precision, 3),
            'Recall': round(recall, 3),
            'F1': round(f1, 3),
            'Coverage': round(coverage, 3)
        })

    return pd.DataFrame(results)


# ==== From comp2.ipynb | code cell 8 ====
def compare2_result_pid(pid, base_csv='standard2_c.csv', result_dir='result'):
    # === Load CSVs
    standard_df = pd.read_csv(base_csv)
    result_df = pd.read_csv(os.path.join(result_dir, f"{pid}_result0.csv"))

    # === Filter by pid
    std_pid_df = standard_df[standard_df['pid'] == pid]
    res_pid_df = result_df[result_df['pid'] == pid]

    # === Set of measures (feature 무시)
    std_measures = set(std_pid_df['measure'])
    res_measures = set(res_pid_df['measure'])

    intersection = std_measures & res_measures
    union = std_measures | res_measures

    iou = len(intersection) / len(union) if union else 0
    precision = len(intersection) / len(res_measures) if res_measures else 0
    recall = len(intersection) / len(std_measures) if std_measures else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    coverage = len(intersection) / len(std_measures) if std_measures else 0

    result = {
        'pid': pid,
        'standard_measures': sorted(std_measures),
        'result_measures': sorted(res_measures),
        'intersection': sorted(intersection),
        'IoU': round(iou, 3),
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'F1': round(f1, 3),
        'Coverage': round(coverage, 3)
    }

    return pd.DataFrame([result])


# ==== From comp2.ipynb | code cell 9 ====
def compare_guide_pid(pid, standard_csv='standard_c.csv', result_dir='result'):
    # Load standard and inter4 files
    standard_df = pd.read_csv(standard_csv)
    result_df = pd.read_csv(os.path.join(result_dir, f"{pid}_inter4.csv"))

    # Filter by pid
    std_df = standard_df[standard_df['pid'] == pid]
    res_df = result_df.copy()

    features = set(std_df['feature'].unique()).union(res_df['feature'].unique())

    results = []

    for feature in sorted(features):
        std_measures = set(std_df[std_df['feature'] == feature]['measure'])
        res_measures = set(res_df[res_df['feature'] == feature]['measure'])

        intersection = std_measures & res_measures
        union = std_measures | res_measures

        iou = len(intersection) / len(union) if union else 0
        precision = len(intersection) / len(res_measures) if res_measures else 0
        recall = len(intersection) / len(std_measures) if std_measures else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
        coverage = len(intersection) / len(std_measures) if std_measures else 0

        results.append({
            'pid': pid,
            'feature': int(feature),
            'standard_measures': sorted(std_measures),
            'result_measures': sorted(res_measures),
            'intersection': sorted(intersection),
            'IoU': round(iou, 3),
            'Precision': round(precision, 3),
            'Recall': round(recall, 3),
            'F1': round(f1, 3),
            'Coverage': round(coverage, 3)
        })

    return pd.DataFrame(results)


# ==== From comp2.ipynb | code cell 10 ====
def compare2_guide_pid(pid, standard_csv='standard2_c.csv', result_dir='result'):
    # Load standard and inter4 files
    standard_df = pd.read_csv(standard_csv)
    result_df = pd.read_csv(os.path.join(result_dir, f"{pid}_inter4.csv"))

    # Filter by pid
    std_df = standard_df[standard_df['pid'] == pid]
    res_df = result_df.copy()
    ###evaluation critical score chopin 3, beethoven 8부터
    
    if(standard_csv == 'standard2_c.csv'):
        res_df = result_df[result_df['critical_score'] > 2]
    else:
        res_df = result_df[result_df['critical_score'] > 7]

    std_measures = set(std_df['measure'])
    res_measures = set(res_df['measure'])

    intersection = std_measures & res_measures
    union = std_measures | res_measures

    iou = len(intersection) / len(union) if union else 0
    precision = len(intersection) / len(res_measures) if res_measures else 0
    recall = len(intersection) / len(std_measures) if std_measures else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    coverage = len(intersection) / len(std_measures) if std_measures else 0

    result = {
        'pid': pid,
        'standard_measures': sorted(std_measures),
        'result_measures': sorted(res_measures),
        'intersection': sorted(intersection),
        'IoU': round(iou, 3),
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'F1': round(f1, 3),
        'Coverage': round(coverage, 3)
    }

    return pd.DataFrame([result])


# ==== From comp2.ipynb | code cell 11 ====
def improve_pid(pid, result_dir='result', shap_dir='shap_v1'):
    # Load result0.csv
    result_path = os.path.join(result_dir, f"{pid}_result0.csv")
    result_df = pd.read_csv(result_path)

    # Load shap_{pid}.csv
    shap_path = os.path.join(shap_dir, f"shap_{pid}.csv")
    shap_df = pd.read_csv(shap_path)

    # Make a copy to modify
    updated_shap_df = shap_df.copy()

    # Process each row in result0.csv
    for _, row in result_df.iterrows():
        measure = int(row['measure'])
        feature = int(row['feature'])  # feature index: 1~19
        value = (row['feature_value'] - row['difference'] + 1) * 7

        # Column name in shap.csv is 'feature_{n}'
        col_name = f'feature_{feature}'
        
        # Update shap value at that (measure, feature) location
        if measure in updated_shap_df['measure'].values and col_name in updated_shap_df.columns:
            updated_shap_df.loc[updated_shap_df['measure'] == measure, col_name] = value

    # Save updated CSV
    output_path = os.path.join(result_dir, f"{pid}_improve.csv")
    updated_shap_df.to_csv(output_path, index=False)


# ==== From comp2.ipynb | code cell 12 ====
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


# ==== From comp2.ipynb | code cell 13 ====
### chopin / beethoven (result 폴더에서 확인)
# 먼저 처리할 pid 목록 정의
pid_list = get_pid_list_from_result_diff(result_dir='result_v2_c')

###chopin
base='standard_c.csv'
base2='standard2_c.csv'

###beethoven
#base='standard_b.csv'
#base2='standard2_b.csv'

# 결과 저장용 리스트 (필요 시)
comparison_results = []
comparison2_results = []
guide_results = []
guide2_results = []

# pid별로 순차 처리
for pid in pid_list:
    print(f"\n🔄 Processing PID: {pid}")

    # 1. segment 분석 및 저장
    ### shap_value_v?
    #selected_segments, score_list, critical_array = segment_single_pid(pid=pid, diff_dir="result", shap_dir="shap_value_v1", smooth_sigma=1)
    #update_segment_pid(pid, selected_segments=selected_segments)

    # 2. 비교 함수 실행 (결과 저장 가능)
    #comp_df = compare_result_pid(pid, base_csv=base, result_dir='result_v6_b')
    #comparison_results.append(comp_df)

    #comp2_df = compare2_result_pid(pid, base_csv=base2, result_dir='result_v6_b')
    #comparison2_results.append(comp2_df)

    #guide_df = compare_guide_pid(pid, standard_csv=base, result_dir='result_v6_b')
    #guide_results.append(guide_df)

    guide2_df = compare2_guide_pid(pid, standard_csv=base2, result_dir='result_v2_c')
    guide2_results.append(guide2_df)

    # 3. shap 개선
    ### shap_v?
    #improve_pid(pid, result_dir='result', shap_dir='shap_v1')

#결과를 하나의 DataFrame으로 병합하고 활용
#all_comp_df = pd.concat(comparison_results, ignore_index=True)
#all_comp2_df = pd.concat(comparison2_results, ignore_index=True)
#all_guide_df = pd.concat(guide_results, ignore_index=True)
all_guide2_df = pd.concat(guide2_results, ignore_index=True)

### 저장
#all_comp_df.to_csv("v6_b_compare.csv", index=False)
#all_comp2_df.to_csv("v6_b_compare2.csv", index=False)
#all_guide_df.to_csv("v6_b_guide.csv", index=False)
all_guide2_df.to_csv("v2_c_guide2.csv", index=False)

