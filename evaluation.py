
# ==== From evaluation.ipynb | code cell 1 ====
import os
import re
import ast
import pandas as pd
import matplotlib.pyplot as plt


# ==== From evaluation.ipynb | code cell 2 ====
def get_pid_list_from_result_diff(version, code):
    result_dir = f'result_{version}_{code}'
    
    pid_list = []
    pattern = re.compile(r'^(\d+)_diff\.csv$')  # 패턴: 숫자_pid_diff.csv

    for filename in os.listdir(result_dir):
        match = pattern.match(filename)
        if match:
            pid = int(match.group(1))
            pid_list.append(pid)

    return sorted(pid_list)


# ==== From evaluation.ipynb | code cell 3 ====
def evaluate_version_score(version, code, pid_list):
    shap_dir = f'overallscore_{version}'
    result_dir = f'result_{version}_{code}'

    records = []

    for pid in pid_list:
        shap_path = os.path.join(shap_dir, f'shap_{pid}.csv')
        score_path = os.path.join(result_dir, f'{pid}_score.csv')

        try:
            # A: shap_{pid}.csv 불러오기 (한 열짜리)
            shap_df = pd.read_csv(shap_path)
            shap_mean = pd.to_numeric(shap_df.iloc[:, 0], errors='coerce').mean()

            # B: {pid}_score.csv 불러오기
            score_df = pd.read_csv(score_path)
            result_mean = pd.to_numeric(score_df['score'], errors='coerce').mean()

            records.append({
                'pid': pid,
                'version': version,
                'code': code,
                'original_mean': round(shap_mean, 6),
                'improve_mean': round(result_mean, 6),
                'diff': round(result_mean - shap_mean, 6)
            })

        except Exception as e:
            print(f"⚠️ PID {pid} 처리 중 오류 발생: {e}")
            continue

    return pd.DataFrame(records)


# ==== From evaluation.ipynb | code cell 4 ====
def evaluate_segment_metrics(version, code):
    base = f"{version}_{code}"

    # === 파일 경로
    compare_df = pd.read_csv(f"{base}_compare.csv")
    guide_df = pd.read_csv(f"{base}_guide.csv")
    compare2_df = pd.read_csv(f"{base}_compare2.csv")
    guide2_df = pd.read_csv(f"{base}_guide2.csv")
    compare_df = compare_df[compare_df['pid'] != 70]
    guide_df = guide_df[guide_df['pid'] != 70]
    compare2_df = compare2_df[compare2_df['pid'] != 70]
    guide2_df = guide2_df[guide2_df['pid'] != 70]

    
    # === 문자열 리스트 파싱
    for df in [compare2_df, guide2_df]:
        df['standard_measures'] = df['standard_measures'].apply(lambda x: ast.literal_eval(str(x)) if pd.notna(x) else [])
        df['result_measures'] = df['result_measures'].apply(lambda x: ast.literal_eval(str(x)) if pd.notna(x) else [])

    
    
    # === FEATURE 기준 비교 (A vs B)
    merged_feat = pd.merge(
        compare_df,
        guide_df,
        on=['pid', 'feature'],
        suffixes=('_compare', '_guide')
    )

    # === MEASURE 기준 비교 (C vs D)
    merged_all = pd.merge(
        compare2_df,
        guide2_df,
        on='pid',
        suffixes=('_compare', '_guide')
    )

    # 비교 지표
    metrics = ['IoU', 'Precision', 'Recall', 'F1', 'Coverage']

    # 차이 계산
    for metric in metrics:
        merged_feat[f'{metric}_diff'] = merged_feat[f'{metric}_compare'] - merged_feat[f'{metric}_guide']
        merged_all[f'{metric}_diff'] = merged_all[f'{metric}_compare'] - merged_all[f'{metric}_guide']

    # === Summary of diffs
    def summarize_diffs(df, level):
        return {
            'level': level,
            **{f'{metric}_diff_mean': df[f'{metric}_diff'].mean() for metric in metrics}
        }

    summary_df = pd.DataFrame([
        summarize_diffs(merged_feat, 'feature'),
        summarize_diffs(merged_all, 'overall')
    ])

    # === Summary of compare, guide, compare2, guide2 (raw values)
    def summarize_metrics(df, source, level):
        return {
            'source': source,
            'level': level,
            **{f'{metric}_mean': df[metric].mean() for metric in metrics if metric in df.columns}
        }

    compare_stats = summarize_metrics(compare_df, 'compare', 'feature')
    guide_stats = summarize_metrics(guide_df, 'guide', 'feature')
    compare2_stats = summarize_metrics(compare2_df, 'compare', 'overall')
    guide2_stats = summarize_metrics(guide2_df, 'guide', 'overall')

    raw_summary_df = pd.DataFrame([compare_stats, guide_stats, compare2_stats, guide2_stats])

    # === 추가: measure 개수 요약
    def measure_lengths(df, source):
        return pd.DataFrame({
            'pid': df['pid'],
            'standard_len': df['standard_measures'].apply(len),
            'result_len': df['result_measures'].apply(len),
            'source': source
        })

    compare2_lengths = measure_lengths(compare2_df, 'compare')
    guide2_lengths = measure_lengths(guide2_df, 'guide')

    measure_summary_df = pd.concat([compare2_lengths, guide2_lengths], ignore_index=True)

    # === Aggregated 통계 (mean, min, max, std, var)
    measure_stat_summary_df = measure_summary_df.groupby('source')[['standard_len', 'result_len']].agg(['mean', 'min', 'max', 'std', 'var'])
    measure_stat_summary_df.columns = ['_'.join(col) for col in measure_stat_summary_df.columns]
    measure_stat_summary_df = measure_stat_summary_df.reset_index()

    # === Return all
    return (
        summary_df,               # 1. 지표 차이 요약
        raw_summary_df,           # 2. 원본 지표 요약
        merged_feat,              # 3. feature 단위 비교 상세
        merged_all,               # 4. 전체 measure 단위 비교 상세
        measure_summary_df,       # 5. 각 pid별 measure 수
        measure_stat_summary_df   # 6. measure 수의 집계 통계
    )


# ==== From evaluation.ipynb | code cell 5 ====
#test
def evaluate2_version_score(version, code, pid_list):
    shap_dir = f'overallscore_{version}'
    result_dir = f'result_{code}'

    records = []

    for pid in pid_list:
        shap_path = os.path.join(shap_dir, f'shap_{pid}.csv')
        score_path = os.path.join(result_dir, f'{pid}_score.csv')

        try:
            # A: shap_{pid}.csv 불러오기 (한 열짜리)
            shap_df = pd.read_csv(shap_path)
            shap_mean = pd.to_numeric(shap_df.iloc[:, 0], errors='coerce').mean()

            # B: {pid}_score.csv 불러오기
            score_df = pd.read_csv(score_path)
            result_mean = pd.to_numeric(score_df['score'], errors='coerce').mean()

            records.append({
                'pid': pid,
                'version': version,
                'code': code,
                'original_mean': round(shap_mean, 6),
                'improve_mean': round(result_mean, 6),
                'diff': round(result_mean - shap_mean, 6)
            })

        except Exception as e:
            print(f"⚠️ PID {pid} 처리 중 오류 발생: {e}")
            continue

    return pd.DataFrame(records)


# ==== From evaluation.ipynb | code cell 6 ====
pid_list = get2_pid_list_from_result_diff("c")
df1 = evaluate2_version_score("v2", "c", pid_list)
pid_list = get2_pid_list_from_result_diff("b")
df2 = evaluate2_version_score("v2", "b", pid_list)


# ==== From evaluation.ipynb | code cell 7 ====
df_total = pd.concat([df1, df2], ignore_index=True)

mean_values = df1.mean(numeric_only=True)
print("chopin 평균:\n", mean_values)

mean_values = df2.mean(numeric_only=True)
print("beethoven 평균:\n", mean_values)

mean_values = df_total.mean(numeric_only=True)
print("전체 평균:\n", mean_values)


# ==== From evaluation.ipynb | code cell 8 ====
def get2_pid_list_from_result_diff(version):
    result_dir = f'result_{version}'
    
    pid_list = []
    pattern = re.compile(r'^(\d+)_diff\.csv$')  # 패턴: 숫자_pid_diff.csv

    for filename in os.listdir(result_dir):
        match = pattern.match(filename)
        if match:
            pid = int(match.group(1))
            pid_list.append(pid)

    return sorted(pid_list)


# ==== From evaluation.ipynb | code cell 9 ====
import numpy as np

def improve_pid(pid, result_dir='result_c', shap_dir='shap_v1'):
    # Load result0.csv
    result_path = os.path.join(result_dir, f"{pid}_result0.csv")
    result_df = pd.read_csv(result_path)

    # Load shap_{pid}.csv
    shap_path = os.path.join(shap_dir, f"shap_{pid}.csv")
    shap_df = pd.read_csv(shap_path)

    # Make a copy to modify
    updated_shap_df = shap_df.copy()
    
    k = 73.7
    
    # Randomly sample k% of result_df
    num_samples = max(1, int(len(result_df) * (k / 100.0)))
    sampled_df = result_df.sample(n=num_samples, random_state=42)  # fixed seed for reproducibility

    for _, row in sampled_df.iterrows():
        measure = int(row['measure'])
        feature = int(row['feature'])  # feature index: 1~19
        value = np.random.uniform(1.0, 7.0)  # random float between 1 and 7

        col_name = f'feature_{feature}'
        if measure in updated_shap_df['measure'].values and col_name in updated_shap_df.columns:
            updated_shap_df.loc[updated_shap_df['measure'] == measure, col_name] = value

    # Save updated CSV
    output_path = os.path.join(result_dir, f"{pid}_improve.csv")
    updated_shap_df.to_csv(output_path, index=False)


# ==== From evaluation.ipynb | code cell 10 ====
# 먼저 처리할 pid 목록 정의
pid_list = get2_pid_list_from_result_diff('b')

###chopin
#base='standard_c.csv'
#base2='standard2_c.csv'

###beethoven
#base='standard_b.csv'
#base2='standard2_b.csv'

# pid별로 순차 처리
for pid in pid_list:
    print(f"\n🔄 Processing PID: {pid}")

    improve_pid(pid, result_dir='result_b', shap_dir='shap_v2')


# ==== From evaluation.ipynb | code cell 11 ====
#v1
pid_list = get_pid_list_from_result_diff("v1", "c")
df1 = evaluate_version_score("v1", "c", pid_list)
pid_list = get_pid_list_from_result_diff("v1", "b")
df2 = evaluate_version_score("v1", "b", pid_list)

df_total = pd.concat([df1, df2], ignore_index=True)

mean_values = df1.mean(numeric_only=True)
print("chopin 평균:\n", mean_values)

mean_values = df2.mean(numeric_only=True)
print("beethoven 평균:\n", mean_values)

mean_values = df_total.mean(numeric_only=True)
print("전체 평균:\n", mean_values)


# ==== From evaluation.ipynb | code cell 12 ====
#v2
pid_list = get_pid_list_from_result_diff("v2", "c")
df1 = evaluate_version_score("v2", "c", pid_list)
pid_list = get_pid_list_from_result_diff("v2", "b")
df2 = evaluate_version_score("v2", "b", pid_list)

df_total = pd.concat([df1, df2], ignore_index=True)

mean_values = df1.mean(numeric_only=True)
print("chopin 평균:\n", mean_values)

mean_values = df2.mean(numeric_only=True)
print("beethoven 평균:\n", mean_values)

mean_values = df_total.mean(numeric_only=True)
print("전체 평균:\n", mean_values)


# ==== From evaluation.ipynb | code cell 13 ====
#v3
pid_list = get_pid_list_from_result_diff("v3", "c")
df1 = evaluate_version_score("v3", "c", pid_list)
pid_list = get_pid_list_from_result_diff("v3", "b")
df2 = evaluate_version_score("v3", "b", pid_list)

df_total = pd.concat([df1, df2], ignore_index=True)

mean_values = df1.mean(numeric_only=True)
print("chopin 평균:\n", mean_values)

mean_values = df2.mean(numeric_only=True)
print("beethoven 평균:\n", mean_values)

mean_values = df_total.mean(numeric_only=True)
print("전체 평균:\n", mean_values)


# ==== From evaluation.ipynb | code cell 14 ====
#v4
pid_list = get_pid_list_from_result_diff("v4", "c")
df1 = evaluate_version_score("v4", "c", pid_list)
pid_list = get_pid_list_from_result_diff("v4", "b")
df2 = evaluate_version_score("v4", "b", pid_list)

df_total = pd.concat([df1, df2], ignore_index=True)

mean_values = df1.mean(numeric_only=True)
print("chopin 평균:\n", mean_values)

mean_values = df2.mean(numeric_only=True)
print("beethoven 평균:\n", mean_values)

mean_values = df_total.mean(numeric_only=True)
print("전체 평균:\n", mean_values)


# ==== From evaluation.ipynb | code cell 15 ====
#v5
pid_list = get_pid_list_from_result_diff("v5", "c")
df1 = evaluate_version_score("v5", "c", pid_list)

mean_values = df1.mean(numeric_only=True)
print("chopin 평균:\n", mean_values)


# ==== From evaluation.ipynb | code cell 16 ====
#v6

pid_list = get_pid_list_from_result_diff("v6", "b")
df2 = evaluate_version_score("v6", "b", pid_list)

mean_values = df2.mean(numeric_only=True)
print("beethoven 평균:\n", mean_values)


# ==== From evaluation.ipynb | code cell 17 ====
df


# ==== From evaluation.ipynb | code cell 18 ====
ver = 'v2'
code = 'c'

pid_list = get_pid_list_from_result_diff(ver, code)

(
    summary_df,
    raw_summary_df,
    detail_feature_df,
    detail_overall_df,
    measure_summary_df,
    measure_stat_summary_df
) = evaluate_segment_metrics(ver, code)


# ==== From evaluation.ipynb | code cell 19 ====
# segment 지표별 real vs gen. real vs guide 비교. 양수이면 real vs gen이 높다는 뜻.

summary_df


# ==== From evaluation.ipynb | code cell 20 ====
# segment 지표별 real vs gen(compare)와 real vs guide(guideline) 결과

raw_summary_df


# ==== From evaluation.ipynb | code cell 21 ====
# segment로 잡힌 measure 개수 통계 차이. standard: real, compare_result: gen, guide_result: guide
measure_stat_summary_df


# ==== From evaluation.ipynb | code cell 23 ====
measure_summary_df


# ==== From evaluation.ipynb | code cell 24 ====
detail_feature_df


# ==== From evaluation.ipynb | code cell 25 ====
detail_overall_df


# ==== From evaluation.ipynb | code cell 26 ====
def evaluate_segment_metrics_combined(version):
    # === 파일 경로 정의
    codes = ['b', 'c']
    compare_dfs, guide_dfs, compare2_dfs, guide2_dfs = [], [], [], []

    for code in codes:
        base = f"{version}_{code}"
        compare_dfs.append(pd.read_csv(f"{base}_compare.csv"))
        guide_dfs.append(pd.read_csv(f"{base}_guide.csv"))
        compare2_dfs.append(pd.read_csv(f"{base}_compare2.csv"))
        guide2_dfs.append(pd.read_csv(f"{base}_guide2.csv"))

    # === concat all dataframes
    compare_df = pd.concat(compare_dfs, ignore_index=True)
    guide_df = pd.concat(guide_dfs, ignore_index=True)
    compare2_df = pd.concat(compare2_dfs, ignore_index=True)
    guide2_df = pd.concat(guide2_dfs, ignore_index=True)
    compare_df = compare_df[compare_df['pid'] != 70]
    guide_df = guide_df[guide_df['pid'] != 70]
    compare2_df = compare2_df[compare2_df['pid'] != 70]
    guide2_df = guide2_df[guide2_df['pid'] != 70]

    # 문자열 리스트 파싱
    for df in [compare2_df, guide2_df]:
        df['standard_measures'] = df['standard_measures'].apply(lambda x: ast.literal_eval(str(x)) if pd.notna(x) else [])
        df['result_measures'] = df['result_measures'].apply(lambda x: ast.literal_eval(str(x)) if pd.notna(x) else [])

    # FEATURE 기준 비교 (A vs B)
    merged_feat = pd.merge(
        compare_df,
        guide_df,
        on=['pid', 'feature'],
        suffixes=('_compare', '_guide')
    )

    # MEASURE 기준 비교 (C vs D)
    merged_all = pd.merge(
        compare2_df,
        guide2_df,
        on='pid',
        suffixes=('_compare', '_guide')
    )

    # 비교 지표
    metrics = ['IoU', 'Precision', 'Recall', 'F1', 'Coverage']

    # 차이 계산
    for metric in metrics:
        merged_feat[f'{metric}_diff'] = merged_feat[f'{metric}_compare'] - merged_feat[f'{metric}_guide']
        merged_all[f'{metric}_diff'] = merged_all[f'{metric}_compare'] - merged_all[f'{metric}_guide']

    # Summary of diffs
    def summarize_diffs(df, level):
        return {
            'level': level,
            **{f'{metric}_diff_mean': df[f'{metric}_diff'].mean() for metric in metrics}
        }

    summary_df = pd.DataFrame([
        summarize_diffs(merged_feat, 'feature'),
        summarize_diffs(merged_all, 'overall')
    ])

    # Summary of raw metrics
    def summarize_metrics(df, source, level):
        return {
            'source': source,
            'level': level,
            **{f'{metric}_mean': df[metric].mean() for metric in metrics if metric in df.columns}
        }

    raw_summary_df = pd.DataFrame([
        summarize_metrics(compare_df, 'compare', 'feature'),
        summarize_metrics(guide_df, 'guide', 'feature'),
        summarize_metrics(compare2_df, 'compare', 'overall'),
        summarize_metrics(guide2_df, 'guide', 'overall'),
    ])

    # measure 수 요약
    def measure_lengths(df, source):
        return pd.DataFrame({
            'pid': df['pid'],
            'standard_len': df['standard_measures'].apply(len),
            'result_len': df['result_measures'].apply(len),
            'source': source
        })

    compare2_lengths = measure_lengths(compare2_df, 'compare')
    guide2_lengths = measure_lengths(guide2_df, 'guide')
    measure_summary_df = pd.concat([compare2_lengths, guide2_lengths], ignore_index=True)

    # 통계 집계
    measure_stat_summary_df = measure_summary_df.groupby('source')[['standard_len', 'result_len']].agg(['mean', 'min', 'max', 'std', 'var'])
    measure_stat_summary_df.columns = ['_'.join(col) for col in measure_stat_summary_df.columns]
    measure_stat_summary_df = measure_stat_summary_df.reset_index()

    return summary_df, raw_summary_df, merged_feat, merged_all, measure_summary_df, measure_stat_summary_df


# ==== From evaluation.ipynb | code cell 27 ====
summary_df, raw_summary_df, merged_feat, merged_all, measure_summary_df, measure_stat_summary_df = evaluate_segment_metrics_combined('v1')


# ==== From evaluation.ipynb | code cell 28 ====
raw_summary_df


# ==== From evaluation.ipynb | code cell 29 ====
measure_stat_summary_df

