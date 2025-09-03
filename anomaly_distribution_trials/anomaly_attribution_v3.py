# =============== AGGREGATED ANOMALY SCORE PIPELINE ===============
# Classifies each row by calculating its total anomaly score under each model
# using gcm.attribute_anomalies. The model that finds the row less anomalous
# (i.e., assigns a lower score) is the winner.

import os, time, json, multiprocessing as mp
os.environ["TQDM_DISABLE"] = "0"

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from dowhy import gcm

# ============================ ANOMALY SCORE UTILS ============================

def topk_influenced_targets_from_adj(adj_df: pd.DataFrame, k=5, threshold=0.3):
    numeric = adj_df.astype(float).fillna(0.0)
    w = numeric.where(numeric.abs() > threshold, 0.0).abs()
    in_weight_sum = w.sum(axis=1)
    return list(in_weight_sum.sort_values(ascending=False).head(k).index)

def attribution_dict_to_vector(attr_dict, variable_names):
    return np.array([float(attr_dict.get(v, np.array([0.0]))[0]) for v in variable_names], dtype=float)

def classify_row_by_anomaly_score(
    row_df, model_B, model_M,
    variable_names,
    T_ben, T_mal,
    tie_break="malicious",
    return_details=True
):
    """
    Classifies a row by aggregating the magnitude of its anomaly attributions.
    The model that assigns a lower total anomaly score is the better fit.
    """
    # 1. Calculate total anomaly score using the benign model (keeping signs)
    total_score_B = 0
    for t in T_ben:
        attr = gcm.attribute_anomalies(model_B, t, anomaly_samples=row_df)
        vec = attribution_dict_to_vector(attr, variable_names)
        total_score_B += np.sum(vec)  # Sum of signed values

    # 2. Calculate total anomaly score using the malicious model (keeping signs)
    total_score_M = 0
    for t in T_mal:
        attr = gcm.attribute_anomalies(model_M, t, anomaly_samples=row_df)
        vec = attribution_dict_to_vector(attr, variable_names)
        total_score_M += np.sum(vec)  # Sum of signed values

    # 3. Classify based on the comparison of the total anomaly scores
    if total_score_M > total_score_B:
        label = "malicious"
    else:
        label = "benign"

    res = {
        "TotalScore_B": total_score_B,
        "TotalScore_M": total_score_M,
        "label": label
    }
    return res if return_details else label


# ===================================== DATA & SPLIT =====================================
SEED = 42
TRAIN_FRAC = 0.8
RESULTS_CSV = "./row_results_anomaly_score_simple_adding.csv"
CONFUSION_2x2_CSV = "./confusion_matrix_2x2_anomaly_score_simple_adding.csv"

print("[INFO] Loading and shuffling data…")
df = pd.read_csv('./UNSW_NB15_freq_scaled.csv')
df = df.drop(columns=['attack_cat'], errors='ignore')
assert 'label' in df.columns, "Expected a 'label' column."

df0 = df[df['label']==0].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
df1 = df[df['label']==1].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n0, n1 = len(df0), len(df1)
n0_tr = int(TRAIN_FRAC*n0); n1_tr = int(TRAIN_FRAC*n1)
train_df = pd.concat([df0.iloc[:n0_tr], df1.iloc[:n1_tr]], axis=0).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
test_df  = pd.concat([df0.iloc[n0_tr:], df1.iloc[n1_tr:]], axis=0).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print(f"[INFO] Train size: {len(train_df)} | Test size: {len(test_df)}")

feature_names = [c for c in df.columns if c != 'label']

# ===================================== GRAPHS & MODELS =====================================
print("\n[INFO] Loading adjacency matrices…")
adj_ben = pd.read_csv('./benign_culingam_adjacency_matrix_gpu.csv', index_col=0)
adj_mal = pd.read_csv('./malicious_culingam_adjacency_matrix_gpu.csv', index_col=0)

train_ben = train_df[train_df['label']==0][feature_names]
train_mal = train_df[train_df['label']==1][feature_names]
test_X = test_df[feature_names]
test_y = test_df['label'].values

print("\n[INFO] Fitting causal models on TRAIN…")
t0 = time.time()
G_benign = nx.from_pandas_adjacency(adj_ben.T, create_using=nx.DiGraph)
G_malicious = nx.from_pandas_adjacency(adj_mal.T, create_using=nx.DiGraph)
model_B = gcm.InvertibleStructuralCausalModel(G_benign)
model_M = gcm.InvertibleStructuralCausalModel(G_malicious)
gcm.auto.assign_causal_mechanisms(model_B, train_ben); gcm.fit(model_B, train_ben)
gcm.auto.assign_causal_mechanisms(model_M, train_mal); gcm.fit(model_M, train_mal)
print(f"[INFO] Models fitted in {time.time()-t0:.2f}s")

K_REF = 3
print("\n[INFO] Selecting top-K targets from adjacencies…")
T_ben = topk_influenced_targets_from_adj(adj_ben, k=K_REF, threshold=0.5)
T_mal = topk_influenced_targets_from_adj(adj_mal, k=K_REF, threshold=0.5)
print("    T_ben:", T_ben)
print("    T_mal:", T_mal)
print("\n[INFO] Setup complete. Proceeding to classification.")


# ===================================== EVALUATION (PARALLEL) =====================================
_GLOBALS = {
    "model_B": model_B, "model_M": model_M,
    "feature_names": feature_names,
    "T_ben": T_ben, "T_mal": T_mal,
    "test_X": test_X, "test_y": test_y, "test_df": test_df
}

def _classify_row_index(i):
    row_df = _GLOBALS["test_X"].iloc[[i]]
    res = classify_row_by_anomaly_score(
        row_df,
        _GLOBALS["model_B"], _GLOBALS["model_M"],
        _GLOBALS["feature_names"],
        _GLOBALS["T_ben"], _GLOBALS["T_mal"],
        tie_break="malicious", return_details=True
    )
    return {
        "row_index": int(_GLOBALS["test_df"].index[i]),
        "true_label": int(_GLOBALS["test_y"][i]),
        "true_name": "benign" if _GLOBALS["test_y"][i]==0 else "malicious",
        "pred_label": res["label"],
        "TotalScore_B": res["TotalScore_B"],
        "TotalScore_M": res["TotalScore_M"]
    }

print(f"\n[INFO] Classifying test rows using aggregated anomaly scores…")
indices = list(range(len(test_X)))
n_cpu = os.cpu_count() or 2
n_proc = max(1, n_cpu // 2)
print(f"[INFO] CPUs={n_cpu}. Using n/2={n_proc} workers for classification.")

t0 = time.time()
ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
with ctx.Pool(processes=n_proc) as pool:
    results = pool.map(_classify_row_index, indices)
elapsed = time.time() - t0
print(f"[INFO] Done in {elapsed:.2f}s  ({len(indices)/max(elapsed,1e-9):.1f} rows/sec).")


# ============================ SAVE RESULTS & CONFUSION MATRIX ============================
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_CSV, index=False)
print(f"\n[INFO] Saved row results → {RESULTS_CSV}")

labels = ["benign", "malicious"]
cm2 = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
for t, p in zip(results_df["true_name"], results_df["pred_label"]):
    cm2.loc[t, p] += 1
cm2.to_csv(CONFUSION_2x2_CSV)
print(f"\n[INFO] 2×2 confusion matrix:\n{cm2}\nSaved → {CONFUSION_2x2_CSV}")

total = cm2.values.sum()
acc = (np.trace(cm2.values)/total) if total else np.nan
tp_b = cm2.loc["benign","benign"]; fp_b = cm2.loc["malicious","benign"]; fn_b = cm2.loc["benign","malicious"]
tp_m = cm2.loc["malicious","malicious"]; fp_m = cm2.loc["benign","malicious"]; fn_m = cm2.loc["malicious","benign"]
prec_b = tp_b / max(tp_b + fp_b, 1); rec_b = tp_b / max(tp_b + fn_b, 1)
prec_m = tp_m / max(tp_m + fp_m, 1); rec_m = tp_m / max(tp_m + fn_m, 1)

print(f"\n[INFO] Accuracy={acc:.4f} | Benign P/R={prec_b:.3f}/{rec_b:.3f} | Malicious P/R={prec_m:.3f}/{rec_m:.3f}")
