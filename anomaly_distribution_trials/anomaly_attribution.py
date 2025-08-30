# ========================== FAST TWO-TEMPLATE PIPELINE (NO-ABSTAIN) ==========================
# Caps gcm.attribute_anomalies usage:
#   - Per row: ≤ 2 * |T*| calls where T* = top5_ben ∪ top5_mal  (|T*| ≤ 10)  → ≤ 20 calls/row
#   - Reference build: only those targets, small sample_n (default 20)
#
# Does:
#   - read, shuffle, stratified train/test
#   - build G_benign / G_malicious from adjacencies
#   - summarize references for selected targets only
#   - classify all test rows (no abstain), multiprocessing with n/2 CPUs
#   - save row results CSV + 2×2 confusion matrix CSV
#
# NOTE: analyze_causal_influences is kept exactly as-is.

import os, time, json, multiprocessing as mp
#os.environ["TQDM_DISABLE"] = "1"   # try to silence tqdm if DoWhy uses it

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from dowhy import gcm

# ----------------------------- DO NOT EDIT: original function -----------------------------
def analyze_causal_influences(adj_df, threshold=0.05, top_n=10):
    """
    Analyzes a causal adjacency matrix to identify both target and source nodes
    based on a LiNGAM-based causal graph.

    This function is corrected to correctly interpret the LiNGAM adjacency matrix
    convention (cause -> effect is represented by [i, j] for effect i and cause j).

    Args:
        adj_df (pd.DataFrame): The causal adjacency matrix.
        threshold (float, optional): The minimum absolute weight to be considered an edge.
                                     Defaults to 0.05.
        top_n (int, optional): The number of top nodes to print for each ranking.
                               Defaults to 10.
    """
    try:
        numeric_adj_df = adj_df.astype(float).fillna(0)
    except ValueError as e:
        print(f"Error converting DataFrame to numeric: {e}")
        print("Please ensure your CSV only contains numeric values or can be converted.")
        return

    G = nx.DiGraph()
    for i in numeric_adj_df.index:
        for j in numeric_adj_df.columns:
            weight = numeric_adj_df.loc[i, j]
            if abs(weight) > threshold:
                G.add_edge(j, i, weight=weight)

    in_degrees = dict(G.in_degree())
    in_weights = {node: 0.0 for node in G.nodes()}
    for u, v, data in G.edges(data=True):
        in_weights[v] += abs(data.get('weight', 0))

    out_degrees = dict(G.out_degree())
    out_weights = {node: 0.0 for node in G.nodes()}
    for u, v, data in G.edges(data=True):
        out_weights[u] += abs(data.get('weight', 0))

    in_degree_series = pd.Series(in_degrees).sort_values(ascending=False)
    in_weight_series = pd.Series(in_weights).sort_values(ascending=False)
    out_degree_series = pd.Series(out_degrees).sort_values(ascending=False)
    out_weight_series = pd.Series(out_weights).sort_values(ascending=False)

    print(f"\n--- Causal Influence Analysis with Threshold={threshold} ---")
    print(f"\nTop {top_n} Nodes by In-Degree (Most Influenced/Received):")
    print(in_degree_series.head(top_n))
    print(f"\nTop {top_n} Nodes by Incoming Weight Sum (Most Influenced/Received, by strength):")
    print(in_weight_series.head(top_n))
    print(f"\nTop {top_n} Nodes by Out-Degree (Most Influential/Source):")
    print(out_degree_series.head(top_n))
    print(f"\nTop {top_n} Nodes by Outgoing Weight Sum (Most Influential/Source, by strength):")
    print(out_weight_series.head(top_n))

    return in_degree_series.head(top_n)
# -------------------------------------------------------------------------------------------

def visualize_causal_graph(adjacency_matrix, variable_names, threshold=0.01):
    """Draw causal graph thresholded (optional; safe to comment out to save time)."""
    if isinstance(adjacency_matrix, pd.DataFrame):
        adjacency_matrix = adjacency_matrix.to_numpy()
    G = nx.DiGraph()
    G.add_nodes_from(variable_names)
    n = len(variable_names)
    for i in range(n):
        for j in range(n):
            if abs(adjacency_matrix[i, j]) > threshold:
                G.add_edge(variable_names[j], variable_names[i], weight=adjacency_matrix[i, j])
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1200, font_size=7, arrowsize=15)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={e:f"{w:.2f}" for e,w in edge_labels.items()}, font_color='red', font_size=6)
    plt.title("Causal Graph (thresholded)")
    plt.show()
    return G

# ============================== FAST UTILS (no-ML, no-abstain) ==============================

def _robust_z(x, median, mad, eps=1e-9):
    denom = mad if mad > eps else eps
    return (x - median) / denom

def _gini(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0: return 0.0
    x = np.abs(x)
    if np.all(x == 0): return 0.0
    x_sorted = np.sort(x); n = x.size
    cum = np.cumsum(x_sorted)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)

def _normalize_abs(vec, eps=1e-12):
    v = np.abs(np.asarray(vec, dtype=float)); s = v.sum()
    return (np.ones_like(v) / max(1, len(v))) if s <= eps else v / s

def _jsd(p, q, eps=1e-12):
    p = np.asarray(p, float) + eps; q = np.asarray(q, float) + eps
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * (np.log2(p) - np.log2(m))) + np.sum(q * (np.log2(q) - np.log2(m))))

def _jaccard_distance(A, B):
    A, B = set(A), set(B)
    if not A and not B: return 0.0
    return 1.0 - (len(A & B) / len(A | B))

def topk_influenced_targets_from_adj(adj_df: pd.DataFrame, k=5, threshold=0.3):
    """Top-k targets by incoming abs weight above threshold."""
    numeric = adj_df.astype(float).fillna(0.0)
    w = numeric.where(numeric.abs() > threshold, 0.0).abs()
    in_weight_sum = w.sum(axis=1)
    return list(in_weight_sum.sort_values(ascending=False).head(k).index)

def attribution_dict_to_vector(attr_dict, variable_names):
    return np.array([float(attr_dict.get(v, np.array([0.0]))[0]) for v in variable_names], dtype=float)

def row_attribution(model, target, row_df, variable_names):
    attr = gcm.attribute_anomalies(model, target, anomaly_samples=row_df)
    return attribution_dict_to_vector(attr, variable_names)  # signed vector

def weighted_sign_flip_rate(cur_signed_vec, ref_sign_vec, ref_weights):
    cur_s = np.sign(cur_signed_vec); ref_s = np.sign(ref_sign_vec); w = np.asarray(ref_weights, float)
    mask = w > 0
    if not np.any(mask): return 0.0
    flips = (cur_s != ref_s) & (ref_s != 0) & (cur_s != 0)
    return float(np.sum(w[mask] * flips[mask]) / np.sum(w[mask]))

def per_target_mismatch(cur_signed_vec, ref_stats):
    v_abs = np.abs(cur_signed_vec)
    L1 = float(v_abs.sum()); G = _gini(v_abs)
    q = _normalize_abs(v_abs); p_ref = ref_stats["p"]
    zL1 = _robust_z(L1, ref_stats["med_L1"], ref_stats["mad_L1"])
    zG  = _robust_z(G,  ref_stats["med_G"],  ref_stats["mad_G"])
    D   = _jsd(p_ref, q)
    F   = weighted_sign_flip_rate(cur_signed_vec, ref_stats["sign_med"], p_ref)
    return float(zL1 + zG + D + F)

def summarize_reference_targets(model, df, variable_names, targets, sample_n=20, random_state=0):
    """
    FAST reference build: only for 'targets' (e.g., top-5 per regime) and only 'sample_n' rows.
    """
    rng = np.random.RandomState(random_state)
    n = len(df); idx = np.arange(n)
    if n > sample_n:
        idx = rng.choice(idx, size=sample_n, replace=False)

    print(f"[INFO] summarize_reference_targets: targets={len(targets)}, sample_n={len(idx)} → "
          f"~{len(targets)*len(idx)} gcm calls for this model.")

    L1, Gval, abs_stack, sign_stack = {}, {}, {}, {}
    for t in targets:
        L1[t], Gval[t], abs_stack[t], sign_stack[t] = [], [], [], []

    for c, i in enumerate(idx, 1):
        if c % 10 == 0 or c == len(idx):
            print(f"    processed {c}/{len(idx)} rows for reference…")
        row_df = df.iloc[i:i+1, :][variable_names]
        for t in targets:
            v = row_attribution(model, t, row_df, variable_names)
            v_abs = np.abs(v)
            L1[t].append(float(v_abs.sum()))
            Gval[t].append(_gini(v_abs))
            abs_stack[t].append(v_abs)
            sign_stack[t].append(np.sign(v))

    ref = {}
    d = len(variable_names)
    for t in targets:
        L1_arr = np.asarray(L1[t], float); G_arr = np.asarray(Gval[t], float)
        abs_med = np.median(np.vstack(abs_stack[t]), axis=0) if len(abs_stack[t]) else np.zeros(d)
        p_t = _normalize_abs(abs_med)
        sign_med = np.sign(np.median(np.vstack(sign_stack[t]), axis=0)) if len(sign_stack[t]) else np.zeros(d)
        med_L1 = float(np.median(L1_arr)) if L1_arr.size else 0.0
        mad_L1 = float(np.median(np.abs(L1_arr - med_L1))) if L1_arr.size else 1.0
        med_G  = float(np.median(G_arr)) if G_arr.size else 0.0
        mad_G  = float(np.median(np.abs(G_arr - med_G))) if G_arr.size else 1.0
        ref[t] = {"p": p_t, "med_L1": med_L1, "mad_L1": mad_L1,
                  "med_G": med_G, "mad_G": mad_G, "sign_med": sign_med}
    return ref

def classify_row_two_templates_fast(
    row_df, model_B, model_M,
    ref_B, ref_M,
    variable_names,
    T_ben_ref, T_mal_ref,
    k_for_setdist=5,
    lambda_setdist=1.0,
    tie_break="malicious",
    return_details=True
):
    """
    FAST, no-abstain:
      - T* = T_ben_ref ∪ T_mal_ref  (|T*| ≤ 10)
      - For each t in T*: compute vB[t], vM[t] once (≤20 gcm calls total)
      - Build T_x (for set distance) from those already-computed vectors only
      - Decide by smaller mismatch (ties → tie_break)
    """
    T_star = list(set(T_ben_ref) | set(T_mal_ref))
    # 1) compute per-target attributions ONCE per model
    vB = {}; vM = {}
    for t in T_star:
        vB[t] = row_attribution(model_B, t, row_df[variable_names], variable_names)
        vM[t] = row_attribution(model_M, t, row_df[variable_names], variable_names)

    # 2) per-model mismatches using available refs (fallback to neutral if missing)
    d = len(variable_names)
    neutral = {"p": np.ones(d)/d, "med_L1":0.0, "mad_L1":1.0, "med_G":0.0, "mad_G":1.0, "sign_med": np.zeros(d)}

    scores_B = []; scores_M = []
    for t in T_star:
        ref_b = ref_B.get(t, neutral); ref_m = ref_M.get(t, neutral)
        scores_B.append(per_target_mismatch(vB[t], ref_b))
        scores_M.append(per_target_mismatch(vM[t], ref_m))

    # Robust aggregate: max (since |T*| ≤ 10; you can swap to percentile if you like)
    attrMismatch_B = float(np.max(scores_B)) if scores_B else 0.0
    attrMismatch_M = float(np.max(scores_M)) if scores_M else 0.0

    # 3) Build T_x from L1 strengths we already computed (NO extra gcm calls)
    L1_strength = {t: max(np.abs(vB[t]).sum(), np.abs(vM[t]).sum()) for t in T_star}
    T_x = [t for t,_ in sorted(L1_strength.items(), key=lambda kv: kv[1], reverse=True)[:k_for_setdist]]

    setDist_B = _jaccard_distance(T_ben_ref, T_x)
    setDist_M = _jaccard_distance(T_mal_ref, T_x)

    mismatch_B = float(attrMismatch_B + lambda_setdist * setDist_B)
    mismatch_M = float(attrMismatch_M + lambda_setdist * setDist_M)

    label = "benign" if mismatch_B < mismatch_M else ("malicious" if mismatch_M < mismatch_B else tie_break)

    res = {
        "Mismatch_B": mismatch_B, "Mismatch_M": mismatch_M,
        "AttrMismatch_B": attrMismatch_B, "AttrMismatch_M": attrMismatch_M,
        "SetDist_B": setDist_B, "SetDist_M": setDist_M,
        "T_x": T_x, "T_star": T_star, "label": label
    }
    return res if return_details else label

# ===================================== DATA & SPLIT =====================================

SEED = 42
TRAIN_FRAC = 0.8
RESULTS_CSV = "./row_results_two_template_fast.csv"
CONFUSION_2x2_CSV = "./confusion_matrix_2x2_fast.csv"

print("[INFO] Loading and shuffling data…")
df = pd.read_csv('./UNSW_NB15_freq_scaled.csv')
if 'attack_cat' in df.columns:
    df = df.drop(columns=['attack_cat'])
assert 'label' in df.columns, "Expected a 'label' column (0=benign, 1=malicious)."

# Stratified shuffle + split
df0 = df[df['label']==0].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
df1 = df[df['label']==1].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n0, n1 = len(df0), len(df1)
n0_tr = int(TRAIN_FRAC*n0); n1_tr = int(TRAIN_FRAC*n1)

train_df = pd.concat([df0.iloc[:n0_tr], df1.iloc[:n1_tr]], axis=0).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
test_df  = pd.concat([df0.iloc[n0_tr:], df1.iloc[n1_tr:]], axis=0).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print(f"[INFO] Train size: {len(train_df)} (benign={n0_tr}, malicious={n1_tr})")
print(f"[INFO] Test  size: {len(test_df)} (benign={(test_df.label==0).sum()}, malicious={(test_df.label==1).sum()})")

feature_names = [c for c in df.columns if c != 'label']

# ===================================== GRAPHS & MODELS =====================================

print("[INFO] Loading adjacency matrices…")
adj_ben = pd.read_csv('./benign_culingam_adjacency_matrix_gpu.csv', index_col=0)
adj_mal = pd.read_csv('./malicious_culingam_adjacency_matrix_gpu.csv', index_col=0)

# (Optional) draw graphs once
# visualize_causal_graph(adj_ben, variable_names=feature_names, threshold=0.3)
# visualize_causal_graph(adj_mal, variable_names=feature_names, threshold=0.3)

train_ben = train_df[train_df['label']==0][feature_names]
train_mal = train_df[train_df['label']==1][feature_names]
test_X = test_df[feature_names]
test_y = test_df['label'].values

print("[INFO] Fitting causal models on TRAIN…")
t0 = time.time()
model_B = gcm.InvertibleStructuralCausalModel(nx.DiGraph())
model_M = gcm.InvertibleStructuralCausalModel(nx.DiGraph())
# Use graph structures from adj matrices:
G_benign = visualize_causal_graph(adj_ben, variable_names=feature_names, threshold=0.3)
G_malicious = visualize_causal_graph(adj_mal, variable_names=feature_names, threshold=0.3)
model_B = gcm.InvertibleStructuralCausalModel(G_benign)
model_M = gcm.InvertibleStructuralCausalModel(G_malicious)

gcm.auto.assign_causal_mechanisms(model_B, train_ben); gcm.fit(model_B, train_ben)
gcm.auto.assign_causal_mechanisms(model_M, train_mal); gcm.fit(model_M, train_mal)
print(f"[INFO] Models fitted in {time.time()-t0:.2f}s")

# Reference target sets (K=5 each) and union cap
K_REF = 3
print("[INFO] Selecting top-K targets from adjacencies (no per-row scan)…")
T_ben = topk_influenced_targets_from_adj(adj_ben, k=K_REF, threshold=0.3)
T_mal = topk_influenced_targets_from_adj(adj_mal, k=K_REF, threshold=0.3)
print("    T_ben:", T_ben)
print("    T_mal:", T_mal)
print(f"    |T*| (union) ≤ {len(set(T_ben)|set(T_mal))} (cap 10)")

print("[INFO] Building FAST references only for selected targets…")
t0 = time.time()
# small sample for speed; tweak if you want more stability
ref_B = summarize_reference_targets(model_B, train_ben, feature_names, targets=T_ben, sample_n=1, random_state=SEED)
ref_M = summarize_reference_targets(model_M, train_mal, feature_names, targets=T_mal, sample_n=1, random_state=SEED)
print(f"[INFO] Reference build done in {time.time()-t0:.2f}s")

# ===================================== EVALUATION (PARALLEL) =====================================

_GLOBALS = {
    "model_B": model_B, "model_M": model_M,
    "ref_B": ref_B, "ref_M": ref_M,
    "feature_names": feature_names,
    "T_ben": T_ben, "T_mal": T_mal,
    "test_X": test_X, "test_y": test_y, "test_df": test_df
}

def _classify_row_index(i):
    row_df = _GLOBALS["test_X"].iloc[[i]]
    res = classify_row_two_templates_fast(
        row_df,
        _GLOBALS["model_B"], _GLOBALS["model_M"],
        _GLOBALS["ref_B"], _GLOBALS["ref_M"],
        _GLOBALS["feature_names"],
        _GLOBALS["T_ben"], _GLOBALS["T_mal"],
        k_for_setdist=5, lambda_setdist=1.0, tie_break="malicious", return_details=True
    )
    return {
        "row_index": int(_GLOBALS["test_df"].index[i]),
        "true_label": int(_GLOBALS["test_y"][i]),
        "true_name": "benign" if _GLOBALS["test_y"][i]==0 else "malicious",
        "pred_label": res["label"],
        "Mismatch_B": res["Mismatch_B"], "Mismatch_M": res["Mismatch_M"],
        "AttrMismatch_B": res["AttrMismatch_B"], "AttrMismatch_M": res["AttrMismatch_M"],
        "SetDist_B": res["SetDist_B"], "SetDist_M": res["SetDist_M"],
        "T_x": json.dumps(res["T_x"]), "T_star": json.dumps(res["T_star"])
    }

print("[INFO] Classifying test rows (≤ 20 gcm calls per row)…")
indices = list(range(len(test_X))); results = []
n_cpu = os.cpu_count() or 2; n_proc = max(1, n_cpu // 2)
print(f"[INFO] CPUs={n_cpu}. Using n/2={n_proc} workers when possible.")
use_parallel = ("fork" in mp.get_all_start_methods())

t0 = time.time()
if use_parallel:
    print("[INFO] Multiprocessing with 'fork'.")
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_proc) as pool:
        chunk = 200
        for j in range(0, len(indices), chunk):
            part = indices[j:j+chunk]
            results.extend(pool.map(_classify_row_index, part))
            print(f"    processed {min(j+chunk, len(indices))}/{len(indices)} rows…")
else:
    print("[WARN] 'fork' not available; single-process fallback.")
    for i, idx in enumerate(indices, 1):
        results.append(_classify_row_index(idx))
        if i % 200 == 0 or i == len(indices):
            print(f"    processed {i}/{len(indices)} rows…")

elapsed = time.time() - t0
print(f"[INFO] Done in {elapsed:.2f}s  ({len(indices)/max(elapsed,1e-9):.1f} rows/sec).")

# ============================ SAVE RESULTS & CONFUSION MATRIX ============================

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_CSV, index=False)
print(f"[INFO] Saved row results → {RESULTS_CSV}")
print(results_df.head(5))

labels = ["benign", "malicious"]
cm2 = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
for t, p in zip(results_df["true_name"], results_df["pred_label"]):
    cm2.loc[t, p] += 1
cm2.to_csv(CONFUSION_2x2_CSV)
print(f"[INFO] 2×2 confusion matrix:\n{cm2}\nSaved → {CONFUSION_2x2_CSV}")

total = cm2.values.sum(); acc = (np.trace(cm2.values)/total) if total else np.nan
tp_b = cm2.loc["benign","benign"]; fp_b = cm2.loc["malicious","benign"]; fn_b = cm2.loc["benign","malicious"]
tp_m = cm2.loc["malicious","malicious"]; fp_m = cm2.loc["benign","malicious"]; fn_m = cm2.loc["malicious","benign"]
prec_b = tp_b / max(tp_b + fp_b, 1); rec_b = tp_b / max(tp_b + fn_b, 1)
prec_m = tp_m / max(tp_m + fp_m, 1); rec_m = tp_m / max(tp_m + fn_m, 1)

print(f"[INFO] Accuracy={acc:.4f} | Benign P/R={prec_b:.3f}/{rec_b:.3f} | Malicious P/R={prec_m:.3f}/{rec_m:.3f}")
# ========================================= END ============================================
