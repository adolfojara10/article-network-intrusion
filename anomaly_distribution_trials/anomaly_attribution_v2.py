# ================== SIGNATURE MATCHING PIPELINE (OPTIMIZED PARALLEL) ==================
# Implements classification by comparing a row's causal attribution signature to
# master signatures for benign and malicious classes.
#
# OPTIMIZATIONS:
#   - Master signature creation is now parallelized (2 workers).
#   - Final row classification remains parallelized (n/2 workers).
#
# NOTE: analyze_causal_influences is kept exactly as-is.

import os, time, json, multiprocessing as mp
os.environ["TQDM_DISABLE"] = "1"   # try to silence tqdm if DoWhy uses it

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
    """
    try:
        numeric_adj_df = adj_df.astype(float).fillna(0)
    except ValueError as e:
        print(f"Error converting DataFrame to numeric: {e}")
        return

    G = nx.DiGraph()
    for i in numeric_adj_df.index:
        for j in numeric_adj_df.columns:
            weight = numeric_adj_df.loc[i, j]
            if abs(weight) > threshold:
                G.add_edge(j, i, weight=weight) # cause j -> effect i

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
    #plt.title("Causal Graph (thresholded)")
    #plt.show()
    return G

# ============================ SIGNATURE MATCHING UTILS ✍️ ============================

def _cosine_similarity(v1, v2, eps=1e-9):
    """Computes cosine similarity between two vectors."""
    v1 = np.asarray(v1, dtype=float); v2 = np.asarray(v2, dtype=float)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    denominator = norm_v1 * norm_v2
    return dot_product / (denominator + eps)

def topk_influenced_targets_from_adj(adj_df: pd.DataFrame, k=5, threshold=0.3):
    """Top-k targets by incoming abs weight above threshold."""
    numeric = adj_df.astype(float).fillna(0.0)
    w = numeric.where(numeric.abs() > threshold, 0.0).abs()
    in_weight_sum = w.sum(axis=1)
    return list(in_weight_sum.sort_values(ascending=False).head(k).index)

def attribution_dict_to_vector(attr_dict, variable_names):
    """Converts DoWhy's attribution dictionary to a numpy vector."""
    return np.array([float(attr_dict.get(v, np.array([0.0]))[0]) for v in variable_names], dtype=float)

def row_attribution(model, target, row_df, variable_names):
    """Wrapper for a single gcm.attribute_anomalies call."""
    attr = gcm.attribute_anomalies(model, target, anomaly_samples=row_df)
    return attribution_dict_to_vector(attr, variable_names)

def create_master_signature(model, df, variable_names, targets, sample_n, random_state, class_name):
    """
    Creates a single, master attribution signature vector for a class.
    This is done by averaging attribution vectors from a sample of the training data.
    """
    rng = np.random.RandomState(random_state)
    n = len(df)
    idx = np.arange(n)
    if n > sample_n:
        idx = rng.choice(idx, size=sample_n, replace=False)

    print(f"[INFO] Building '{class_name}' signature: targets={len(targets)}, sample_n={len(idx)} → "
          f"~{len(targets) * len(idx)} gcm calls.")

    all_vectors = []
    # Simplified print statement for parallel execution
    print(f"    Processing {len(idx)} rows for '{class_name}' signature...")
    for i in idx:
        row_df = df.iloc[i:i+1, :][variable_names]
        for t in targets:
            v = row_attribution(model, t, row_df, variable_names)
            all_vectors.append(v)

    if not all_vectors:
        return np.zeros(len(variable_names))

    master_signature = np.mean(np.array(all_vectors), axis=0)
    print(f"    '{class_name}' signature built.")
    return master_signature

def classify_row_by_signature(
    row_df, model_B, model_M,
    master_sig_B, master_sig_M,
    variable_names,
    T_ben, T_mal,
    tie_break="malicious",
    return_details=True
):
    """
    Classifies a row by comparing its generated signatures to the master signatures.
    """
    # 1. Generate the row's signature under the benign model
    row_vectors_b = [row_attribution(model_B, t, row_df[variable_names], variable_names) for t in T_ben]
    row_sig_B = np.mean(np.array(row_vectors_b), axis=0) if row_vectors_b else np.zeros(len(variable_names))

    # 2. Generate the row's signature under the malicious model
    row_vectors_m = [row_attribution(model_M, t, row_df[variable_names], variable_names) for t in T_mal]
    row_sig_M = np.mean(np.array(row_vectors_m), axis=0) if row_vectors_m else np.zeros(len(variable_names))

    # 3. Compare signatures using Cosine Similarity
    sim_B = _cosine_similarity(row_sig_B, master_sig_B)
    sim_M = _cosine_similarity(row_sig_M, master_sig_M)

    # 4. Classify based on higher similarity
    label = "benign" if sim_B > sim_M else ("malicious" if sim_M > sim_B else tie_break)

    res = {
        "Similarity_B": sim_B, "Similarity_M": sim_M, "label": label
    }
    return res if return_details else label

# ===================================== DATA & SPLIT =====================================

SEED = 42
TRAIN_FRAC = 0.8
RESULTS_CSV = "./row_results_signature_mp.csv"
CONFUSION_2x2_CSV = "./confusion_matrix_2x2_signature_mp.csv"

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

train_ben = train_df[train_df['label']==0][feature_names]
train_mal = train_df[train_df['label']==1][feature_names]
test_X = test_df[feature_names]
test_y = test_df['label'].values

print("[INFO] Fitting causal models on TRAIN…")
t0 = time.time()
G_benign = visualize_causal_graph(adj_ben, variable_names=feature_names, threshold=0.5)
G_malicious = visualize_causal_graph(adj_mal, variable_names=feature_names, threshold=0.5)
plt.close('all') # Close plots to prevent them from showing during script run

model_B = gcm.InvertibleStructuralCausalModel(G_benign)
model_M = gcm.InvertibleStructuralCausalModel(G_malicious)

gcm.auto.assign_causal_mechanisms(model_B, train_ben); gcm.fit(model_B, train_ben)
gcm.auto.assign_causal_mechanisms(model_M, train_mal); gcm.fit(model_M, train_mal)
print(f"[INFO] Models fitted in {time.time()-t0:.2f}s")

K_REF = 3
print("[INFO] Selecting top-K targets from adjacencies…")
T_ben = topk_influenced_targets_from_adj(adj_ben, k=K_REF, threshold=0.5)
T_mal = topk_influenced_targets_from_adj(adj_mal, k=K_REF, threshold=0.5)
print("    T_ben:", T_ben)
print("    T_mal:", T_mal)

# ========================== PARALLEL SIGNATURE CREATION ===========================
# Helper function for multiprocessing pool
def _create_signature_task(args):
    return create_master_signature(*args)

print("\n[INFO] Building master signatures in parallel (2 workers)…")
t0 = time.time()
use_parallel_build = ("fork" in mp.get_all_start_methods())
sample_n_ref = 1 # small sample for speed; tweak if you want more stability

benign_args = (model_B, train_ben, feature_names, T_ben, sample_n_ref, SEED, "benign")
malicious_args = (model_M, train_mal, feature_names, T_mal, sample_n_ref, SEED, "malicious")
task_args = [benign_args, malicious_args]

if use_parallel_build:
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=2) as pool:
        results = pool.map(_create_signature_task, task_args)
    master_sig_B, master_sig_M = results[0], results[1]
else:
    print("[WARN] 'fork' not available for signature build; using single-process fallback.")
    master_sig_B = _create_signature_task(benign_args)
    master_sig_M = _create_signature_task(malicious_args)

print(f"[INFO] Master signatures built in {time.time()-t0:.2f}s")

# ===================================== EVALUATION (PARALLEL) =====================================

_GLOBALS = {
    "model_B": model_B, "model_M": model_M,
    "master_sig_B": master_sig_B, "master_sig_M": master_sig_M,
    "feature_names": feature_names,
    "T_ben": T_ben, "T_mal": T_mal,
    "test_X": test_X, "test_y": test_y, "test_df": test_df
}

def _classify_row_index(i):
    row_df = _GLOBALS["test_X"].iloc[[i]]
    res = classify_row_by_signature(
        row_df,
        _GLOBALS["model_B"], _GLOBALS["model_M"],
        _GLOBALS["master_sig_B"], _GLOBALS["master_sig_M"],
        _GLOBALS["feature_names"],
        _GLOBALS["T_ben"], _GLOBALS["T_mal"],
        tie_break="malicious", return_details=True
    )
    return {
        "row_index": int(_GLOBALS["test_df"].index[i]),
        "true_label": int(_GLOBALS["test_y"][i]),
        "true_name": "benign" if _GLOBALS["test_y"][i]==0 else "malicious",
        "pred_label": res["label"],
        "Similarity_B": res["Similarity_B"],
        "Similarity_M": res["Similarity_M"]
    }

print(f"\n[INFO] Classifying test rows using signature matching…")
indices = list(range(len(test_X))); results = []
n_cpu = os.cpu_count() or 2; n_proc = max(1, n_cpu // 2)
print(f"[INFO] CPUs={n_cpu}. Using n/2={n_proc} workers for classification.")
use_parallel_eval = ("fork" in mp.get_all_start_methods())

t0 = time.time()
if use_parallel_eval:
    print("[INFO] Multiprocessing with 'fork'.")
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_proc) as pool:
        chunk = 200
        for j in range(0, len(indices), chunk):
            part = indices[j:j+chunk]
            results.extend(pool.map(_classify_row_index, part))
            print(f"    processed {min(j+chunk, len(indices))}/{len(indices)} rows…")
else:
    print("[WARN] 'fork' not available for classification; using single-process fallback.")
    for i, idx in enumerate(indices, 1):
        results.append(_classify_row_index(idx))
        if i % 200 == 0 or i == len(indices):
            print(f"    processed {i}/{len(indices)} rows…")

elapsed = time.time() - t0
print(f"[INFO] Done in {elapsed:.2f}s  ({len(indices)/max(elapsed,1e-9):.1f} rows/sec).")

# ============================ SAVE RESULTS & CONFUSION MATRIX ============================

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_CSV, index=False)
print(f"\n[INFO] Saved row results → {RESULTS_CSV}")
print(results_df.head(5))

labels = ["benign", "malicious"]
cm2 = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
for t, p in zip(results_df["true_name"], results_df["pred_label"]):
    cm2.loc[t, p] += 1
cm2.to_csv(CONFUSION_2x2_CSV)
print(f"\n[INFO] 2×2 confusion matrix:\n{cm2}\nSaved → {CONFUSION_2x2_CSV}")

total = cm2.values.sum(); acc = (np.trace(cm2.values)/total) if total else np.nan
tp_b = cm2.loc["benign","benign"]; fp_b = cm2.loc["malicious","benign"]; fn_b = cm2.loc["benign","malicious"]
tp_m = cm2.loc["malicious","malicious"]; fp_m = cm2.loc["benign","malicious"]; fn_m = cm2.loc["malicious","benign"]
prec_b = tp_b / max(tp_b + fp_b, 1); rec_b = tp_b / max(tp_b + fn_b, 1)
prec_m = tp_m / max(tp_m + fp_m, 1); rec_m = tp_m / max(tp_m + fn_m, 1)

print(f"\n[INFO] Accuracy={acc:.4f} | Benign P/R={prec_b:.3f}/{rec_b:.3f} | Malicious P/R={prec_m:.3f}/{rec_m:.3f}")
# ========================================= END ============================================