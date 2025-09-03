# =============== AGGREGATED ANOMALY SCORE PIPELINE (BATCHED) ===============
# Classifies rows in batches for significant performance improvement.
# The main loop is sequential (batch-by-batch) to avoid parallelism conflicts,
# but operations within each batch are vectorized for speed.
#
# Does:
#   - read, shuffle, stratified train/test
#   - build and fit G_benign / G_malicious
#   - classify rows in batches by comparing aggregated anomaly scores
#   - save row results CSV + 2×2 confusion matrix CSV

import os, time, json
os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from dowhy import gcm
from tqdm import tqdm # Import tqdm for progress bar


def visualize_causal_graph(adjacency_matrix, variable_names, threshold=0.01):
    """
    Visualizes a causal graph based on a given adjacency matrix, with a threshold.

    This function is fixed to align with the LiNGAM model's convention, where edges
    are drawn from the column variable to the row variable for a non-zero entry.

    Args:
        adjacency_matrix (np.ndarray or pd.DataFrame): The adjacency matrix
            representing the causal relationships. A non-zero value at [i, j]
            means a causal effect from variable j to variable i.
        variable_names (list): A list of strings for the variable names.
        threshold (float): A minimum absolute value for a coefficient to be
            considered a valid causal link. Coefficients smaller than this
            will not be included in the graph.
    """
    # Convert the DataFrame to a NumPy array for robust integer indexing
    if isinstance(adjacency_matrix, pd.DataFrame):
        adjacency_matrix = adjacency_matrix.to_numpy()

    num_variables = len(variable_names)
    G = nx.DiGraph()

    # Add nodes to the graph
    G.add_nodes_from(variable_names)

    # Add edges based on the adjacency matrix and the new threshold
    for i in range(num_variables):
        for j in range(num_variables):
            # Check for a non-zero entry that is also above the threshold
            if abs(adjacency_matrix[i, j]) > threshold:
                # Draw the edge from variable 'j' to variable 'i'
                G.add_edge(variable_names[j], variable_names[i], weight=adjacency_matrix[i, j])

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000,
            font_size=8, arrowsize=20)

    # Add edge labels (weights) for clarity
    edge_labels = nx.get_edge_attributes(G, 'weight')
    # Round the weights for better readability
    rounded_labels = {edge: f'{weight:.2f}' for edge, weight in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=rounded_labels, font_color='red', font_size=6)

    #plt.title("Causal Graph Visualization (with Threshold)")
    #plt.show()

    return G

# ============================ ANOMALY SCORE UTILS (BATCHED) ============================

def topk_influenced_targets_from_adj(adj_df: pd.DataFrame, k=5, threshold=0.3):
    numeric = adj_df.astype(float).fillna(0.0)
    w = numeric.where(numeric.abs() > threshold, 0.0).abs()
    in_weight_sum = w.sum(axis=1)
    return list(in_weight_sum.sort_values(ascending=False).head(k).index)

def attribution_dict_to_matrix(attr_dict, variable_names, num_samples):
    """
    Converts DoWhy's attribution dictionary from a batch call into a numpy matrix.
    """
    matrix = np.zeros((num_samples, len(variable_names)))
    var_to_idx = {var: i for i, var in enumerate(variable_names)}

    for var, values in attr_dict.items():
        if var in var_to_idx:
            matrix[:, var_to_idx[var]] = values.flatten()

    return matrix

def classify_batch_by_anomaly_score(
    batch_df, model_B, model_M,
    variable_names,
    T_ben, T_mal,
    tie_break="malicious"
):
    """
    Classifies a whole batch of samples by aggregating anomaly attribution magnitudes.
    """
    num_samples = len(batch_df)

    # 1. Calculate total anomaly scores using the benign model for the whole batch
    total_scores_B = np.zeros(num_samples)
    for t in T_ben:
        attr = gcm.attribute_anomalies(model_B, t, anomaly_samples=batch_df)
        matrix = attribution_dict_to_matrix(attr, variable_names, num_samples)
        # Calculate L2 norm for each row in the batch and add to total
        total_scores_B += np.linalg.norm(matrix, axis=1)

    # 2. Calculate total anomaly scores using the malicious model for the whole batch
    total_scores_M = np.zeros(num_samples)
    for t in T_mal:
        attr = gcm.attribute_anomalies(model_M, t, anomaly_samples=batch_df)
        matrix = attribution_dict_to_matrix(attr, variable_names, num_samples)
        total_scores_M += np.linalg.norm(matrix, axis=1)

    # 3. Vectorized classification for the entire batch
    # Default to tie_break, then apply conditions.
    labels = np.full(num_samples, tie_break, dtype=object)
    labels = np.where(total_scores_B < total_scores_M, "benign", labels)
    labels = np.where(total_scores_M < total_scores_B, "malicious", labels)

    return total_scores_B, total_scores_M, labels



# ===================================== DATA & SPLIT =====================================
SEED = 42
TRAIN_FRAC = 0.8
BATCH_SIZE = 256 # Define batch size for processing
RESULTS_CSV = "./row_results_anomaly_score_batched.csv"
CONFUSION_2x2_CSV = "./confusion_matrix_2x2_anomaly_score_batched.csv"

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
G_benign = visualize_causal_graph(adj_ben, feature_names, 1.0)
G_malicious = visualize_causal_graph(adj_mal, feature_names, 1.0)
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


# ===================================== EVALUATION (BATCHED) =====================================
print(f"\n[INFO] Classifying test rows using batched anomaly scores (batch size: {BATCH_SIZE})…")
results = []
t0 = time.time()

# --- MODIFIED PART: BATCH PROCESSING LOOP ---
for i in tqdm(range(0, len(test_X), BATCH_SIZE), desc="Classifying batches"):
    batch_end = min(i + BATCH_SIZE, len(test_X))
    batch_df = test_X.iloc[i:batch_end]

    # Get scores and labels for the entire batch at once
    scores_B, scores_M, labels = classify_batch_by_anomaly_score(
        batch_df, model_B, model_M,
        feature_names, T_ben, T_mal
    )

    # Unpack the batch results into the final list
    true_labels_batch = test_y[i:batch_end]
    for j in range(len(batch_df)):
        original_index = test_df.index[i + j]
        results.append({
            "row_index": int(original_index),
            "true_label": int(true_labels_batch[j]),
            "true_name": "benign" if true_labels_batch[j] == 0 else "malicious",
            "pred_label": labels[j],
            "TotalScore_B": scores_B[j],
            "TotalScore_M": scores_M[j]
        })
# --- END MODIFIED PART ---

elapsed = time.time() - t0
print(f"[INFO] Done in {elapsed:.2f}s  ({len(test_X)/max(elapsed,1e-9):.1f} rows/sec).")


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

