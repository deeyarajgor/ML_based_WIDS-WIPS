# The first results were almost perfect, which is too good to be useful.
# This script asks harder questions instead:
#   1. What if the model can't use the deauth-specific features?
#   2. What if the attack is much gentler than the flood in the capture?
#   3. Does training on gentler attacks help?
# Run:  python src/experiments.py      (takes a minute or so)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.tree import DecisionTreeClassifier

import config
from features import (FEATURE_COLUMNS, FEATURE_COLUMNS_BLIND, FEATURE_COLUMNS_STRICT,
                      build_windows, load_frames)

config.OUT_DIR.mkdir(exist_ok=True)
rng = np.random.default_rng(42)       # fixed seed so the random thinning repeats

# three levels of difficulty:
#   all features         - includes the deauth counters (easiest)
#   no deauth counters   - counters removed, but the flood still shows up in
#                          total traffic rate and signal strength
#   deauth frames hidden - every feature ignores deauth frames (hardest)
FEATURE_SETS = {"all features": FEATURE_COLUMNS,
                "no deauth counters": FEATURE_COLUMNS_BLIND,
                "deauth frames hidden": FEATURE_COLUMNS_STRICT}
MODELS = {
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"),
    "Decision Tree": lambda: DecisionTreeClassifier(
        random_state=42, class_weight="balanced"),
}

# "keep" = the fraction of attack frames we keep. 1.0 is the real capture,
# 0.01 is a very gentle attack with 1% of the original frames.
TEST_LEVELS = [1.0, 0.25, 0.1, 0.05, 0.02, 0.01]
MIXED_TRAIN_LEVELS = [1.0, 0.5, 0.1, 0.02]   # strengths shown to the "mixed" models

# ---------- load the real capture ----------
print("Loading frames...")
frames = load_frames(config.DATA_FILE)
is_attack = (frames[config.COL_LABEL] != config.NORMAL_LABEL).to_numpy()

# build the windows once for every strength we need
windows_by_keep = {}
for keep in sorted(set(TEST_LEVELS) | set(MIXED_TRAIN_LEVELS), reverse=True):
    if keep == 1.0:
        w = build_windows(frames)
    else:
        # throw away most attack frames at random, then rebuild the windows.
        # A window that loses every attack frame is correctly labelled normal.
        drop = is_attack & (rng.random(len(frames)) > keep)
        w = build_windows(frames[~drop])
    windows_by_keep[keep] = w.sort_values("window").reset_index(drop=True)

full = windows_by_keep[1.0]
cut = int(len(full) * (1 - config.TEST_FRACTION))
cut_window = full.iloc[cut]["window"]        # test period starts at this second
print(f"{len(frames):,} frames, {len(full)} windows, test starts at t={cut_window}s")

# ---------- training sets: first 70% of the capture only ----------
train_sets = {
    "full-strength only": full[full["window"] < cut_window],
    "mixed strengths": pd.concat(
        [windows_by_keep[k][windows_by_keep[k]["window"] < cut_window]
         for k in MIXED_TRAIN_LEVELS]),
}

# train every combination once
trained = {}
for train_name, train in train_sets.items():
    for fs_name, cols in FEATURE_SETS.items():
        for model_name, make in MODELS.items():
            trained[(train_name, fs_name, model_name)] = make().fit(
                train[cols], train["label"])

# ---------- test on attacks of every strength (last 30% of the capture) ----------
rows = []
matrices = {}      # kept for the confusion-matrix picture

for keep in TEST_LEVELS:
    windows = windows_by_keep[keep]
    test = windows[windows["window"] >= cut_window]
    y = test["label"]

    for (train_name, fs_name, model_name), model in trained.items():
        pred = model.predict(test[FEATURE_SETS[fs_name]])
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "model": model_name, "training": train_name, "features": fs_name,
            "attack_kept": f"{keep * 100:g}%",
            "attack_windows": int(y.sum()),
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })
        if model_name == "Random Forest":
            matrices[(train_name, fs_name, keep)] = cm

results = pd.DataFrame(rows)
results.to_csv(config.OUT_DIR / "experiment_results.csv", index=False)

# ---------- print a readable summary ----------
order = [f"{k * 100:g}%" for k in TEST_LEVELS]
pd.set_option("display.width", 200)
for metric in ["accuracy", "recall", "precision"]:
    table = results.pivot_table(index=["model", "training", "features"],
                                columns="attack_kept", values=metric)[order]
    print(f"\n{metric.upper()} by attack strength (100% = real capture):")
    print(table.round(3).to_string())
print("\n(full table with accuracy, f1 and confusion counts is in experiment_results.csv)")

# ---------- picture 1: confusion matrices, Random Forest, all features ----------
show = [1.0, 0.1, 0.02]
fig, axes = plt.subplots(len(train_sets), len(show), figsize=(11, 7))
for r, train_name in enumerate(train_sets):
    for c, keep in enumerate(show):
        ConfusionMatrixDisplay(matrices[(train_name, "all features", keep)],
                               display_labels=["Normal", "Attack"]).plot(
            ax=axes[r][c], colorbar=False, values_format="d", cmap="Blues")
        axes[r][c].set_title(f"trained on {train_name}\n{keep * 100:g}% of attack frames in test",
                             fontsize=10)
plt.suptitle("Random Forest, all features (test set)")
plt.tight_layout()
plt.savefig(config.OUT_DIR / "experiment_confusion.png", dpi=200)
plt.close()

# ---------- picture 2: detection rate as the attack gets gentler ----------
plt.figure(figsize=(9, 5.5))
styles = {"full-strength only": "--", "mixed strengths": "-"}
colours = {"all features": "tab:green", "no deauth counters": "tab:red",
           "deauth frames hidden": "tab:blue"}
for train_name, style in styles.items():
    for fs_name in FEATURE_SETS:
        part = results[(results["model"] == "Random Forest")
                       & (results["training"] == train_name)
                       & (results["features"] == fs_name)]
        plt.plot(order, part["recall"], style, marker="o", color=colours[fs_name],
                 label=f"{fs_name}, trained on {train_name}")
plt.xlabel("Share of attack frames kept in test (100% = real capture)")
plt.ylabel("Recall (share of attack windows caught)")
plt.title("Random Forest: detection rate vs attack intensity")
plt.ylim(-0.02, 1.05)
plt.grid(alpha=0.3)
plt.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.tight_layout()
plt.savefig(config.OUT_DIR / "recall_vs_intensity.png", dpi=200)
plt.close()

print(f"\nSaved table and two figures to {config.OUT_DIR}")