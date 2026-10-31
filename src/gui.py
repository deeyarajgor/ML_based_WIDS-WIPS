# gui.py
# Shows test results for the same Random Forest trained three ways, from
# easiest to hardest, so you can see how much the deauth counters were
# giving away. Nothing else.
# Run:  python src/gui.py      (takes a few seconds to start: it rebuilds the windows)

import tkinter as tk

import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import config
from features import (FEATURE_COLUMNS, FEATURE_COLUMNS_BLIND, FEATURE_COLUMNS_STRICT,
                      build_windows, load_frames)

SETS = [
    ("All features", "includes the deauth counters", FEATURE_COLUMNS),
    ("No deauth counters", "flood only shows in traffic rate and signal", FEATURE_COLUMNS_BLIND),
    ("Deauth frames hidden", "hardest: only side effects on other traffic", FEATURE_COLUMNS_STRICT),
]

# build windows, split by time (first 70% train, last 30% test)
print("Loading capture...")
windows = build_windows(load_frames(config.DATA_FILE))
windows = windows.sort_values("window").reset_index(drop=True)
cut = int(len(windows) * (1 - config.TEST_FRACTION))
train, test = windows.iloc[:cut], windows.iloc[cut:]
truth = test["label"]

results = []
for name, note, cols in SETS:
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(train[cols], train["label"])
    pred = model.predict(test[cols])
    scores = [("Accuracy", accuracy_score(truth, pred)),
              ("Precision", precision_score(truth, pred, zero_division=0)),
              ("Recall", recall_score(truth, pred, zero_division=0)),
              ("F1", f1_score(truth, pred, zero_division=0))]
    results.append((name, note, scores, confusion_matrix(truth, pred, labels=[0, 1])))

root = tk.Tk()
root.title("Wi-Fi attack detector - results")
root.geometry("1120x560")
root.configure(bg="white")

tk.Label(root, text="Deauthentication attack detector", font=("Helvetica", 18, "bold"),
         bg="white", fg="#1F2937").pack(pady=(16, 0))
tk.Label(root, text=f"Test results on {len(test)} one-second windows "
                    f"({int(truth.sum())} attack, {int((truth == 0).sum())} normal)",
         font=("Helvetica", 11), bg="white", fg="#6B7280").pack()

row = tk.Frame(root, bg="white")
row.pack(fill="both", expand=True, padx=12, pady=12)

for name, note, scores, cm in results:
    col = tk.Frame(row, bg="#F9FAFB", padx=10, pady=10)
    col.pack(side="left", fill="both", expand=True, padx=6)
    tk.Label(col, text=name, font=("Helvetica", 14, "bold"), bg="#F9FAFB", fg="#1F2937").pack()
    tk.Label(col, text=note, font=("Helvetica", 9), bg="#F9FAFB", fg="#6B7280").pack()

    nums = tk.Frame(col, bg="#F9FAFB")
    nums.pack(pady=8)
    for label, value in scores:
        cell = tk.Frame(nums, bg="#F9FAFB")
        cell.pack(side="left", padx=7)
        tk.Label(cell, text=f"{value:.1%}", font=("Helvetica", 13, "bold"),
                 bg="#F9FAFB", fg="#1F2937").pack()
        tk.Label(cell, text=label, font=("Helvetica", 9), bg="#F9FAFB", fg="#6B7280").pack()

    fig = Figure(figsize=(3.2, 3), facecolor="#F9FAFB")
    ax = fig.add_subplot(111)
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"]).plot(
        ax=ax, colorbar=False, values_format="d", cmap="Blues")
    fig.tight_layout()
    FigureCanvasTkAgg(fig, master=col).get_tk_widget().pack(fill="both", expand=True)

root.mainloop()