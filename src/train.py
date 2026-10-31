# train.py
# Builds the windows, trains the Random Forest, and reports how well it did.
# Uses the strict feature set: every feature ignores deauth frames, so the model
# has to spot the attack from side effects on the other traffic (no label leak).
# Run:  python src/train.py

import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)

import config
from features import FEATURE_COLUMNS_STRICT, build_windows, load_frames

config.OUT_DIR.mkdir(exist_ok=True)
config.MODEL_FILE.parent.mkdir(exist_ok=True)

# 1. load the frames and turn them into windows
print("Loading frames...")
frames = load_frames(config.DATA_FILE)
windows = build_windows(frames).sort_values("window").reset_index(drop=True)
print(f"{len(frames):,} frames -> {len(windows):,} windows")
print(windows["label"].value_counts().rename({0: "normal", 1: "attack"}))

# 2. split by time: the first 70% of the capture is for learning, the last
# 30% is kept hidden for testing. We don't shuffle, because windows from
# neighbouring seconds look almost identical, and a random split would let
# the model test on near-copies of what it trained on.
cut = int(len(windows) * (1 - config.TEST_FRACTION))
train = windows.iloc[:cut]
test = windows.iloc[cut:]
print(f"\nTrain: {len(train)} windows ({train['label'].sum()} attack)")
print(f"Test:  {len(test)} windows ({test['label'].sum()} attack)")

# 3. train the model
model = RandomForestClassifier(n_estimators=100, random_state=42,
                               class_weight="balanced")  # keeps the rarer class from being ignored
model.fit(train[FEATURE_COLUMNS_STRICT], train["label"])

# 4. test it on windows it has never seen
predictions = model.predict(test[FEATURE_COLUMNS_STRICT])
truth = test["label"]

print(f"\nAccuracy: {accuracy_score(truth, predictions):.4f}")
print(classification_report(truth, predictions,
                            target_names=["normal", "attack"], digits=3))

cm = confusion_matrix(truth, predictions, labels=[0, 1])
print("Confusion matrix (rows = real, columns = predicted):")
print(cm)

# 5. save the confusion matrix picture for the poster
disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Deauth attack"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion matrix (test set)")
plt.tight_layout()
plt.savefig(config.OUT_DIR / "confusion_matrix.png", dpi=200)
plt.close()

# which features did the forest rely on most?
print("\nFeature importance:")
importances = sorted(zip(FEATURE_COLUMNS_STRICT, model.feature_importances_),
                     key=lambda pair: pair[1], reverse=True)
for name, score in importances:
    print(f"  {name:18s} {score:.3f}")

# 6. save the model, and the test windows so detect.py can replay them
joblib.dump(model, config.MODEL_FILE)
test.to_csv(config.OUT_DIR / "test_windows.csv", index=False)
print(f"\nSaved model to {config.MODEL_FILE}")
print(f"Saved confusion matrix to {config.OUT_DIR / 'confusion_matrix.png'}")