# Replays the test windows second by second, like a live feed, and asks the
# trained model "attack or normal?" for each one. It raises an alert when an
# attack starts and says when it ends.
# Run:  python src/detect.py

import argparse
import time
from datetime import datetime

import joblib
import pandas as pd

import config
from features import FEATURE_COLUMNS_STRICT

parser = argparse.ArgumentParser()
parser.add_argument("--delay", type=float, default=0.05,
                    help="seconds to pause per window (makes the demo watchable)")
args = parser.parse_args()

model = joblib.load(config.MODEL_FILE)
windows = pd.read_csv(config.OUT_DIR / "test_windows.csv")   # saved by train.py
windows = windows.sort_values("window").reset_index(drop=True)

# ask the model about every window up front (faster than one at a time)
windows["flagged"] = model.predict(windows[FEATURE_COLUMNS_STRICT])

alert_log = config.OUT_DIR / "alerts.log"
CALM_NEEDED = 3        # clear windows in a row before we call the attack over
under_attack = False
calm_count = 0
attack_start = 0
n_alerts = 0


def log(text):
    """Print a line and also append it to the alert log."""
    print(text)
    with open(alert_log, "a") as f:
        f.write(text + "\n")


print("WIDS running...\n")

for _, row in windows.iterrows():
    time.sleep(args.delay)
    second = int(row["window"])
    stamp = datetime.now().strftime("%H:%M:%S")

    if row["flagged"]:
        calm_count = 0
        if not under_attack:
            # first flagged window after quiet traffic = a new attack
            under_attack = True
            attack_start = second
            n_alerts += 1
            log(f"[{stamp}] ALERT: deauth flood detected at t={second}s "
                f"({row['deauth_count']:.0f} deauth frames in 1s)")
    else:
        if under_attack:
            calm_count += 1
            if calm_count >= CALM_NEEDED:
                under_attack = False
                log(f"[{stamp}] Attack over. Lasted about {second - CALM_NEEDED - attack_start + 1}s "
                    f"(t={attack_start}s to t={second - CALM_NEEDED}s)")
        elif second % 20 == 0:
            print(f"[{stamp}] t={second}s  traffic normal")

print(f"\nDone. {n_alerts} attack(s) detected. Log: {alert_log}")