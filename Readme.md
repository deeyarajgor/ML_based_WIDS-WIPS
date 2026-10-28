# Basic ML-Based Wireless Intrusion Detection System (WIDS)

A small prototype that uses a Random Forest (scikit-learn) to label one-second
windows of Wi-Fi traffic as normal or a deauthentication flood, and raises an
alert when an attack starts.

## Dataset

Deauthentication CSV from the WPA3 Attacks Dataset (Mendeley Data, CC BY 4.0):
Halbouni, A., Ong, L.-Y., Leow, M.-C., DOI 10.17632/cxx5t5nw7z.1

The data is not included in this repository. Download the Deauthentication CSV
yourself and put it in the `data/` folder.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then open `src/config.py` and set `DATA_FILE` to the name of your CSV.

## Run

```
python src/train.py        # trains the model, prints accuracy and confusion matrix
python src/detect.py       # replays held-out traffic and prints alerts
python src/experiments.py  # harder tests: weaker attacks, no deauth features
```

Results (model, figures, alert log) are written to `models/` and `outputs/`.

## How it works

1. `features.py` groups frames into 1-second windows and counts things in each
   (frames per second, deauth frames, management frames, signal strength).
2. `train.py` trains a Random Forest on the first 70% of the capture and tests
   it on the last 30%.
3. `detect.py` replays the test windows in order and alerts on attacks.
4. `experiments.py` checks how results change when the attack is weaker or the
   deauth-specific features are removed.

## Limits

This dataset has no source-MAC column, so windows describe the whole network
and there is no prevention (blocking) step. Deauth frames never appear in the
normal traffic, so accuracy on the plain test is near perfect; see the
experiments for harder cases.