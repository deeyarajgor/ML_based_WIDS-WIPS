# features.py
# Turns a list of individual Wi-Fi frames into one row per time window.
# A single frame tells us almost nothing. But "300 deauth frames arrived in
# one second" tells us a lot - that's what the model learns from.
#
# This dataset has no source-MAC column, so each window describes all the
# traffic on the network during that second, not one device.

import pandas as pd

import config


def to_number(series):
    """Turn a column into numbers. Normally pandas has already done this,
    but real captures sometimes hold hex ("0x000c") or several values in
    one cell ("-45,-46"), so there's a slower fallback for those."""
    fast = pd.to_numeric(series, errors="coerce")
    if fast.notna().sum() >= series.notna().sum():
        return fast

    def convert(value):
        text = str(value).split(",")[0].strip()
        try:
            return float(int(text, 0))
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return float("nan")
    return series.map(convert)


def load_frames(path, nrows=None):
    """Read only the columns we need - the file has 154 and we use 4."""
    cols = [config.COL_TIME, config.COL_SUBTYPE, config.COL_RSSI, config.COL_LABEL]
    frames = pd.read_csv(path, usecols=cols, nrows=nrows, low_memory=False)

    frames[config.COL_TIME] = to_number(frames[config.COL_TIME])
    frames[config.COL_SUBTYPE] = to_number(frames[config.COL_SUBTYPE])
    frames[config.COL_RSSI] = to_number(frames[config.COL_RSSI])

    # a frame with no timestamp can't be placed in a window
    return frames.dropna(subset=[config.COL_TIME])


def build_windows(frames):
    """Group frames into 1-second windows and count things in each."""
    df = frames.copy()

    # which window does each frame fall in? (0 = first second, 1 = next, ...)
    t0 = df[config.COL_TIME].min()
    df["window"] = ((df[config.COL_TIME] - t0) // config.WINDOW_SECONDS).astype(int)

    # helper columns that are easy to sum up later
    df["is_deauth"] = (df[config.COL_SUBTYPE] == config.DEAUTH_SUBTYPE).astype(int)
    df["is_mgmt"] = (df[config.COL_SUBTYPE] < 16).astype(int)   # subtypes 0-15 are management frames
    df["is_attack"] = (df[config.COL_LABEL] != config.NORMAL_LABEL).astype(int)

    # same counts again but pretending deauth frames don't exist (used by experiments.py)
    df["is_mgmt_nd"] = ((df[config.COL_SUBTYPE] < 16) & (df["is_deauth"] == 0)).astype(int)
    df["subtype_nd"] = df[config.COL_SUBTYPE].where(df["is_deauth"] == 0)

    feats = df.groupby("window").agg(
        total_frames=("is_deauth", "size"),             # frames seen in this window
        deauth_count=("is_deauth", "sum"),              # how many were deauth
        mgmt_count=("is_mgmt", "sum"),                  # how many were management frames
        rssi_mean=(config.COL_RSSI, "mean"),            # average signal strength
        rssi_std=(config.COL_RSSI, "std"),              # how much the signal varies
        subtype_variety=(config.COL_SUBTYPE, "nunique"),# number of different frame types
        mgmt_count_nd=("is_mgmt_nd", "sum"),            # management frames, deauth not counted
        subtype_variety_nd=("subtype_nd", "nunique"),   # frame types, deauth not counted
        attack_frames=("is_attack", "sum"),             # only used to make the label
    ).reset_index()

    feats["frames_per_sec"] = feats["total_frames"] / config.WINDOW_SECONDS
    feats["deauth_ratio"] = feats["deauth_count"] / feats["total_frames"]

    # fill gaps: a window with one frame has no std, some frames have no signal reading
    feats["rssi_std"] = feats["rssi_std"].fillna(0)
    feats["rssi_mean"] = feats["rssi_mean"].fillna(feats["rssi_mean"].median())

    # label the window: malicious if it contains any attack frame at all
    feats["label"] = (feats["attack_frames"] > 0).astype(int)
    return feats.drop(columns=["attack_frames", "total_frames"])


# the columns the model is allowed to look at.
# No timestamp or window number in here on purpose. If the model saw the
# time, it could "cheat" by learning when in the file the attack happened.
FEATURE_COLUMNS = ["frames_per_sec", "deauth_count", "deauth_ratio",
                   "mgmt_count", "rssi_mean", "rssi_std", "subtype_variety"]

# the same idea but with everything that directly counts deauth frames removed.
# frames_per_sec still includes them, because a flood really does raise the
# overall traffic rate - that's a fair thing for a real IDS to notice.
FEATURE_COLUMNS_BLIND = ["frames_per_sec", "mgmt_count_nd", "rssi_mean",
                         "rssi_std", "subtype_variety_nd"]