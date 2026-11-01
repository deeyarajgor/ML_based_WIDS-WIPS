# Turns the big capture (Deauth.csv) into the small one-row-per-second table
# that the model actually uses, and saves it to data/windows.csv.
# That small file (a few hundred KB) is what goes on GitHub, so anyone can
# re-run the results without having the original dataset.
# Run once:  python src/export_windows.py

import config
from features import build_windows, load_frames

print("Loading frames...")
frames = load_frames(config.DATA_FILE)
windows = build_windows(frames).sort_values("window").reset_index(drop=True)

out = config.ROOT / "data" / "windows.csv"
windows.to_csv(out, index=False)
print(f"{len(frames):,} frames -> {len(windows)} windows, saved to {out}")