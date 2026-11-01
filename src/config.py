# All the settings live here so the other files stay clean.

from pathlib import Path

# project folders 
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "Deauth.csv"     
MODEL_FILE = ROOT / "models" / "wids_model.joblib"
OUT_DIR = ROOT / "outputs"

#column names in the csv (these match the Mendeley deauth file)
COL_TIME = "frame.time_epoch"          # capture time in whole seconds
COL_SUBTYPE = "wlan.fc.type_subtype"   # frame type, e.g. 12 = deauthentication
COL_RSSI = "radiotap.dbm_antsignal"    # signal strength in dBm
COL_LABEL = "Label"                    # what the dataset says the frame is
NORMAL_LABEL = "Normal"                # label text used for harmless traffic

# detection settings
WINDOW_SECONDS = 1     # how long each time window is
DEAUTH_SUBTYPE = 12    # 802.11 subtype number for deauthentication frames
TEST_FRACTION = 0.3    # last 30% of the capture is kept aside for testing