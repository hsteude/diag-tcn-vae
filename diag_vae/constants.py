import os
from datetime import datetime

DATA_DIR_PATH = "./data"
RAW_DATA_DIR_PATH = os.path.join(DATA_DIR_PATH, "raw")
PROC_DATA_DIR_PATH = os.path.join(DATA_DIR_PATH, "processed")

# SWaT
SWAT_RAW_FOLDER_PATH = os.path.join(RAW_DATA_DIR_PATH, "SWaT.A1 _ A2_Dec 2015")
SWAT_RAW_LABEL_PATH = os.path.join(SWAT_RAW_FOLDER_PATH, "List_of_attacks_Final.xlsx")
SWAT_RAW_NORMAL_V0_PATH = os.path.join(
    SWAT_RAW_FOLDER_PATH, "Physical", "SWaT_Dataset_Normal_v0.xlsx"
)
SWAT_RAW_NORMAL_V1_PATH = os.path.join(
    SWAT_RAW_FOLDER_PATH, "Physical", "SWaT_Dataset_Normal_v1.xlsx"
)
SWAT_RAW_ATTACK_V0_PATH = os.path.join(
    SWAT_RAW_FOLDER_PATH, "Physical", "SWaT_Dataset_Attack_v0.xlsx"
)

SWAT_PARQUET_NORMAL_V0_PATH = os.path.join(
    PROC_DATA_DIR_PATH, "swat_dataset_normal_v0.parquet"
)
SWAT_PARQUET_NORMAL_V1_PATH = os.path.join(
    PROC_DATA_DIR_PATH, "swat_dataset_normal_v1.parquet"
)
SWAT_PARQUET_ATTACK_V0_PATH = os.path.join(
    PROC_DATA_DIR_PATH, "swat_dataset_attack_v0.parquet"
)

SWAT_TRAIN_PATH = os.path.join(PROC_DATA_DIR_PATH, "swat_train.parquet")
SWAT_VAL_PATH = os.path.join(PROC_DATA_DIR_PATH, "swat_val.parquet")
SWAT_LABEL_PATH = os.path.join(PROC_DATA_DIR_PATH, "labels.csv")
SWAT_MIN_DATE = datetime(2015, 12, 22)
SWAT_MAX_DATE = datetime(2016, 1, 2)


SWAT_SENSOR_COLS = [
    "FIT101",
    "LIT101",
    "MV101",
    "P101",
    "P102",
    "AIT201",
    "AIT202",
    "AIT203",
    "FIT201",
    "MV201",
    "P201",
    "P202",
    "P203",
    "P204",
    "P205",
    "P206",
    "DPIT301",
    "FIT301",
    "LIT301",
    "MV301",
    "MV302",
    "MV303",
    "MV304",
    "P301",
    "P302",
    "AIT401",
    "AIT402",
    "FIT401",
    "LIT401",
    "P401",
    "P402",
    "P403",
    "P404",
    "UV401",
    "AIT501",
    "AIT502",
    "AIT503",
    "AIT504",
    "FIT501",
    "FIT502",
    "FIT503",
    "FIT504",
    "P501",
    "P502",
    "PIT501",
    "PIT502",
    "PIT503",
    "FIT601",
    "P601",
    "P602",
    "P603",
]

SWAT_SYMBOLS_MAP = {
    "Comp_1": ["FIT101", "LIT101", "MV101", "P101", "P102"],
    "Comp_2": [
        "AIT201",
        "AIT202",
        "AIT203",
        "FIT201",
        "MV201",
        "P201",
        "P202",
        "P203",
        "P204",
        "P205",
        "P206",
    ],
    "Comp_3": [
        "DPIT301",
        "FIT301",
        "LIT301",
        "MV301",
        "MV302",
        "MV303",
        "MV304",
        "P301",
        "P302",
    ],
    "Comp_4": [
        "AIT401",
        "AIT402",
        "FIT401",
        "LIT401",
        "P401",
        "P402",
        "P403",
        "P404",
        "UV401",
    ],
    "Comp_5": [
        "AIT501",
        "AIT502",
        "AIT503",
        "AIT504",
        "FIT501",
        "FIT502",
        "FIT503",
        "FIT504",
        "P501",
        "P502",
        "PIT501",
        "PIT502",
        "PIT503",
    ],
    "Comp_6": ["FIT601", "P601", "P602", "P603"],
}
