import diag_vae.constants as const
import pandas as pd
from datetime import datetime


raw_file_paths = [const.SWAT_RAW_NORMAL_V1_PATH, const.SWAT_RAW_ATTACK_V0_PATH]
df_normal_v1, df_attack_v0 = [
    pd.read_excel(path, header=1) for path in raw_file_paths
]

proc_file_paths = [const.SWAT_PARQUET_NORMAL_V1_PATH, const.SWAT_PARQUET_ATTACK_V0_PATH]
df_normal_v1, df_attack_v0 = [pd.read_parquet(path) for path in proc_file_paths]

for df in [df_normal_v1, df_attack_v0]:
    df["Timestamp"] = pd.to_datetime(df[" Timestamp"])

# convert to pandas time series
df_normal_v1 = df_normal_v1.set_index("Timestamp", drop=True)
df_attack_v0 = df_attack_v0.set_index("Timestamp", drop=True)
df_normal_v1 = df_normal_v1.drop(" Timestamp", axis=1)
df_attack_v0 = df_attack_v0.drop(" Timestamp", axis=1)


# fix column names (some begin with white spaces)
df_normal_v1.columns = [s.replace(" ", "") for s in df_normal_v1.columns]
df_attack_v0.columns = [s.replace(" ", "") for s in df_attack_v0.columns]


# read raw labels files
df_label = pd.read_excel(const.SWAT_RAW_LABEL_PATH)

# filter labels df to the attack that have a end date attached
# transofrm end time to full timestmap
df_label_time = df_label[df_label['End Time'].notna()].copy()
df_label_time.loc[:, 'End Time'] = [datetime.combine(datetime.date(a), b) for a,b in zip(
    df_label_time['Start Time'], df_label_time['End Time'])]
df_label_time = df_label_time.reset_index(drop=True)

# ok, lets remove everything smaller than min_date and larger than max datefrom
# the attacks and labels
# See EDA notebook for why we do so!

df_label_time = df_label_time[
    (df_label_time["Start Time"] > const.SWAT_MIN_DATE)
    & (df_label_time["Start Time"] < const.SWAT_MAX_DATE)
 ]
df_attack_v0 = df_attack_v0[
    (df_attack_v0.index > const.SWAT_MIN_DATE)
    & (df_attack_v0.index < const.SWAT_MAX_DATE)
]


df_normal_v1.to_parquet(const.SWAT_TRAIN_PATH)
df_attack_v0.to_parquet(const.SWAT_VAL_PATH)
df_label_time.to_csv(const.SWAT_LABEL_PATH)
