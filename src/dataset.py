import pandas as pd
from glob import glob

files = glob("../data/raw/MetaMotion/*.csv")


def read_data_from_files(files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()
    data_path = "../data/raw/MetaMotion\\"
    acc_set = 1
    gyr_set = 1

    for f in files:
        
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)
        
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])
    # --------------------------------------------------------------
    # Working with datetimes
    # --------------------------------------------------------------
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")
    acc_df.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1, inplace=True)
    gyr_df.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1, inplace=True)
    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)
# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------
df_merged = pd.concat(
    [acc_df.iloc[:, :3], gyr_df],axis=1
)
df_merged.info()
df_merged.set = df_merged.set.astype("Int64")

df_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]
# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------
df_merged.info()
# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz
df_merged.resample("100ms").apply(sampling)
days = [g for n, g in df_merged.groupby(pd.Grouper(freq="D"))]
days[0]
data_resampled = pd.concat(df.resample("200ms").apply(sampling).dropna() for df in days)
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
data_resampled[data_resampled.set == 1].head()
data_resampled.info()
data_resampled.to_pickle("../data/interim/data_resampled.pkl")
