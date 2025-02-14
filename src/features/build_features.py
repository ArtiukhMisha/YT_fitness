import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporaryAbstraction import NumericalAbstraction


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle('../../data/interim/02_outliers_removed_chauvenet.pkl')


predictor_columns = list(df.columns[:6])

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['lines.linewidth'] = 2
# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
# df.info()
# subset = df[df['set'] == 35]['gyr_y'].plot()
for col in predictor_columns:
    df[col] = df[col].interpolate()
df.info()
# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
df[df['set'] == 25]['acc_y'].plot()

duration = df[df['set'] == 25].index[-1] - df[df['set'] == 25].index[0] 

for s in df['set'].unique():
    start = df[df['set'] == s].index[0]
    stop = df[df['set'] == s].index[-1]
    duration = stop - start
    df.loc[df['set'] == s, 'duration'] = duration.seconds   
    
duration_df = df.groupby('category')['duration'].mean()
duration_df.iloc[0]/5
duration_df.iloc[1]/10
# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
df_lowpass = df.copy()
lowpass = LowPassFilter()

fs = 1000/200
cutoff = 1.3

df_lowpass = lowpass.low_pass_filter(df_lowpass,'acc_y', fs, cutoff)

subset = df_lowpass[df_lowpass['set'] == 35]
print(subset['label'][0])

# fig, ax = plt.subplots(nrows=2,sharex=True,figsize=(20,10))
# ax[0].plot(subset['acc_y'].reset_index(drop=True), label='raw data')
# ax[1].plot(subset['acc_y_lowpass'].reset_index(drop=True), label='low pass filter')
# ax[0].legend(loc='upper center',bbox_to_anchor=(0.5, 1.15),fancybox=True, shadow=True, ncol=5)
# ax[1].legend(loc='upper center',bbox_to_anchor=(0.5, 1.15),fancybox=True, shadow=True, ncol=5)

for col in predictor_columns:
    df_lowpass = lowpass.low_pass_filter(df_lowpass, col, fs, cutoff)
    df_lowpass[col] = df_lowpass[col+'_lowpass']
    del df_lowpass[col+'_lowpass']

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10,10))
plt.plot(range(1,len(predictor_columns)+1),pc_values)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 4)

subset = df_pca[df_pca['set'] == 35]
subset[["pca_1",'pca_2','pca_3','pca_4']].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared['acc_x']**2 + df_squared['acc_y']**2 + df_squared['acc_z']**2
gyr_r = df_squared['gyr_x']**2 + df_squared['gyr_y']**2 + df_squared['gyr_z']**2

df_squared['acc_r'] = np.sqrt(acc_r)
df_squared['gyr_r'] = np.sqrt(gyr_r)

subset = df_squared[df_squared['set'] == 35]
subset[['acc_r','gyr_r']].plot(subplots=True)
# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------