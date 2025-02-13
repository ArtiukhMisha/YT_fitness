import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

df = pd.read_pickle("../data/interim/data_resampled.pkl")

# 
# Plot 1 column
# 
set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"])
plt.plot(set_df["acc_y"].reset_index(drop=True))

# plot all exersices
for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"][:100].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

#styling plots
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 6)
mpl.rcParams["figure.dpi"] = 100

# 
# plot different exercises
# 
category_df = (
    df.query('label == "squat"').query('participant == "A"').reset_index(drop=True)
)

fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot(legend=True)
ax.set_xlabel("Time")
ax.set_ylabel("acc_y")

# 
# plot different participants
# 
participant_df = (
    df.query('label == "bench"').sort_values("participant").reset_index(drop=True)
)

fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot(legend=True)
ax.set_xlabel("Time")
ax.set_ylabel("acc_y")


# 
# plot x,y,z acc 
# 
label = "squat"
participant = "A"
all_axis_df = (
    df.query(f'label == "{label}"')
    .query(f'participant == "{participant}"')
    .reset_index(drop=True)
)

fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_xlabel("samples")
ax.set_ylabel("acc_y")
plt.legend()


# 
# plot gyr x,y,z for every label and participant
# 
for label in df["label"].unique():
    for participant in df["participant"].unique():
        all_axis_df = (
            df.query(f'label == "{label}"')
            .query(f'participant == "{participant}"')
            .reset_index(drop=True)
        )
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_xlabel("samples")
            ax.set_ylabel("gyr_y")
            plt.title(f"{label}({participant})".title())
            plt.legend()
        
        
# 
# combine acc and gyr 
# 
label = "row"
participant = "A"
combi_df = (
    df.query(f'label == "{label}"')
    .query(f'participant == "{participant}"')
    .reset_index(drop=True)
)

fig, ax = plt.subplots(nrows=2, sharex=True,figsize=(20, 10))
combi_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
combi_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])
ax[0].legend(loc="upper center",bbox_to_anchor=(0.5, 1.2), ncol=3,fancybox=True, shadow=True)
ax[1].legend(loc="upper center",bbox_to_anchor=(0.5, 1.2), ncol=3,fancybox=True, shadow=True)
ax[1].set_xlabel("samples")



#
# plot and save all exercises
#
for label in df["label"].unique():
    for participant in df["participant"].unique():
        combined_df = (
            df.query(f'label == "{label}"')
            .query(f'participant == "{participant}"')
            .reset_index(drop=True)
        )
        if len(combined_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True,figsize=(20, 10))
            combined_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            combined_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])
            ax[0].legend(loc="upper center",bbox_to_anchor=(0.5, 1.2), ncol=3,fancybox=True, shadow=True)
            ax[1].legend(loc="upper center",bbox_to_anchor=(0.5, 1.2), ncol=3,fancybox=True, shadow=True)
            ax[1].set_xlabel("samples")
            plt.savefig(f"../reports/figures/{label.title()} ({participant}).png")
            plt.show()