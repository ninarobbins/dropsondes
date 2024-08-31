import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime

import sys

sys.path.append("./")
sys.path.append("../")
from droputils.physics_utils import add_theta, add_theta_v

fs = 14

flight_time = datetime(2024, 8, 11, 0, 0, 0)
flight_index = f"HALO-{flight_time.strftime('%Y%m%d')}a"
flight_id = flight_index[5:9] + flight_index[9:11] + flight_index[11:13]
flight_date = flight_index[5:9] + "-" + flight_index[9:11] + "-" + flight_index[11:13]

# Dropsonde data

level_3_path = "/Users/ninarobbins/Desktop/PhD/Dropsondes/data/Level_3/Level_3.nc"
dropsonde_ds = (
    xr.open_dataset(level_3_path)
    .rename({"launch_time_(UTC)": "launch_time"})
    .rename({"gpsalt": "alt"})
    .swap_dims({"sonde_id": "launch_time"})
)

launch_time_strings = dropsonde_ds.coords["launch_time"].values
launch_time_datetimes = np.array([np.datetime64(date) for date in launch_time_strings])
dropsonde_ds = dropsonde_ds.assign_coords(
    launch_time=("launch_time", launch_time_datetimes)
)

dropsonde_ds = dropsonde_ds.sel(launch_time=flight_date)

# Add thetas
dropsonde_ds = add_theta(dropsonde_ds)
dropsonde_ds = add_theta_v(dropsonde_ds)

# Plot

row = 1
col = 5

fig, ax = plt.subplots(row, col, sharey=True, figsize=(17, 6))


r = ["ta", "theta", "theta_v", "q", "rh"]
r_titles = [
    "T / $\degree$C",
    "$\\theta$ / K",
    "$\\theta_v$ / K",
    "q / kg kg${-1}$",
    "RH / %",
]

print(f"Plotting vertical profiles of {r}...")

for j in range(col):
    d = dropsonde_ds[r[j]]
    for i in range(1, len(dropsonde_ds["launch_time"]) - 1):
        ax[j].plot(
            d.isel(launch_time=i),
            dropsonde_ds["alt"] / 1000,
            c="grey",
            alpha=0.25,
            linewidth=0.5,
        )

    ax[j].plot(
        np.nanmean(d, axis=0),
        dropsonde_ds["alt"] / 1000,
        linewidth=3,
        c="k",
    )
    ax[j].set_xlabel(r_titles[j], fontsize=fs)
    ax[j].spines["right"].set_visible(False)
    ax[j].spines["top"].set_visible(False)
    if j == 0:
        ax[j].set_ylabel("Altitude (km)", fontsize=fs)


# Save
plt.savefig(
    f"/Users/ninarobbins/Desktop/PhD/ORCESTRA/Figures/dropsondes/{flight_index}/all_profiles_dropsondes_PERCUSION_HALO_{flight_id}.png",
    bbox_inches="tight",
)
