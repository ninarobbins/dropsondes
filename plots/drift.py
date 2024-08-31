import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime

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


# Plot
f, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

for i in range(len(dropsonde_ds["launch_time"])):
    max_id = np.max(np.where(~np.isnan(dropsonde_ds["lon"].isel(launch_time=i))))

    ax[0].plot(
        dropsonde_ds["lat"].isel(launch_time=i)
        - dropsonde_ds["lat"].isel(launch_time=i).isel(alt=max_id),
        dropsonde_ds["alt"],
        linewidth=1.5,
        c="grey",
        alpha=0.75,
    )

    ax[0].set_xlabel("Drift in Latitude / $\degree$", fontsize=fs)
    ax[0].set_ylabel("Altitude / m", fontsize=fs)
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["top"].set_visible(False)

    ax[1].plot(
        dropsonde_ds["lon"].isel(launch_time=i)
        - dropsonde_ds["lon"].isel(launch_time=i).isel(alt=max_id),
        dropsonde_ds["alt"],
        linewidth=1.5,
        c="grey",
        alpha=0.75,
    )

    ax[1].set_xlabel("Drift in Longitude / $\degree$", fontsize=fs)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["left"].set_visible(False)

# Save
plt.savefig(
    f"/Users/ninarobbins/Desktop/PhD/ORCESTRA/Figures/dropsondes/{flight_index}/drift_dropsondes_PERCUSION_HALO_{flight_id}.png",
    bbox_inches="tight",
)
