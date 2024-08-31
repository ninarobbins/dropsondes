import sys

sys.path.append("./")
sys.path.append("../")
from droputils.physics_utils import add_iwv

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

fs = 14

flight_time = datetime(2024, 8, 11, 0, 0, 0)
flight_id = flight_time.strftime("%Y%m%d")
flight_index = f"HALO-{flight_time.strftime('%Y%m%d')}a"
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

# Compute IWV
dropsonde_ds = add_iwv(dropsonde_ds)

# Plot spatiotemporal evolution
fig, ax = plt.subplots(figsize=(15, 5))

# Plotting
scatter = ax.scatter(
    dropsonde_ds["launch_time"].values,
    dropsonde_ds["lat"].isel(alt=-700).values,
    s=90,
    c=dropsonde_ds["iwv"].values,
    edgecolor="grey",
    cmap="Blues_r",
    vmin=45,
    vmax=70,
)

# Set x-axis limits
ax.set_xlim(
    np.min(dropsonde_ds["launch_time"].values) - np.timedelta64(4, "m"),
    np.max(dropsonde_ds["launch_time"].values) + np.timedelta64(4, "m"),
)

# Hide the top and right spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("IWV / kg m$^{-2}$", fontsize=fs - 2)

# Format the x-axis with time
myFmt = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(myFmt)

# Set labels
ax.set_xlabel("Time / UTC", fontsize=fs)
ax.set_ylabel("Latitude / $\degree$N", fontsize=fs)


# Save
plt.savefig(
    f"/Users/ninarobbins/Desktop/PhD/ORCESTRA/Figures/dropsondes/HALO-{flight_id}a/spatiotemporal_IWV_dropsondes_PERCUSION_HALO_{flight_id}.png",
    bbox_inches="tight",
)
