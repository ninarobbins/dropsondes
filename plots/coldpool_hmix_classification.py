import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


import sys

sys.path.append("./")
sys.path.append("../")
from droputils.physics_utils import (
    add_mr,
    add_theta,
    add_theta_v,
    add_density,
    find_ml_height_from_gradient,
)

fs = 14

# Dropsonde data
level_3_path = "/Volumes/Upload/HALO/Dropsonde/dropsondes/Level_3/PERCUSION_Level_3.nc"
dropsonde_ds = (
    xr.open_dataset(level_3_path)
    .rename({"launch_time_(UTC)": "launch_time"})
    .swap_dims({"sonde_id": "launch_time"})
)

launch_time_strings = dropsonde_ds.coords["launch_time"].values
launch_time_datetimes = np.array([np.datetime64(date) for date in launch_time_strings])
dropsonde_ds = dropsonde_ds.assign_coords(
    launch_time=("launch_time", launch_time_datetimes)
)

# Add variables
dropsonde_ds = add_mr(dropsonde_ds)
dropsonde_ds = add_theta(dropsonde_ds)
dropsonde_ds = add_theta_v(dropsonde_ds)
dropsonde_ds = add_density(dropsonde_ds)

# Compute mixed layer height
low_lim = 100
high_lim = 5000
dropsonde_ds = find_ml_height_from_gradient(
    dropsonde_ds.sel(alt=slice(low_lim, high_lim)),
    var="theta_v",
    threshold=0.2,
    lower_lim_m=low_lim,
)

# Classify into cold pool / environment / none
cp_soundings = dropsonde_ds.where(dropsonde_ds.hmix_grad_theta_v < 400, drop=True)
env_soundings = dropsonde_ds.where(dropsonde_ds.hmix_grad_theta_v > 500, drop=True)
none_soundings = dropsonde_ds.where(
    dropsonde_ds.hmix_grad_theta_v > 400, drop=True
).where(dropsonde_ds.hmix_grad_theta_v < 500, drop=True)

fig, ax = plt.subplots(1, 1, figsize=(14, 5))

cp_soundings.hmix_grad_theta_v.plot.scatter(
    x="launch_time", ax=ax, c="dodgerblue", label="cold pool"
)
env_soundings.hmix_grad_theta_v.plot.scatter(
    x="launch_time", ax=ax, c="red", label="environment"
)
none_soundings.hmix_grad_theta_v.plot.scatter(
    x="launch_time", ax=ax, c="grey", label="none"
)

ax.legend(loc="upper right")
ax.set_xlabel("Date", fontsize=fs)
ax.set_ylabel(r"$H_{mix}$ from $\theta_v$", fontsize=fs)
ax.tick_params(axis="both", labelsize=fs - 2)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

plt.show()

# Save
plt.savefig(
    "/Users/ninarobbins/Desktop/PhD/ORCESTRA/Figures/dropsondes/cold_pool_classification_dropsondes_PERCUSION_HALO.png",
    bbox_inches="tight",
)
