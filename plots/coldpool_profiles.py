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
dropsonde_ds = find_ml_height_from_gradient(
    dropsonde_ds.sel(alt=slice(low_lim, None)),
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


# Plot
row = 1
col = 5

fig, ax = plt.subplots(row, col, sharey=True, figsize=(17, 6))

variables = ["ta", "theta", "theta_v", "q", "rh"]
variable_titles = [
    "T / $\degree$C",
    "$\\theta$ / K",
    "$\\theta_v$ / K",
    "q / kg kg${-1}$",
    "RH / %",
]

for i, axis in enumerate(ax):
    axis.plot(
        cp_soundings[variables[i]].mean(dim="launch_time", skipna=True),
        cp_soundings.alt,
        c="dodgerblue",
        label="cold pool",
    )
    axis.plot(
        env_soundings[variables[i]].mean(dim="launch_time", skipna=True),
        env_soundings.alt,
        c="red",
        label="env",
    )
    axis.plot(
        none_soundings[variables[i]].mean(dim="launch_time", skipna=True),
        none_soundings.alt,
        c="grey",
        label="undefined",
    )

    axis.set_xlabel(r_titles[i], fontsize=fs)
    axis.tick_params(axis="both", labelsize=fs - 2)
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)

ax[0].set_ylabel(r"Altitude / m", fontsize=fs - 2)
ax[0].legend(fontsize=fs, bbox_to_anchor=(0.3, 0.99))

# Save
plt.savefig(
    "/Users/ninarobbins/Desktop/PhD/ORCESTRA/Figures/dropsondes/cold_pool_profiles_dropsondes_PERCUSION_HALO.png",
    bbox_inches="tight",
)

plt.show()
