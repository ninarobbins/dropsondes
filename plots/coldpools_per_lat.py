import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


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
# Make new CP flag variable
cp_flag = np.full(len(dropsonde_ds.launch_time), 0)

for i in range(len(dropsonde_ds.launch_time)):
    if dropsonde_ds.hmix_grad_theta_v.isel(launch_time=i) < 400:
        cp_flag[i] = 1

alt = dropsonde_ds.alt.values
lats = dropsonde_ds.lat.values

# Mask NaNs
mask = ~np.isnan(lats)

# Create interpolator
f = interp1d(alt[mask], lats[mask], bounds_error=False, fill_value="extrapolate")

# Apply interpolator to the full range of altitudes
lats_interpolated = f(alt)
print(lats_interpolated[:, 200])

lats = dropsonde_ds.lat.interpolate_na(dim="alt", fill_value="extrapolate").fillna(
    method="pad"
)
lats = dropsonde_ds.lat.isel(alt=200)
print(lats.values)
# Make PDF of number of cold pools per latitude
plt.hist2d(lats, cp_flag, bins=5, cmap="Blues")
