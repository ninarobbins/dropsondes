# %%
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import sys

sys.path.append("./")
sys.path.append("../")

import droputils.rough_segments as segments  # noqa: E402
import droputils.data_utils as data_utils  # noqa: E402
import droputils.circle_products as circle_products  # noqa: E402


# %%


level_3_path = "/Users/helene/Documents/Data/Dropsonde/complete/dropsondes/Level_3/PERCUSION_HALO_Level_3.nc"


ds_lev3 = xr.open_dataset(level_3_path)

flight_ids = list(segments.starts.keys())
flight_id = flight_ids[0]
dict_ds_c = data_utils.get_circle_data(ds_lev3, flight_id)


# %%

dict_ds_c = data_utils.get_circle_data(ds_lev3, flight_id)
c_name = list(dict_ds_c.keys())[0]
circle_south = dict_ds_c[c_name]

# circle_south = circle_south.expand_dims({"circle":1})
# circle_south = circle_south.assign_coords(circle=("circle", [f"{flight_id}_{c_name}"]))

# %%
# gives reasonable results
all_data = []
for flight_id in flight_ids:
    print(flight_id)
    dict_ds_c = data_utils.get_circle_data(ds_lev3, flight_id)
    c_names = list(dict_ds_c.keys())
    flight_c = []
    for c_name in c_names:
        circle = dict_ds_c[c_name]
        circle = circle_products.get_xy_coords_for_circles(circle)
        circle = circle_products.apply_fit2d(circle)
        circle = circle_products.get_div_and_vor(circle)
        circle = circle_products.get_density(circle)
        circle = circle_products.get_vertical_velocity(circle)
        circle = circle_products.get_omega(circle)
        circle = circle.expand_dims({"position": [c_name]})
        circle = circle.expand_dims({"flight_id": [flight_id]})
        flight_c.append(circle.copy())
    try:
        all_data.append(xr.concat(flight_c, dim="position"))
    except ValueError:
        pass
ds = xr.concat(all_data, dim="flight_id")
ds.to_netcdf(
    "/Users/helene/Documents/Data/Dropsonde/complete/dropsondes/Level_3/PERCUSION_HALO_Level_4.nc"
)

# %%
sns.set_palette("turbo", n_colors=6)
fig, axes = plt.subplots(ncols=3, figsize=(18, 6))

ds.sel(position="south")
for c_type, ax in zip(["south", "center", "north"], axes):
    ds_type = ds.sel(position=c_type)
    for flight_id in ds_type.flight_id.values:
        ds_type.sel(flight_id=flight_id).w_vel.plot(ax=ax, y="gpsalt", label=flight_id)
    ax.set_title(f"circles {c_type}")

sns.despine(offset=10)
for ax in axes.flatten():
    ax.legend()

# %%
