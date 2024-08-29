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
# %%

dict_ds_c = data_utils.get_circle_data(ds_lev3, flight_id)
c_name = list(dict_ds_c.keys())[0]
circle_south = dict_ds_c[c_name]

# circle_south = circle_south.expand_dims({"circle":1})
# circle_south = circle_south.assign_coords(circle=("circle", [f"{flight_id}_{c_name}"]))

# %%
# gives reasonable results
circle_south = circle_products.get_xy_coords_for_circles(circle_south)
circle_south = circle_products.apply_fit2d(circle_south)
circle_south = circle_products.get_div_and_vor(circle_south)
circle_south = circle_products.get_density(circle_south)
circle_south = circle_products.get_vertical_velocity(circle_south)
# %%

test_vel = circle_products.get_omega(circle_south)
test_geet = circle_products.get_omega_geet(circle_south)


# %%


sns.set_palette("viridis")
fig, ax = plt.subplots()
# test_vel.omega.plot(y="gpsalt", ax=ax, label="me")
test_geet.omega.plot(y="gpsalt", ax=ax, label="geet")
ax.legend()
