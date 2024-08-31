import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime
import cartopy.crs as ccrs
import metpy.calc as mpcalc
from metpy.units import units
import matplotlib.ticker as mticker
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from orcestra.flightplan import LatLon, path_preview, plot_cwv
import intake
import easygems.healpix as egh

# -------------------------------------------
################ PLOT FLIGHT ################

flight_time = datetime(2024, 8, 11, 15, 0, 0)
flight_time_str = flight_time.strftime("%Y-%m-%d")
flight_index = f"HALO-{flight_time.strftime('%Y%m%d')}"
flight_id = flight_index[5:9] + flight_index[9:11] + flight_index[11:13]
flight_date = flight_index[5:9] + "-" + flight_index[9:11] + "-" + flight_index[11:13]

try:
    tracks = xr.open_dataset(
        "/Volumes/ORCESTRA/"
        + flight_index
        + "a/bahamas/QL_"
        + flight_index
        + "a_BAHAMAS_V01.nc"
    )
    path = LatLon(lat=tracks["IRS_LAT"], lon=tracks["IRS_LON"], label=flight_index)

    fig, ax = plt.subplots(
        figsize=(15, 8),
        facecolor="white",
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    path_preview(path, ax=ax, show_waypoints=False, color="#FF5349")

except FileNotFoundError:
    print("BAHAMAS data not found, using standard map without flight track")

    fig, ax = plt.subplots(
        figsize=(15, 8),
        facecolor="white",
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    # Set plot boundaries
    lon_w = -34
    lon_e = -14
    lat_s = 3
    lat_n = 19
    ax.set_extent([lon_w, lon_e, lat_s, lat_n], crs=ccrs.PlateCarree())

    # Assigning axes ticks
    xticks = np.arange(-180, 180, 4)
    yticks = np.arange(-90, 90, 4)

    # Setting up the gridlines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="-",
    )
    gl.xlocator = mticker.FixedLocator(xticks)
    gl.ylocator = mticker.FixedLocator(yticks)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 12, "color": "k"}
    gl.ylabel_style = {"size": 12, "color": "k"}


# # -------------------------------------------
# ################ PLOT IFS ################

# Define dates for forecast initialization and flight
issued_time = datetime(2024, 8, 11, 0, 0, 0)
issued_time_str = issued_time.strftime("%Y-%m-%d")

# Load forecast and plot
cat = intake.open_catalog("https://tcodata.mpimet.mpg.de/internal.yaml")
ds = cat.HIFS(datetime=issued_time).to_dask().pipe(egh.attach_coords)
cwv_flight_time = ds["tcwv"].sel(time=flight_time, method="nearest")

# Plot CWV
plot_cwv(cwv_flight_time)

# # -------------------------------------------
################ PLOT DROPSONDES #############

level_3_path = "/Users/ninarobbins/Desktop/PhD/Dropsondes/data/Level_3/Level_3.nc"
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

dropsonde_ds = dropsonde_ds.sel(launch_time="2024-08-11")

# Compute IWV
iwv = [None] * len(dropsonde_ds["launch_time"])

for i in range(len(dropsonde_ds["launch_time"])):
    try:
        iwv[i] = mpcalc.precipitable_water(
            dropsonde_ds.p.isel(launch_time=i).values * units.Pa,
            mpcalc.dewpoint_from_relative_humidity(
                dropsonde_ds.ta[i], dropsonde_ds.rh[i]
            ).data.magnitude
            * units.degC,
        ).magnitude
    except ValueError:
        continue

dropsonde_ds["iwv"] = (["launch_time"], iwv)

variable = "iwv"
variable_label = r"Integrated Water Vapor / kg m$^{-2}$"
alpha = 1
vmin = 45
vmax = 70
nlevels = 9

cmap = plt.cm.Blues
levels = np.linspace(vmin, vmax, nlevels)
norm = colors.BoundaryNorm(levels, cmap.N)

im_launches = ax.scatter(
    dropsonde_ds["lon"].isel(gpsalt=10),
    dropsonde_ds["lat"].isel(gpsalt=10),
    marker="o",
    edgecolor="grey",
    s=40,
    transform=ccrs.PlateCarree(),
    c=dropsonde_ds[variable],
    cmap="Blues",
    zorder=10,
    norm=norm,
)

divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="3%", pad=0.4, axes_class=plt.Axes)
cbar = plt.colorbar(im_launches, cax=cax, orientation="horizontal", ticks=levels)
cbar.set_label(variable_label)
cbar.set_ticks(levels)
cbar.set_label(variable_label)

# Save
plt.savefig(
    f"/Users/ninarobbins/Desktop/PhD/ORCESTRA/Figures/dropsondes/HALO-{flight_id}a/IWV_dropsondes_IFS_PERCUSION_HALO_{flight_id}.png",
    bbox_inches="tight",
)
