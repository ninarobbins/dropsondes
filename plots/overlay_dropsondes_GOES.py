import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from orcestra.flightplan import LatLon, path_preview
from orcestra.weathermaps import goes_overlay, dropsondes_overlay

import sys

sys.path.append("./")
sys.path.append("../")
from droputils.physics_utils import add_iwv


# -------------------------------------------
################ PLOT FLIGHT ################

flight_time = datetime(2024, 8, 11, 12, 0, 0)
sat_time_str = flight_time.strftime("%Y-%m-%d %H:%M")
flight_time_str = flight_time.strftime("%Y-%m-%d")
flight_index = f"HALO-{flight_time.strftime('%Y%m%d')}a"
flight_id = flight_index[5:9] + flight_index[9:11] + flight_index[11:13]
flight_date = flight_index[5:9] + "-" + flight_index[9:11] + "-" + flight_index[11:13]

try:
    tracks = xr.open_dataset(
        "/Volumes/ORCESTRA/"
        + flight_index
        + "/bahamas/QL_"
        + flight_index
        + "_BAHAMAS_V01.nc"
    )
    path = LatLon(lat=tracks["IRS_LAT"], lon=tracks["IRS_LON"], label=flight_index)

    fig, ax = plt.subplots(
        figsize=(15, 8),
        facecolor="white",
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    path_preview(path, ax=ax, show_waypoints=False, color="#FF5349")

except FileNotFoundError:
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


# -------------------------------------------
################ PLOT GOES-16 ################

goes_overlay(sat_time_str, ax)


# # -------------------------------------------
################ PLOT DROPSONDES #############

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

dropsondes_overlay(
    dropsonde_ds,
    ax,
    variable="iwv",
    variable_label=r"Integrated Water Vapor / kg m$^{-2}$",
    colormap="Blues_r",
    markershape="o",
    markersize=40,
    edgecolor="grey",
)

# Save
plt.savefig(
    f"/Users/ninarobbins/Desktop/PhD/ORCESTRA/Figures/dropsondes/HALO-{flight_id}a/IWV_dropsondes_GOES_PERCUSION_HALO_{flight_id}.png",
    bbox_inches="tight",
)
