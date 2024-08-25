import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from orcestra.flightplan import LatLon, path_preview, plot_cwv
import intake
import easygems.healpix as egh

from plotting import dropsondes_overlay


# -------------------------------------------
################ PLOT FLIGHT ################

flight_time = datetime(2024, 8, 11, 15, 0, 0)
flight_time_str = flight_time.strftime("%Y-%m-%d")
flight_index = f"HALO-{flight_time.strftime('%Y%m%d')}"
flight_date = flight_index[5:9] + flight_index[9:11] + flight_index[11:13]


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

cat = intake.open_catalog("https://tcodata.mpimet.mpg.de/internal.yaml")

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

dropsonde_ds = xr.open_dataset(
    f"/Volumes/Upload/HALO/Dropsonde/{flight_date}/quickgrid/quickgrid-dropsondes-{flight_date}.nc"
)


# Plot dropsondes
dropsondes_overlay(dropsonde_ds, ax, colormap="Blues", alpha=0.75, nlevels=9)

# Save
plt.savefig(
    f"../figures/HALO-{flight_date}a/IWV_dropsondes_IFS_PERCUSION_HALO_{flight_date}.png",
    bbox_inches="tight",
)
