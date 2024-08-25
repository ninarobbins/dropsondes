import xarray as xr
import numpy as np
import glob
from tqdm import tqdm
import metpy.calc as mpcalc
from metpy.units import units
import logging


def grid_together(
    flight_dir, max_alt=14000, vertical_spacing=10, fullness_threshold=0.2
):
    l1_dir = flight_dir + "/Level_1/"

    allfiles = sorted(glob.glob(l1_dir + "*QC.nc"))

    ori_list = [None] * len(allfiles)

    check_vars = ["alt", "pres", "u_wind", "v_wind", "lat", "lon", "mr"]

    g = 0
    fail_counter = 0
    for i in tqdm(allfiles):
        sonde = i.replace(f"{l1_dir}", "").replace("QC.nc", "")

        check_vars = ["pres", "u_wind", "v_wind", "lat", "lon", "mr"]
        ds = xr.open_dataset(i)
        fraction_list = np.array(
            [(ds[var].count(dim="time") / ds.time.count()).values for var in check_vars]
        )

        where_fraction = np.where(fraction_list <= fullness_threshold)

        vars_low_fraction = ", ".join([check_vars[i] for i in where_fraction[0]])
        str_fraction_with_vars = "".join(
            [f"{check_vars[i]} : {fraction_list[i]}" + "\n" for i in where_fraction[0]]
        )

        if len(where_fraction[0]) != 0:
            print_msg = (
                f"For sonde {sonde}, variable/s {vars_low_fraction} found to have profile fullness less than {fullness_threshold}. Ignoring this sonde now for quicklooks purposes. Check this sonde later during final QC. Following are the relevant profile-fullness fractions."
                + "\n"
                + f"{str_fraction_with_vars}"
            )

            logging.info(print_msg)
            print(print_msg)

            fail_counter += 1

        else:
            ori_list[g] = (
                xr.open_dataset(i)
                .drop("alt")
                .dropna(dim="time", subset=["time"])
                .isel(obs=0)
                .swap_dims({"time": "gpsalt"})
                .reset_coords()
                .dropna(
                    dim="gpsalt",
                    subset=["pres", "u_wind", "v_wind", "lat", "lon", "mr"],
                    how="any",
                )
                .rename({"gpsalt": "alt"})
                .interp(alt=np.arange(0, max_alt + vertical_spacing, vertical_spacing))
            )

        g = g + 1

    ds_list = list(filter(None, ori_list))
    ds_flight = xr.concat(ds_list, dim="launch_time")

    logging.info(
        f"{fail_counter} out of the {len(allfiles)} sondes provided did not contain enough data and hence have been excluded from this quick processing."
    )

    logging.info(
        f"Gridded all individual sondes for {l1_dir} at {vertical_spacing} m vertical resolution to {max_alt} m"
    )

    return ds_flight


def derived_products(ds_flight):
    ds_flight["T"] = ds_flight["tdry"] + 273.15

    ds_flight["q"] = (
        ["launch_time", "alt"],
        mpcalc.specific_humidity_from_mixing_ratio(ds_flight["mr"]).data,
    )

    iwv = [None] * len(ds_flight["launch_time"])

    for i in range(len(ds_flight["launch_time"])):
        iwv[i] = mpcalc.precipitable_water(
            ds_flight["pres"].isel(launch_time=i).values * units.mbar,
            ds_flight["dp"].isel(launch_time=i).values * units.degC,
        ).magnitude

    ds_flight["iwv"] = (["launch_time"], iwv)

    logging.info("Added derived variables: T, q and iwv to ds_flight")

    return ds_flight


def save_ds(ds_flight, flight_dir):
    save_filepath = (
        f"{flight_dir.dir}PERCUSION_HALO_Dropsondes_quickgrid_{flight_dir.flightdir}.nc"
    )
    ds_flight.to_netcdf(save_filepath)

    logging.info(f"File Saved:{save_filepath}")
