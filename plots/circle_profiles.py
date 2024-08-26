# %%
import sys
import os
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append("./")
sys.path.append("../")
import droputils.plot_utils as plot_utils  # noqa: E402
import droputils.data_utils as data_utils  # noqa: E402
import droputils.physics_utils as physics  # noqa: E402

# %%
flight_id = sys.argv[1]
config_path = sys.argv[2]

config = data_utils.get_config(config_path)
l3_path = data_utils.get_l3_path(config, flight_id)

ds = xr.open_dataset(l3_path)
dict_ds_c = data_utils.get_circle_data(ds, flight_id)

# %%
plt_variables = ["theta", "rh", "u", "v"]
plt_units = ["K", "%", r"m s$^{-1}$", r"m s$^{-1}$"]
alt_var = "gpsalt"

# %%
fig, axes = plt.subplots(ncols=len(plt_variables), figsize=(18, 6))

# prepare adding of ticks
yticks = np.arange(0, 16000, 2000).tolist()
ytick_labels = yticks.copy()
ytick_colors = ["black"] * len(yticks)
colors = plot_utils.get_colors(dict_ds_c.keys())

# prepare xticks in rh plot
axes[1].set_xlim(0, 1)
xticks = axes[1].get_xticks().tolist()
xtick_labels = [np.round(curr, 1) for curr in xticks]
xtick_colors = ["black"] * len(xticks)
counter = 0  # needed for positions of annotations

sns.despine(offset=10)
for circle, ds_c in dict_ds_c.items():
    # remove nan sondes
    ds_c = (
        ds_c.where(ds_c["ta"].isnull().sum(dim=alt_var) < 1300, drop=True)
        .where(ds_c["rh"].isnull().sum(dim=alt_var) < 1300, drop=True)
        .where(ds_c["p"].isnull().sum(dim=alt_var) < 1300, drop=True)
    )
    nb_sondes = ds_c.sizes["sonde_id"]
    color = colors[circle]
    print(f"Circle {circle} has {nb_sondes} valid sondes")

    # plot
    for ax, plt_var, unit in zip(axes, plt_variables, plt_units):
        ds_c[plt_var].mean("sonde_id").plot(y=alt_var, ax=ax, label=circle, color=color)
        ax.set_xlabel(f"{plt_var} / {unit}")

    # calculate annotations
    fl_mean, fl_ind = physics.get_levels_circle(ds_c, alt_var=alt_var)
    max_rh = physics.get_rh_max_circle(ds_c, hmin=8000, alt_var=alt_var)
    lcl_pressure, lcl_temperature = physics.get_lcl_circle(ds_c, alt_var=alt_var)
    indices = physics.get_heights_from_array(
        ds_c["p"], values=lcl_pressure.magnitude, alt_var=alt_var
    )
    lcl_height = ds_c[alt_var].isel({alt_var: indices}).values
    # add yticks freezing
    ytick_labels = ytick_labels + [""] * nb_sondes
    yticks = yticks + fl_ind.tolist()
    ytick_colors = ytick_colors + [colors[circle]] * nb_sondes
    # add freezing level line
    for ax in axes[:]:
        ax.axhline(fl_mean, color=color, alpha=0.2)
    # add annotation freezing level
    axes[0].text(300, 7500, "Freezing level")
    axes[0].text(
        300, 7500 - (counter + 1) * 500, f"{circle}: {int(fl_mean)}m", color=color
    )
    # add xticks rh
    xtick_labels = xtick_labels + [""]
    xticks = xticks + [max_rh["rh"].mean()]
    xtick_colors = xtick_colors + [colors[circle]]
    # add lines rh
    axes[1].axvline(max_rh["rh"].mean(), color=color, alpha=0.3)
    axes[1].axhline(max_rh["height"].mean(), color=color, alpha=0.3)
    # add annotation RH
    axes[1].text(0, -500, "Max RH above 8000m:")
    axes[1].text(
        1 - ((counter + 1) * 0.2), -1000, np.round(max_rh["rh"].mean(), 2), color=color
    )
    axes[1].text(0, 12500 - counter * 750, int(max_rh["height"].mean()), color=color)
    # add LCL annotation
    axes[0].text(340, 2000, "LCL")
    axes[0].text(
        340,
        2000 - (counter + 1) * 500,
        f"{circle}: {int(lcl_height.mean())}m ",
        color=color,
    )

    counter += 1


# set freezing level yticks
plot_utils.set_yticks_with_color(axes, yticks, ytick_colors, labels=ytick_labels)
# set rh xticks
plot_utils.set_xticks_with_color([axes[1]], xticks, xtick_colors, labels=xtick_labels)

# set freezing level_ticks

# plot editing
fig.suptitle(f"Flight {flight_id} (circle means)")
# add zero line for wind
for ax in axes[-2:].flatten():
    ax.axvline(0, color="grey", alpha=0.5)
axes[0].set_ylabel(
    f'{ds_c[alt_var].attrs['long_name']} / {ds_c[alt_var].attrs['units']} '
)
axes[0].legend()
for ax in axes[1:].flatten():
    ax.set_ylabel("")
fig.tight_layout()


quicklook_path = os.path.dirname(l3_path.replace("Level_3", "Quicklooks"))
fig.savefig(f"{quicklook_path}/{flight_id}_circle_profiles.png", dpi=80)


# %%
