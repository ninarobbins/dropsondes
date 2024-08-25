import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../quicktools")))
from quickgrid import grid_together, derived_products
from datetime import datetime

flight_time = datetime(2024, 8, 11, 0, 0, 0)
flight_id = flight_time.strftime("%Y%m%d")

# Flight directory
dropsonde_dir = f"/Volumes/ORCESTRA/HALO-{flight_id}a/dropsondes"

# Grid flight dataset from Level 1
dropsonde_ds = grid_together(dropsonde_dir)
dropsonde_ds = derived_products(dropsonde_ds)

# Change in theta_v from moisture
