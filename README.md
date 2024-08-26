Analysis, interesting plots, and quicklooks for the ORCESTRA dropsondes.


# Circle Overview

The circle overview plot can be produced by 
```
python3 plots/circle_profiles.py <flight_id> <config_file>
```
where the config file should be the same as the one used for processing.

I.e. to run it on the example data here it should be 
```
python3 plots/circle_profiles.py 20240811 ./orcestra_drop.cfg
```
That has to be run from the dropsonde folder, because the data directory in ./orcestra_drop.cfg is given relatively. If the config contains an absolute path to the data it does not matter where you call the script. 