# The Levelised Cost of Hydrogen (LCOH) model

The LCOH Ninja is a Python project to simulate the hydrogen output and cost from green hydrogen plants powered by wind and solar, globally. The wind and solar power profiles are taken from [Renewables.ninja](https://www.renewables.ninja/) (wind) and developed using the [PV Lib module](https://pvlib-python.readthedocs.io/en/stable/) (solar)


## Requirements

[Python](https://www.python.org/) version 3+

Required libraries:
 * numpy
 * xarray
 * matplotlib
 * cartopy
 * pandas
 * scipy
 * joblib
 * datetime
 * os
 * regionmask


## FLOWCHART

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}} }%%
flowchart LR
RenewableProfiles["`**Renewable Profiles**<br>
    Generated from NASA MERRA-2 Data<br>
    Using renewable.calculator.py`"]
Files["`**File Input Reader**<br>
    Renewable profiles and input assumptions are read in<br>
    Using geodata_v3.py and filepreprocessor_v3.py`"]
ElectrolyserModel["`**Electrolyser Model**<br>
    Renewable profiles are smoothed and used to generate hydrogen outputs<br>
    Using electrolyser_v4.py`"]
EconomicModel["`**Economic Model**<br>
    Deployment and operating costs are estimated for each gridpoint<br>
    Using economicmodel_v4.py`"]
hydrogenModel["`**Hydrogen Model**<br>
    Main file, runs all other files and optimises the electrolyser size at each gridpoint<br>
    Using hydrogenmodel_v4.py`"]
    RenewableProfiles --> Files --> ElectrolyserModel --> EconomicModel --> hydrogenModel
```

## SETUP

### Generate and download renewable power profiles
First, download the necessary input data from NASA's [MERRA-2 reanalysis](https:///gmao.gsfc.nasa.gov/reanalysis/MERRA-2/), including windspeed at 10m and solar irradiance. Using [PV Lib](https://pvlib-python.readthedocs.io/en/stable/) and the [Virtual Wind Farm Model](https://github.com/renewables-ninja/vwf/tree/master), generate renewable profiles for the year and geographies of interest. Renewable profiles can also be developed using the renewable_calculator.py script.

Save the files into the running folder under /MERRA2_INPUTS/SOLAR_CF and /MERRA2_INPUTS/WIND_CF and with the necessary filenames.

### Set up the LCOH model
With the required libraries installed, the model should just require editing of the paths in the hydrogen_model.py script and setting the conditions for the model run in lines 862 - 870 (difference in solar fraction, electrolyser technology, electrolyser size optimisation, electrolyser capex and the number of cores).

### Process renewables data into hydrogen production and economics
Performing the LCOH model run is very computationally expensive and requires ~16 GB working memory on your computer. To run the model, run the hydrogen_model.py script with the desired regions and input files. This process should yield a set of NetCDF files in the output folder that has been specified, which contain details of the cost, investment requirements and hydrogen output at each MERRA-2 gridcell.


## USAGE INSTRUCTIONS

First, edit the paths in the hydrogen_model script and ensure that the input folders (DATA, MERRA2_INPUTS/WIND_CF/ and MERRA2_INPUTS/SOLAR_CF) are set up correctly.

Second, ensure that all the required data files are in the /DATA/ folder. Several files need to be downloaded and directly added to that folder, including: [ETOPO_bathymetry.nc](https://www.ncei.noaa.gov/products/etopo-global-relief-model) and [distance2shore.nc](https://catalog.data.gov/dataset/distance-to-nearest-coastline-0-04-degree-grid). 

Third, decide on the model parameters that you wish to use, including: 
 * fract_diff (difference in solar fraction being computed e.g., 0.25 to give results for 0%, 25%, 50%, 75%...)
 * elec_tech (PEM or Alkaline), optimisation (whether to optimise the electrolyser size)
 * elec_capex (cost of the electrolyser in USD/kW)
 * num_cores (number of computing cores to use for the parallelisation) and latitude and longitude limits for the calculation

Fourth, run the hydrogen_model.py. I recommend that you save a different version for each model configuration run. Given the large volumes of data and the optimisation of electrolyser size on a location by location basis, it is very computationally intensive (e.g., 3 weeks for a year of data globally on a 16-core Xeon processor).

## POSTPROCESSING

Graph plotting functions and other postprocessing scripts (e.g., cost-supply curve generation, LCOH heatmaps) are included in the postprocessing folder. 


## LICENSE
BSD 3-Clause License
Copyright (C) 2023-2025  Luke Hatton
All rights reserved.

See `LICENSE` for more detail


## Citation

The LCOH code was developed by Luke Hatton, who can be contacted at l.hatton23@imperial.ac.uk

L Hatton, 2025.  A global, levelised cost of hydrogen (LCOH) model. Currently under review.

