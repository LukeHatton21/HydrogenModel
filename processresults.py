import numpy as np
import xarray as xr
import os
import re
import glob
import pandas as pd
import warnings
from abatement_model import AbatementEstimator
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)

class ResultsProcessor:
    def __init__(self, data_root, output_folder, technology_scenario):
        """ Class to process results from the Hydrogen Model"""

        self.data_root = data_root.rstrip("/")
        self.output_folder = output_folder
        self.technology_scenario = technology_scenario

        self.paths = {
            "country_grid_combined": f"{self.data_root}/country_grid_combined.nc",
            "country_grids": f"{self.data_root}/country_grids.nc",
            "country_mapping": f"{self.data_root}/country_mapping.csv",
            "country_mapping_regions": f"{self.data_root}/country_mapping_regions.csv",
            "wind_cf": f"{self.data_root}/WIND_CF_MEAN.2022.nc",
            "solar_cf": f"{self.data_root}/SOLAR_CF_MEAN.2022.nc",
            "land_cover": f"{self.data_root}/GlobalLandCover.nc",
            "land_mapping": f"{self.data_root}/LandUseCSV.csv",
            "ember": f"{self.data_root}/Ember Yearly Data 2026.csv",
            "ifi": f"{self.data_root}/IFI Grid Factors.csv",
            "gem": f"{self.data_root}/Global-Integrated-Power-February-2026.csv",
            "pem_ref": f"{self.data_root}/Solar_2000USD_REAL_2026.nc",
        }

        # Technology config
        if technology_scenario in ["PEM", "PEM_2030"]:
            self.technology = "PEM"
        else:
            self.technology = "ALK"

        cost_dict = {"PEM": 2000, "PEM_2030": 1391, "ALK": 1700, "ALK_2030": 1183}
        year_dict = {"PEM": 2026, "PEM_2030": 2030, "ALK": 2026, "ALK_2030": 2030}
        self.cost = cost_dict[self.technology_scenario]
        self.year = year_dict[self.technology_scenario]
        self.lifetime = 20

        # Load required data only
        self.raw_country_data = xr.open_dataset(self.paths["country_grid_combined"])
        self.country_data = xr.open_dataset(self.paths["country_grids"])
        self.wind_cf = xr.open_dataset(self.paths["wind_cf"])
        self.solar_cf = xr.open_dataset(self.paths["solar_cf"])
        self.land_cover = xr.open_dataset(self.paths["land_cover"])
        self.land_mapping = pd.read_csv(self.paths["land_mapping"])
        self.country_index = pd.read_csv(self.paths["country_mapping"])
        self.country_index.rename(columns={"country": "Country code"}, inplace=True)

        self.pem_solar = xr.open_dataset(self.paths["pem_ref"])
        self.country_grid = self.get_countries_in_required_resolution(self.raw_country_data, self.pem_solar)

        # Abatement estimator
        self.abatement_class = AbatementEstimator(
            country_grids=self.country_grid,
            country_mapping=self.paths["country_mapping"],
            ember_data=self.paths["ember"],
            IFI_data=self.paths["ifi"],
            GEM_data=self.paths["gem"],
        )

        # Scenario results folder
        self.folder = f"{self.data_root}/{self.technology_scenario}"

        self.read_in_hybrid_results(self.technology, self.folder, self.cost)
        self.cheapest = self.get_cheapest_lcoh(self.technology)



        
    def get_countries_in_required_resolution(self, country_data, target_data):

        # Reindex using the xarray reindex function
        new_coords = {'latitude': target_data.latitude, 'longitude': target_data.longitude}
        countries_resampled = country_data.reindex(new_coords, method='nearest')

        
        # Create new dataset with countries
        data_vars = {'country': countries_resampled['country']}
        coords = {'latitude': target_data.latitude,
                  'longitude': target_data.longitude}
        countries_output = xr.Dataset(data_vars=data_vars, coords=coords)
        
        return countries_output    
        
        
    def read_in_hybrid_results(self, technology, results_folder, cost):
        # Custom sort key to extract solar fraction as an integer (0-100)
        def solar_fraction_key(filepath):
            filename = os.path.basename(filepath)

            if filename.startswith("Wind"):
                return 0

            # Extract numeric solar fraction e.g. "Solar90%_..." -> 90
            match = re.search(r"Solar(\d+)", filename)
            if match:
                return int(match.group(1))

            # "Solar_2000USD..." with no digits = 100% solar
            if filename.startswith("Solar_"):
                return 100

            return -1  # Fallback for unexpected filenames
        
        # Get correct folder
        results_folder = results_folder + "/"
        
        # Get wind and solar results
        setattr(self, technology + "_wind_results", xr.open_dataset(results_folder + "Wind_" + str(technology) + "NOM" + str(cost) + f"USD_{self.year}.nc"))
        setattr(self, technology + "_solar_results", xr.open_dataset(results_folder + "Solar_" + str(technology) + "NOM" + str(cost) + f"USD_{self.year}.nc"))
        
        # Read in hybrid files and set up empty combined dataset
        hybrid_files = glob.glob(results_folder + "*.nc")
        combined_ds = None
        
        # Sort the files based on the solar fraction
        hybrid_files = sorted(hybrid_files, key=solar_fraction_key)
        
        # Get solar results
        attr_name = f"{technology}_solar_results"
        solar_results = getattr(self, attr_name)
        
        
        # Extract regions
        country_grids = xr.open_dataset(self.data_root + "country_grid_combined.nc")
        country_mapping = pd.read_csv(self.data_root + "country_mapping_regions.csv", encoding="cp1252")
        country_grids = country_grids.rename({"country":"index"})
        country_df = country_grids.to_dataframe().reset_index()
        country_df = country_df.merge(country_mapping, how="left", on="index")
        country_ds = country_df.drop_duplicates(subset=["latitude", "longitude"]).set_index(["latitude", "longitude"]).to_xarray()
        country_ds = country_ds.assign_coords({"latitude":country_ds.latitude, "longitude": country_ds.longitude})
        country_ds = country_ds.reindex(latitude=solar_results.latitude, longitude=solar_results.longitude, method="nearest")
        
        
        for i, file in enumerate(hybrid_files):
            

            # Get the variable name and dataset
            variable_name = technology + f"_{(i)*10}"
            ds = xr.open_dataset(file)
            
            
            if i == 0:
                country_grid = self.get_countries_in_required_resolution(self.raw_country_data, ds["levelised_cost"])

            
            # Calculate the average electricity production across the year
            ds["electricity_production"] = ds["renewable_electricity"]
            
            # Add region
            ds["region"] = country_ds["region"]

            # Calculate the abatement potential
            ds = self.abatement_class.calculate_abatement_potential(ds)
            
            
            # Expand the dimensions
            ds = ds.expand_dims({"solar_fraction":[i*10]})
            
            # Calculate the embodied carbon
            if i == 0:
                tech = "Wind"
            elif i == 10:
                tech = "Solar"
            else:
                tech = "Hybrid"
            
            # Calculate technical potential
            supply_ds = self.get_supply_curves(ds["levelised_cost"], ds["hydrogen_production"], tech=tech, solar_fractions=ds["solar_fraction"]/100, technology=technology)
            ds["hydrogen_technical_potential"] = supply_ds["hydrogen_technical_potential"]

            
            # Combine the dataset
            if combined_ds is None:
                combined_ds = ds
            else:
                combined_ds = combined_ds.combine_first(ds)
                
            ds = ds.sel(solar_fraction = i*10)
            ds = ds.drop_vars('solar_fraction')
            # Save the dataset onto the ResultsProcessor object
            setattr(self, variable_name, ds)
            print(f"Read in file {i}")
            
        # Save the combined results onto the ResultsProcessor object
        setattr(self, technology + "_results", combined_ds)
            
    def get_cheapest_lcoh(self, technology):
        
        # Extract collated results
        if technology == "PEM":
            collated_results = self.PEM_results
        else:
            collated_results = self.ALK_results
            
        # Extract the LCOH for the specified technology
        lcoh = collated_results['levelised_cost']
        
        # Calculate the cheapest lcoh across the dimensions
        min_lcoh = lcoh.min(dim="solar_fraction")
        
        # Get the minimum index
        min_index = lcoh.idxmin(dim="solar_fraction")
        
        # Extract the cheapest solar fraction at each location
        indexed_results = collated_results.sel(solar_fraction=min_index, method='nearest')
        
        # Add solar fraction to indexed_results
        indexed_results['Optimal_SF'] = min_index
        # Add to the ResultsProcessor class
        setattr(self,technology + "_cheapest_sf", indexed_results)
        
        return indexed_results

    
    def calculate_shares_levelised_cost(self, ds):
        
        # Extract onshore costs
        ds["onshore_costs"] = xr.where(np.isnan(ds["levelised_cost"]), np.nan,ds["wind_costs"])
        
        # Calculate LCOH without discounting
        ds["LCOH_NoCoC"] = (ds["total_capital_costs"]/ds["hydrogen_production"]/20/1000) + (ds["solar_costs"]*0.023*ds["solar_fraction"]/100 + ds["onshore_costs"]*0.026*(1-ds["solar_fraction"]/100) + ds["electrolyser_costs"]*0.03)/ds["hydrogen_production"]/1000 + 2 * 9/1000
        ds["COC_LCOH"] = 1 - ds["LCOH_NoCoC"]/ds["levelised_cost"]
        ds["CAPEX_LCOH"] = 1 - ds["LCOH_NoCoC"]/ds["levelised_cost"]
        
        # Drop interim calculations
        ds = ds.drop_vars(["LCOH_NoCoC", "onshore_costs"])
        
        return ds

    def get_utilisations(self, annual_production, tech, solar_fractions=None, technology=None):
        
        latitudes = annual_production.latitude.values
        longitudes = annual_production.longitude.values
        target_grid = annual_production.isel(solar_fraction=0, drop=True) if "solar_fraction" in annual_production.dims else annual_production
        global_cover = self.land_cover.reindex_like(target_grid)
        mapping = self.land_mapping
        attr_name = f"{technology}_solar_results"
        solar_results = getattr(self, attr_name)
        
        utilisation = xr.zeros_like(annual_production)
        utilisation = utilisation.reindex_like(target_grid)
        utilisation = utilisation.fillna(1)
        for i in np.arange(0, 21, 1):
            # Use xarray's where and isin functions to map land use categories to values
            if tech == "Solar":
                utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['PV LU'].iloc[i], utilisation)
            elif tech =="Wind":
                utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['Wind LU'].iloc[i], utilisation)
                utilisation = xr.where(np.isnan(solar_results['levelised_cost']), 1, utilisation)      
            elif tech =="Hybrid":
                wind_utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['Wind LU'].iloc[i], utilisation)
                wind_utilisation = xr.where(np.isnan(solar_results['levelised_cost']), 1, wind_utilisation)
                solar_utilisation = xr.where(global_cover['cover'] == mapping['Number'].iloc[i], mapping['PV LU'].iloc[i], utilisation)
                utilisation = solar_fractions * solar_utilisation + wind_utilisation * (1 - solar_fractions)
                
        return utilisation    
        
    def get_supply_curves(self, levelised_costs, annual_production, tech, plot_global=None, solar_fractions=None, technology=None):
    
        # Calculate area of each grid point in kms 
        latitudes = annual_production.latitude.values
        longitudes = annual_production.longitude.values
        grid_areas = self.get_areas(annual_production)
        utilisation_factors = self.get_utilisations(annual_production, tech, solar_fractions, technology)
        
        # Set out constants
        if tech == "Wind":
            power_density = 6520 # kW/km2
        elif tech == "Solar":
            power_density = 32950  # kW/km2
        elif tech == "Hybrid":
            power_density = 6520 + solar_fractions * (32950 - 6520)
        installed_capacity = 1000
        
        # Scale annual hydrogen production by turbine density
        max_installed_capacity = power_density * grid_areas['area'] * utilisation_factors
        ratios = max_installed_capacity / installed_capacity
        technical_hydrogen_potential = annual_production * ratios
        
        # Create new dataset with cost and production volume
        if solar_fractions is not None:
            data_vars = {'Optimal_SF': solar_fractions, 'hydrogen_technical_potential': technical_hydrogen_potential,
                     'levelised_cost': levelised_costs, 'country': self.country_grid['country']}
        else:
            data_vars = {'hydrogen_technical_potential': technical_hydrogen_potential,
                     'levelised_cost': levelised_costs, 'country': self.country_grid['country']}
        coords = {'latitude': latitudes,
                  'longitude': longitudes}
        supply_curve_ds = xr.Dataset(data_vars=data_vars, coords=coords)
        
        return supply_curve_ds
    


    
    def get_areas(self, annual_production):
        
        latitudes = annual_production.latitude.values
        longitudes = annual_production.longitude.values

        # Add an extra value to latitude and longitude coordinates
        latitudes_extended = np.append(latitudes, latitudes[-1] + np.diff(latitudes)[-1])
        longitudes_extended = np.append(longitudes, longitudes[-1] + np.diff(longitudes)[-1])

        # Calculate the differences between consecutive latitude and longitude points
        dlat_extended = np.diff(latitudes_extended)
        dlon_extended = np.diff(longitudes_extended)
        
        # Calculate the Earth's radius in kms
        radius = 6371

        # Compute the mean latitude value for each grid cell
        mean_latitudes_extended = (latitudes_extended[:-1] + latitudes_extended[1:]) / 2
        mean_latitudes_2d = mean_latitudes_extended[:, np.newaxis]

        # Convert the latitude differences and longitude differences from degrees to radians
        dlat_rad_extended = np.radians(dlat_extended)
        dlon_rad_extended = np.radians(dlon_extended)

        # Compute the area of each grid cell using the Haversine formula
        areas_extended = np.outer(dlat_rad_extended, dlon_rad_extended) * (radius ** 2) * np.cos(np.radians(mean_latitudes_2d))

        # Create a dataset with the three possible capital expenditures 
        area_dataset = xr.Dataset()
        area_dataset['latitude'] = latitudes
        area_dataset['longitude'] = longitudes
        area_dataset['area'] = (['latitude', 'longitude'], areas_extended, {'latitude': latitudes, 'longitude': longitudes})
        
        return area_dataset
       
    def expand_end_use_flags(self, df):
        """
        Convert end-use indicator columns into a single 'EndUse' column.

        For rows with multiple end-use flags equal to 1, duplicate the row once
        for each flagged end-use.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe containing end-use flag columns.

        Returns
        -------
        pandas.DataFrame
            Reshaped dataframe with a new 'EndUse' column.
        """

        end_use_cols = [
            'EndUse_Refining',
            'EndUse_Ammonia',
            'EndUse_Methanol',
            'EndUse_Iron&Steel',
            'EndUse_Other Ind',
            'EndUse_Mobility',
            'EndUse_Power',
            'EndUse_Grid inj.',
            'EndUse_CHP',
            'EndUse_Domestic heat',
            'EndUse_Biofuels',
            'EndUse_Synfuels',
            'EndUse_CH4 grid inj.',
            'EndUse_CH4 mobility'
        ]

        missing_cols = [col for col in end_use_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing end-use columns: {missing_cols}")

        # Keep all non-end-use columns as identifiers
        id_cols = [col for col in df.columns if col not in end_use_cols]

        # Convert from wide to long
        df_long = df.melt(
            id_vars=id_cols,
            value_vars=end_use_cols,
            var_name="EndUse",
            value_name="flag"
        )

        # Keep only rows where the flag indicates the end-use is active
        df_long = df_long[df_long["flag"] == 1].copy()

        # Clean the EndUse labels by removing the prefix
        df_long["EndUse"] = df_long["EndUse"].str.replace("EndUse_", "", regex=False)

        # Drop the temporary flag column
        df_long = df_long.drop(columns="flag").reset_index(drop=True)

        return df_long
        
    def add_nearest_xarray_value_to_df(self,
    df,
    dataarray,
    variable=None,
    df_lat_col="Latitude",
    df_lon_col="Longitude",
    da_lat_col="latitude",
    da_lon_col="longitude",
    output_col="mitigation_potential"
):
        """
        Match nearest grid-cell values from an xarray DataArray to dataframe lat/lon points.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing latitude and longitude columns.
        dataarray : xarray.DataArray
            Xarray DataArray with latitude and longitude coordinates.
        df_lat_col : str
            Latitude column in df.
        df_lon_col : str
            Longitude column in df.
        da_lat_col : str
            Latitude coordinate name in dataarray.
        da_lon_col : str
            Longitude coordinate name in dataarray.
        output_col : str
            Name of output column to add to df.

        Returns
        -------
        pandas.DataFrame
            Copy of dataframe with matched values in output_col.
        """

        df = df.copy()
        # If Dataset, extract a variable
        if isinstance(dataarray, xr.Dataset):
            if variable is None:
                raise ValueError(
                    "Input is an xarray.Dataset. Please provide `variable`."
                )
        
        if variable not in dataarray.data_vars:
            raise KeyError(f"'{variable}' not found in Dataset variables")
        dataarray = dataarray[variable]
        if df_lat_col not in df.columns:
            raise KeyError(f"'{df_lat_col}' not found in dataframe")
        if df_lon_col not in df.columns:
            raise KeyError(f"'{df_lon_col}' not found in dataframe")

        if da_lat_col not in dataarray.coords:
            raise KeyError(f"'{da_lat_col}' not found in DataArray coordinates")
        if da_lon_col not in dataarray.coords:
            raise KeyError(f"'{da_lon_col}' not found in DataArray coordinates")

        lat_vals = xr.DataArray(df[df_lat_col].values, dims="points")
        lon_vals = xr.DataArray(df[df_lon_col].values, dims="points")

        matched = dataarray.sel(
            {da_lat_col: lat_vals, da_lon_col: lon_vals},
            method="nearest"
        )

        df[output_col] = matched.values

        return df

            
# Call the Results Processor Model for PEM
results_model_pem = ResultsProcessor(
    data_root="./DATA",
    output_folder="./OUTPUTS",
    technology_scenario="PEM"
)
ds = results_model_pem.cheapest
ds.to_netcdf("./OUTPUTS/PEM_Results_Cheapest_Tech.nc")

## IEA Data
# Get the IEA project data
iea_projects = pd.read_csv("./DATA/IEA_Hydrogen_Projects.csv")
benchmarks = pd.read_csv("./DATA/mitigation_potentials.csv")

# Merge selected iea projects onto the global results
selected_iea_projects = iea_projects .merge(benchmarks, how="left", on="EndUse")
selected_iea_projects = iea_projects.loc[(iea_projects["Technology"].isin(["PEM", "Alkaline", "SOEC", "Other Electrolysis"])) & ~(iea_projects["Technology_electricity"].isin(["Grid", "Nuclear"]))]
selected_iea_projects = results_model_pem.expand_end_use_flags(df=selected_iea_projects)
selected_iea_projects = selected_iea_projects.merge(benchmarks, how="left", on="EndUse")
for x in ["mitigation_potential", "IFI_Emissions_Factor", "Abatement_EV", "Abatement_IND", "Abatement_HP", "hydrogen_production", "renewable_electricity"]:
    selected_iea_projects = results_model_pem.add_nearest_xarray_value_to_df(selected_iea_projects,ds[x],
        df_lat_col="Latitude",
        df_lon_col="Longitude",
        da_lat_col="latitude",
        da_lon_col="longitude",
        output_col=x)
selected_iea_projects.to_csv("./OUTPUTS/IEA_Project_Data.csv")