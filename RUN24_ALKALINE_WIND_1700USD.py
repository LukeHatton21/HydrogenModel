import numpy as np
import xarray as xr
import time
import dask
import csv
import os
#import cartopy.crs as ccrs
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import sys
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from joblib import Parallel, delayed
from electrolyser_v4 import Electrolyser
from economicmodel_v4 import Economic_Profile
from geodata_v3 import Global_Data
from filepreprocessor_v3 import All_Files
from datetime import datetime  
import os 
import regionmask
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'



class HydrogenModel:
    def __init__(self, dataset, params_file_econ=None, data_path=None, output_folder=None, elec_capex=None, lifetime=None, years=None, resolution=None, solar_fraction=None, elec_tech=None, stack_lifetime=None, water_price=None):
        
        
        """ Function to initialise the Hydrogen Model Class. Runs the LCOH on the basis of 1 MW of Renewables
        
        Inputs:
        Dataset: Contains the renewable profile, pre-processed for the input years, in an Xarray Dataset
        Lifetime: Number of operational years for the hydrogen plant
        Years: Years of renewable data used
        Params_File_Elec: Parameters to initialise the Electrolyser Class
        Params_File_Renew: Parameters to initialise the Economic Class with
        Data Path: Path to the Input Folder
        Output_Folder: Path to the Output Folder
        Solar_CF, Wind_CF: Data for the Solar and Wind Mean Capacity Factors
        Solar Fraction: Fraction of the Renewable Installation that is Solar
        Elec_Capex: Value for the Electrolyser CAPEX, in USD/kW
        Elec_Tech: Electrolyser Technology (Either PEM or Alkaline)
        Stack Lifetime: Lifetime of the Electrolyser Stack, in Years
        Water Cost: In USD/m^3"""
        
        
        # Initialise Electrolyser Class
        self.electrolyser_class = Electrolyser(elec_tech)

        # Initialise Economic Class
        self.economic_profile_class = self.parameters_from_csv(params_file_econ, 'economic')
        
        # Initialise Geodata Class and Initialise Geodata Class
        if resolution is not None:
            self.renewables_data = self.interpolate_data(dataset, resolution)
            self.geodata_class = Global_Data((data_path + "ETOPO_bathymetry.nc"),(data_path+"distance2shore.nc"), (data_path+"country_grids.nc"), self.renewables_data, resolution)
        else: 
            self.renewables_data = dataset
            self.geodata_class = Global_Data((data_path + "ETOPO_bathymetry.nc"),(data_path+"distance2shore.nc"), (data_path+"country_grids.nc"), dataset)
        self.geodata = self.geodata_class.get_all_data_variables()
        
        
        # Specify the input and output folders
        self.output_folder = output_folder
        self.country_wacc_mapping = pd.read_csv((data_path + "new_country_waccs.csv"))
        self.country_data = xr.open_dataset((data_path + "country_grids.nc"))
        
        
        # Set up other inputs
        self.stack_lifetime = stack_lifetime
        self.lifetime = lifetime
        self.years = years
        self.solar_fraction = solar_fraction
        self.economic_profile_class.solar_fraction = solar_fraction
        self.water_price = water_price
        
        # Set up Electrolyser Parameters
        self.electrolyser_capacity = self.economic_profile_class.electrolyser_capacity
        self.electrolyser_class.elec_capacity_array = xr.zeros_like(dataset) + self.electrolyser_capacity
        
        
        # Remove High Seas from dataset
        self.high_seas = self.remove_high_seas()

        #### MOVE BATTERY TO ECONOMIC MODEL ###
        self.unsmoothed_data = self.renewables_data
        self.renewables_data = self.economic_profile_class.battery_smoothing(self.unsmoothed_data)
        
        print("Hydrogen Model Class initialised")
        
        
    def remove_high_seas(self):
        
        nan_mask_sea = xr.where(np.isnan(self.geodata['sea']), True, False)
        nan_mask_land = xr.where(np.isnan(self.geodata['land']), True, False)
        combined_nan_mask = nan_mask_sea & nan_mask_land
        return combined_nan_mask
    
    
    
    def interpolate_data(self, dataset, resolution):
        latitudes = dataset.latitude.values
        latitudes_rounded = np.around(latitudes, decimals=1)
        longitudes = dataset.longitude.values
        longitudes_rounded = np.around(longitudes, decimals=1)
        new_latitudes = np.arange(latitudes_rounded[2], latitudes_rounded[-2]+resolution, resolution)
        new_longitudes = np.arange(longitudes_rounded[2], longitudes_rounded[-2]+resolution, resolution)
        interp_dataset = dataset.interp(latitude=new_latitudes, longitude=new_longitudes, method="linear")
        return interp_dataset
    
            
    def parameters_from_csv(self, file_path, class_name):
        try:
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # skip the header row
                params = {}
                for row in reader:
                    if len(row) == 2:
                        param_name = row[0].strip()
                        param_value = row[1].strip()
                        try:
                            param_value = float(param_value)
                            if param_value.is_integer():
                                param_value = int(param_value)
                        except ValueError:
                            pass
                        params[param_name] = param_value
            if class_name == 'economic':
                class_initiated = Economic_Profile(**params)
                self.renewables_capacity = class_initiated.renewables_capacity
            elif class_name == 'electrolyser':
                class_initiated = Electrolyser(**params)
            return class_initiated
        except ValueError as e:
            print("Error: {}".format(e))
        except TypeError as f:
            print("Error: {}".format(f))
    
    
    def cashflow_discount(self, data, rate):
        # Read number of years, latitudes and longitudes
        years = data.sizes['year']
        lat_num = data.sizes['latitude']
        lon_num = data.sizes['longitude']
    
        # Create array for storage
        discounted_data = xr.zeros_like(data)
        
        # Apply discounting using nested for loops
        for year in range(years):
                #print(f"{year:.0f} discounting complete")
                for lat in range(lat_num):
                    for lon in range(lon_num):
                        discounted_data[year, lat, lon] = data[year, lat, lon] / (
                                (1 + rate) ** year)
                        
        return discounted_data
    
    def country_wacc_discounts(self, data, electrolyser=None, solar_capex=None, wind_capex=None):
        # Read number of years, latitudes and longitudes
        years = data.sizes['year']
        latitudes = data.latitude.values
        longitudes = data.longitude.values
    
        # Create array for storage
        discounted_data = xr.zeros_like(data)
        
        for year in range(years):
            for count_lat, lat in enumerate(latitudes):
                for count_lon, lon in enumerate(longitudes):
                    
                    # Get renewables discount rate
                    rate = self.get_country_wacc(lat, lon, solar_capex, wind_capex)
                    
                    # Calculate electrolyser discount rate (renewables +5%)
                    if electrolyser is not None:
                        rate = rate + 0.05
                        
                    # Check if the rate exists and if not, replace with a default
                    if np.isnan(rate) or rate == 0:
                        default_rate = self.economic_profile_class.renew_discount_rate
                        
                        # Add 5% to the default rate for electrolyser
                        if electrolyser is not None:
                               default_rate = default_rate + 0.05
                                
                        # Discount the data with the default rate if applicable
                        discounted_data[year, count_lat, count_lon] = data[year, count_lat, count_lon] / (
                                (1 + default_rate) ** year)
                    else:
                        discounted_data[year, count_lat, count_lon] = data[year, count_lat, count_lon] / (
                                (1 + rate) ** year)
       
                        
        return discounted_data
    
    def hydrogen_wacc_discounts(self, data, elec_capex, ren_capex, solar_capex, wind_capex):
        # Read number of years, latitudes and longitudes
        years = data.sizes['year']
        latitudes = data.latitude.values
        longitudes = data.longitude.values
        
        # Create array for storage
        discounted_data = xr.zeros_like(data)
        
        for year in range(years):
            for count_lat, lat in enumerate(latitudes):
                for count_lon, lon in enumerate(longitudes):
                    ren_rate = self.get_country_wacc(lat, lon, solar_capex, wind_capex)
                    if np.isnan(ren_rate) or ren_rate == 0:
                        ren_rate =  self.economic_profile_class.renew_discount_rate
                    elec_rate = ren_rate + 0.05
                    composite_rate = (elec_capex*elec_rate + ren_capex*ren_rate) / (elec_capex + ren_capex)
                    discounted_data[year, count_lat, count_lon] = data[year, count_lat, count_lon] / (
                                (1 + composite_rate) ** year)      
       
                        
        return discounted_data
    
    def get_country_wacc(self, lat, lon, solar_capex, wind_capex):
    
        # Retrieve CSV file with mapping of countries and waccs
        country_wacc_mappings = self.country_wacc_mapping
        geodata = self.geodata
        land_value = geodata['land'].sel({'latitude': lat, 'longitude': lon}).values 
        
        
        # Retrieve offshore mask
        if np.isnan(land_value):
            sea_value = geodata['sea'].sel({'latitude': lat, 'longitude': lon}).values 
            if np.isnan(sea_value):
                country_wacc = np.nan
            else:
                country_row = country_wacc_mappings.loc[country_wacc_mappings['index'] == sea_value]
                country_wacc = country_row.loc[country_row.index[0],'offshore wacc']
                
        else: 
            country_row = country_wacc_mappings.loc[country_wacc_mappings['index'] == land_value]
            solar_rate = country_row.loc[country_row.index[0],'solar pv wacc']
            wind_rate = country_row.loc[country_row.index[0],'onshore wacc']
            composite_rate = (solar_capex*solar_rate + wind_capex*wind_rate) / (wind_capex + solar_capex)
            country_wacc = composite_rate
        
        
        return country_wacc
        
    def extend_to_lifetime(self, data, lifetime):
        
        years = data.sizes['year'] - 1
        n_duplications = round(lifetime/years)
        remainder = lifetime % years
        
        # Separate 0th year and operation years 
        zeroth_year = data[0:1,:,:]
        operational_years = data[1:, :, :]
        remainder_index = remainder+1
        remainder_years = data[1:remainder_index, :, :]
        
        # Duplicate a set amount of times and then concenate over operational years
        duplicated_data = xr.concat([operational_years] * n_duplications, dim ='year')
        first_year = duplicated_data['year'][0]
        final_year = duplicated_data['year'][-1]
        year_range = final_year - first_year + 1
        duplicated_with_r = xr.concat((duplicated_data, remainder_years), dim = 'year')
        new_year_range = np.arange(first_year, (first_year + n_duplications * year_range + remainder), step = 1)
        duplicated_data =  duplicated_with_r.assign_coords({'year': new_year_range})
        
        # Concenate with the 0th year
        combined_data = xr.concat((zeroth_year, duplicated_data), dim ='year')
        # Return data
        return combined_data
   

    
    def cost_function(self, capacity, renewables_gridpoint):
        aggregated_results = self.parallel_levelised_cost_v2(renewables_data=renewables_gridpoint, capacity=capacity)
        return aggregated_results['levelised_cost'].mean()
    
    def objective_function(self, capacity, renewables_gridpoint):
        aggregated_results = self.parallel_levelised_cost_v2(renewables_data=renewables_gridpoint, capacity=capacity)
        mean_lcoh = aggregated_results['levelised_cost'].values.mean()
        return np.round(mean_lcoh, decimals = 4)
     
    
    def global_parallel_calculation(self, num_cores, linear_running=None):
        start_time = time.time()
        latitudes = self.renewables_data.latitude.values
        longitudes = self.renewables_data.longitude.values
        
        # Create a list of arguments for each grid point
        grid_point_args = []
        for count_lat, lat in enumerate(latitudes):
            for count_lon, lon in enumerate(longitudes):
                grid_point_args.append((lat, lon))
    
        # Use joblib to parallelize the processing of grid points
        if linear_running is None:
            parallel_results = Parallel(n_jobs=num_cores, verbose=5)(delayed(self.optimise_grid_point)(lat=lat, lon=lon) for lat, lon in grid_point_args)
        else: 
            parallel_results = Parallel(n_jobs=num_cores, verbose=5)(delayed(self.levelised_cost_grid_point)(lat=lat, lon=lon) for lat, lon in grid_point_args)
        
        # Extract results
        print("Extracting results")
        levelised_costs = []
        levelised_costs_ren = []
        levelised_costs_elec = []         
        electrolyser_capacities = []
        hydrogen_production = []
        loop_times = []
        capital_costs = []
        configurations = []
        renew_costs = []
        solar_costs =[]
        wind_costs =[]
        elec_costs = []
        
        # Loop over results to extract by individual longitude and latitude
        for i, result in enumerate(parallel_results):
            result_ds, loop_time = result
            levelised_costs.append(result_ds['levelised_cost'])
            levelised_costs_ren.append(result_ds['levelised_cost_ren'])
            levelised_costs_elec.append(result_ds['levelised_cost_elec'])
            electrolyser_capacities.append(result_ds['electrolyser_capacity'])
            hydrogen_production.append(result_ds['hydrogen_production'])
            loop_times.append(loop_time)
            capital_costs.append(result_ds['total_capital_costs'])
            configurations.append(result_ds['configuration'])
            renew_costs.append(result_ds['renewables_costs'])
            solar_costs.append(result_ds['solar_costs'])
            wind_costs.append(result_ds['wind_costs'])
            elec_costs.append(result_ds['electrolyser_costs'])
        
        # Reshape results and store in an xarray dataset
        print("Extracting results and reshaping")
        loop_times_array = np.array(loop_times)   
        levelised_costs_array = np.reshape(levelised_costs, (len(latitudes), len(longitudes)), order='C')
        levelised_costs_ren = np.reshape(levelised_costs_ren, (len(latitudes), len(longitudes)), order='C')
        levelised_costs_elec = np.reshape(levelised_costs_elec, (len(latitudes), len(longitudes)), order='C')
        electrolyser_capacity_array = np.reshape(electrolyser_capacities, (len(latitudes), len(longitudes)), order='C')
        hydrogen_production_array = np.reshape(hydrogen_production, (len(latitudes), len(longitudes)), order='C')
        capital_costs_array = np.reshape(capital_costs, (len(latitudes), len(longitudes)), order='C')
        configuration_array = np.reshape(configurations, (len(latitudes), len(longitudes)), order='C')
        renew_costs_array = np.reshape(renew_costs, (len(latitudes), len(longitudes)), order='C')
        solar_costs_array = np.reshape(solar_costs, (len(latitudes), len(longitudes)), order='C')
        wind_costs_array = np.reshape(wind_costs, (len(latitudes), len(longitudes)), order='C')
        elec_costs_array = np.reshape(elec_costs, (len(latitudes), len(longitudes)), order='C')
        
        combined_results = xr.Dataset({'levelised_cost': (['latitude', 'longitude'], levelised_costs_array),
                                       'levelised_cost_ren': (['latitude', 'longitude'], levelised_costs_ren),
                                        'levelised_cost_elec': (['latitude', 'longitude'], levelised_costs_elec),
                                       'electrolyser_capacity': (['latitude', 'longitude'], electrolyser_capacity_array ),
                                       'total_capital_costs': (['latitude', 'longitude'], capital_costs_array ),
                                       'renewables_costs': (['latitude', 'longitude'], renew_costs_array),
                                       'solar_costs': (['latitude', 'longitude'], solar_costs_array),
                                       'wind_costs': (['latitude', 'longitude'], wind_costs_array),
                                       'electrolyser_costs': (['latitude', 'longitude'], elec_costs_array), 
                                       'configuration': (['latitude', 'longitude'], configuration_array ),
                                       'hydrogen_production': (['latitude', 'longitude'], hydrogen_production_array),},
                                      coords={'latitude': latitudes,'longitude': longitudes})
        
        
        print("Results extracted")
        
        end_time = time.time()
        total_time = end_time - start_time
        loop_time = np.nanmean(loop_times_array)
        
        if linear_running is None:
            operation = "Optimisation"
        else: 
            operation = "Calculation"
        print(f"{operation} took an average of {loop_time:.2f} seconds to run for each grid point")
        print(f"{operation} took {total_time:.2f} seconds to run for all grid points")
        return combined_results
    
    
    
    
    def optimise_grid_point(self, lat, lon):
        
        loop_start = time.time()
        
        # Get renewables data at each gridpoint
        renewables_gridpoint = self.renewables_data.sel(longitude = lon, latitude=lat)
        renewables_gridpoint = renewables_gridpoint.expand_dims(latitude=[lat], longitude=[lon])
        renewables_gridpoint = renewables_gridpoint.transpose("time", "latitude", "longitude")
                    
        # Evaluate nature of gridpoint
        offshore_status = self.geodata['offshore'].sel(longitude = lon, latitude=lat)   
        
        # Evaluate nature of gridpoint
        high_seas_status = self.high_seas.sel(longitude=lon, latitude=lat)
        landmask = self.geodata['land'].sel(longitude=lon, latitude=lat)
        
        # Check if location is high sea
        if high_seas_status == True:
            da = xr.DataArray(np.array([[np.nan]]), coords={'latitude': [lat], 'longitude': [lon]}, dims={'latitude', 'longitude'})
            
            data_vars = {'levelised_cost': da,
                        'levelised_cost_ren': da,
                    'levelised_cost_elec': da,
                     'hydrogen_production': da,
                     'electrolyser_capacity': da,
                     'total_capital_costs': da,
                     'configuration': da,
                     'renewables_costs': da, 
                     'solar_costs': da,
                     'wind_costs': da,
                     'electrolyser_costs': da}
            coords = {'latitude': lat,
                  'longitude': lon}
            high_seas_results = xr.Dataset(data_vars=data_vars, coords=coords)
            return high_seas_results, np.nan
        
        # If solar is being examined, check if the location is in the sea
        if self.solar_fraction > 0:
            if np.isnan(landmask) == True:
                da = xr.DataArray(np.array([[np.nan]]), coords={'latitude': [lat], 'longitude': [lon]}, dims={'latitude', 'longitude'})
            
                data_vars = {'levelised_cost': da,
                        'levelised_cost_ren': da,
                    'levelised_cost_elec': da,
                     'hydrogen_production': da,
                     'electrolyser_capacity': da,
                     'total_capital_costs': da,
                     'configuration': da,
                     'renewables_costs': da, 
                     'solar_costs': da,
                     'wind_costs': da,
                     'electrolyser_costs': da}
                coords = {'latitude': lat,
                  'longitude': lon}
                seas_results = xr.Dataset(data_vars=data_vars, coords=coords)
                
                return seas_results, np.nan
        
        # Calculate Hydrogen Output
        renewables_profile = renewables_gridpoint * self.renewables_capacity
        if self.geodata['offshore'].sel(longitude=lon, latitude=lat) == 1:
            electrolyser_yearly_output = self.electrolyser_class.calculate_yearly_output(renewables_profile, self.electrolyser_capacity, offshore=True)
        else:
            electrolyser_yearly_output = self.electrolyser_class.calculate_yearly_output(renewables_profile, self.electrolyser_capacity)
        hydrogen_yearly_output = electrolyser_yearly_output['hydrogen_produced']
        
        if hydrogen_yearly_output[1] == 0:
            da = xr.DataArray(np.array([[np.nan]]), coords={'latitude': [lat], 'longitude': [lon]}, dims={'latitude', 'longitude'})
            
            data_vars = {'levelised_cost': da,
                        'levelised_cost_ren': da,
                    'levelised_cost_elec': da,
                     'hydrogen_production': da,
                     'electrolyser_capacity': da,
                     'total_capital_costs': da,
                     'configuration': da,
                     'renewables_costs': da, 
                     'solar_costs': da,
                     'wind_costs': da,
                     'electrolyser_costs': da}
            coords = {'latitude': lat,
                  'longitude': lon}
            zero_ouput_results = xr.Dataset(data_vars=data_vars, coords=coords)
            return zero_ouput_results, np.nan   
        
                    
        # Set up optimisation problem
        #initial_guess = [self.electrolyser_capacity]
        
        #if self.solar is not None:
            #mean_cf = self.solar_cf['CF'].sel(latitude=lat, longitude=lon).values
            #initial_guess = [(mean_cf * 33800) + 6550]
        #else:
            #mean_cf = self.wind_cf['CF'].sel(latitude=lat, longitude=lon).values
            #initial_guess = [(mean_cf * 15600) + 8780]
        #if initial_guess[0] > 18000:
            #initial_guess[0] = 18000
        low_bound = 0 * self.renewables_capacity
        upp_bound = 1.0 * self.renewables_capacity
        bounds = [(low_bound, upp_bound)]
            
                    
        # Use SciPy's Minimisation Function
        #result = basinhopping(self.cost_function, initial_guess, minimizer_kwargs={"args": (renewables_gridpoint,), "bounds": bounds}, stepsize=500, niter=250)
        result = minimize(self.objective_function, x0=0.25*self.renewables_capacity, args= (renewables_gridpoint,), bounds=bounds, method='powell')
                    
        # Store electrolyser capacity at that location
        optimal_electrolyser_capacity = result.x[0]
        optimal_lcoh = result.fun
        electrolyser_capacity = optimal_electrolyser_capacity 
                    
        # Store lowest achievable LCOH
        levelised_cost = optimal_lcoh
                    
        # Store hydrogen production
        aggregated_results = self.parallel_levelised_cost_v2(renewables_gridpoint, electrolyser_capacity)
        
        loop_end = time.time()
        loop_time = loop_end - loop_start
        return aggregated_results, loop_time

    
    
    def levelised_cost_grid_point(self, lat, lon):
        
        loop_start = time.time()
        
        
        # Evaluate nature of gridpoint
        high_seas_status = self.high_seas.sel(longitude=lon, latitude=lat)
        landmask = self.geodata['land'].sel(longitude=lon, latitude=lat)

        # Check if location is sea
        if high_seas_status == True:
            da = xr.DataArray(np.array([[np.nan]]), coords={'latitude': [lat], 'longitude': [lon]}, dims={'latitude', 'longitude'})
            
            data_vars = {'levelised_cost': da,
                        'levelised_cost_ren': da,
                    'levelised_cost_elec': da,
                     'hydrogen_production': da,
                     'electrolyser_capacity': da,
                     'total_capital_costs': da,
                     'configuration': da,
                     'renewables_costs': da, 
                     'solar_costs': da,
                     'wind_costs': da,
                     'electrolyser_costs': da}
            coords = {'latitude': lat,
                  'longitude': lon}
            high_seas_results = xr.Dataset(data_vars=data_vars, coords=coords)
            return high_seas_results, np.nan
        
        # If solar is being examined, check if the location is in the sea
        if self.solar_fraction > 0:
            if np.isnan(landmask) == True:
                da = xr.DataArray(np.array([[np.nan]]), coords={'latitude': [lat], 'longitude': [lon]}, dims={'latitude', 'longitude'})
            
                data_vars = {'levelised_cost': da,
                        'levelised_cost_ren': da,
                     'levelised_cost_elec': da,
                     'hydrogen_production': da,
                     'electrolyser_capacity': da,
                     'total_capital_costs': da,
                     'configuration': da,
                     'renewables_costs': da,
                     'solar_costs': da,
                     'wind_costs': da,
                     'electrolyser_costs': da}
                coords = {'latitude': lat,
                  'longitude': lon}
                seas_results = xr.Dataset(data_vars=data_vars, coords=coords)
                return seas_results, np.nan
        
        # Get renewables data at each gridpoint
        renewables_gridpoint = self.renewables_data.sel(longitude=lon, latitude=lat)
        renewables_gridpoint = renewables_gridpoint.expand_dims(latitude=[lat], longitude=[lon])
        renewables_gridpoint = renewables_gridpoint.transpose("time", "latitude", "longitude")
          
         # Calculate Hydrogen Output
        renewables_profile = renewables_gridpoint * self.renewables_capacity
        if self.geodata['offshore'].sel(longitude=lon, latitude=lat) == 1:
            electrolyser_yearly_output = self.electrolyser_class.calculate_yearly_output(renewables_profile, self.electrolyser_capacity, offshore=True)
        else:
            electrolyser_yearly_output = self.electrolyser_class.calculate_yearly_output(renewables_profile, self.electrolyser_capacity)
        hydrogen_yearly_output = electrolyser_yearly_output['hydrogen_produced']
        
        if hydrogen_yearly_output[1] == 0:
            da = xr.DataArray(np.array([[np.nan]]), coords={'latitude': [lat], 'longitude': [lon]}, dims={'latitude', 'longitude'})
            
            data_vars = {'levelised_cost': da,
                         'levelised_cost_ren': da,
                     'levelised_cost_elec': da,
                     'hydrogen_production': da,
                     'electrolyser_capacity': da,
                     'total_capital_costs': da,
                     'configuration': da,
                     'renewables_costs': da,
                     'solar_costs': da,
                     'wind_costs': da,
                     'electrolyser_costs': da}
            coords = {'latitude': lat,
                  'longitude': lon}
            zero_ouput_results = xr.Dataset(data_vars=data_vars, coords=coords)
            return zero_ouput_results, np.nan  


        
        # Get geodata
        geodata = self.geodata.sel(longitude=lon, latitude=lat)
        # Store hydrogen production
        aggregated_results = self.parallel_levelised_cost_v2(renewables_gridpoint, self.electrolyser_capacity)
        
        
        loop_end = time.time()
        loop_time = loop_end - loop_start
        return aggregated_results, loop_time
    
    
    def parallel_levelised_cost(self, renewables_data, capacity, elec_print=None):
        
        # Extract latitudes and longitudes
        latitude = renewables_data.latitude.values
        longitude = renewables_data.longitude.values
        
        # Setup variables stored in the Hydrogen Model Class
        lifetime = self.lifetime
        geodata = self.geodata.sel(latitude=latitude, longitude=longitude)
        elec_capacity = int(capacity)
       
        
        # Call the Renewables Profile and Electrolyser Classes
        renewables_profile = renewables_data * self.renewables_capacity
        electrolyser_yearly_output = self.electrolyser_class.calculate_yearly_output(renewables_profile, elec_capacity)
        combined_yearly_output = self.economic_profile_class.calculate_fractional_costs(renewables_profile, geodata, self.solar_fraction, elec_capacity)
        
        
        # Access relevant yearly variables for the LCOH calculation
        hydrogen_produced_yearly = electrolyser_yearly_output['hydrogen_produced']
        electrolyser_costs_yearly = combined_yearly_output['electrolyser costs']
        renewables_costs_yearly = combined_yearly_output['renewable costs']
        
        # Extract wind and solar costs
        solar_costs_yearly = combined_yearly_output['solar costs']
        wind_costs_yearly = combined_yearly_output['wind costs']
        
        # If annual hydrogen output is zero, replace with a value very close to zero
        if hydrogen_produced_yearly[1] == 0:
            hydrogen_produced_yearly[1] = 0.0001
        
        # Extract required variables
        total_capital_cost = combined_yearly_output['total costs'][0, :, :]
        renewables_cost = renewables_costs_yearly[0, :, :]
        solar_cost = solar_costs_yearly[0, :, :]
        wind_cost = wind_costs_yearly[0, :, :]
        electrolyser_cost = electrolyser_costs_yearly[0, :, :]
        configuration = combined_yearly_output['configuration']
        latitudes = combined_yearly_output.latitude.values
        longitudes = combined_yearly_output.longitude.values
        
        # Read the dimensions of the yearly output
        years = electrolyser_yearly_output.sizes['year'] - 1
        
        # If the size of the data is less than the lifetime, duplicate the data
        if years < lifetime:
            n_duplicates = round(lifetime / years)
            hydrogen_produced_yearly = self.extend_to_lifetime(hydrogen_produced_yearly, lifetime)
            electrolyser_costs_yearly = self.extend_to_lifetime(electrolyser_costs_yearly, lifetime)
            renewables_costs_yearly = self.extend_to_lifetime(renewables_costs_yearly, lifetime)
        
        # Add in the cost of desalination
        desalinated_water_costs = 0.002 * hydrogen_produced_yearly
        electrolyser_costs_yearly = electrolyser_costs_yearly + desalinated_water_costs
        
        # Add in the cost of stack replacement
        electrolyser_costs_yearly[self.stack_lifetime+1, :, :] = self.economic_profile_class.elec_capex * 0.41 * elec_capacity 
        
        # Discount renewables and electrolyser costs separately # CHANGE THIS
        discounted_renew_costs = self.country_wacc_discounts(renewables_costs_yearly, solar_capex = int(solar_cost), wind_capex=int(wind_cost))
        discounted_elec_costs = self.country_wacc_discounts(electrolyser_costs_yearly, 1, solar_capex = int(solar_cost), wind_capex=int(wind_cost))
        discounted_output = self.hydrogen_wacc_discounts(hydrogen_produced_yearly, int(electrolyser_cost), int(renewables_cost), solar_capex = int(solar_cost), wind_capex=int(wind_cost))
        discounted_costs = discounted_renew_costs + discounted_elec_costs
            
            
        # Sum the discounted costs and hydrogen produced
        discounted_costs_sum = discounted_costs.sum(dim='year')
        hydrogen_produced_sum = discounted_output.sum(dim='year')
        
        # Calculate the average annual hydrogen production in tonnes per annum
        annual_hydrogen = hydrogen_produced_yearly.mean(dim='year') / 1000
        
        
        # Calculate the levelised costs, filtering to account for the locations that are too far from the shoreline
        levelised_cost_raw = np.divide(discounted_costs_sum, hydrogen_produced_sum)
        levelised_cost = xr.where(levelised_cost_raw == 0, np.nan, levelised_cost_raw)
        
        # Create dataset with results
        data_vars = {'levelised_cost': levelised_cost,
                     'hydrogen_production': annual_hydrogen / (self.renewables_capacity / 1000),
                     'electrolyser_capacity': xr.full_like(configuration, elec_capacity/self.renewables_capacity),
                     'total_capital_costs': total_capital_cost/(self.renewables_capacity / 1000),
                     'configuration': configuration,
                     'renewables_costs': renewables_cost/(self.renewables_capacity / 1000),
                     'solar_costs': solar_cost/(self.renewables_capacity / 1000),
                     'wind_costs': wind_cost/(self.renewables_capacity / 1000),
                     'electrolyser_costs': electrolyser_cost/(self.renewables_capacity / 1000)}
        coords = {'latitude': latitudes,
                  'longitude': longitudes}
        aggregated_results = xr.Dataset(data_vars=data_vars, coords=coords)
        
        return aggregated_results

    
    
    def parallel_levelised_cost_v2(self, renewables_data, capacity):
        
        # Extract latitudes and longitudes
        latitude = renewables_data.latitude.values
        longitude = renewables_data.longitude.values
        
        # Extract geodata and set values for the lifetime and electrolyser capacity
        lifetime = self.lifetime
        geodata = self.geodata.sel(latitude=latitude, longitude=longitude)
        elec_capacity = int(capacity)
       
        
        # Call the Renewables Profile and Electrolyser Classes
        renewables_profile = renewables_data * self.renewables_capacity
        if geodata['offshore'] == 1:
            electrolyser_yearly_output = self.electrolyser_class.calculate_yearly_output(renewables_profile, elec_capacity, offshore=True)
        else:
            electrolyser_yearly_output = self.electrolyser_class.calculate_yearly_output(renewables_profile, elec_capacity)
        combined_yearly_output = self.economic_profile_class.calculate_fractional_costs_v2(renewables_profile, geodata, self.solar_fraction, elec_capacity)
                
        # Access relevant yearly variables for the LCOH calculation
        hydrogen_produced_yearly = electrolyser_yearly_output['hydrogen_produced']
        electrolyser_costs_yearly = combined_yearly_output['electrolyser costs']
        renewables_costs_yearly = combined_yearly_output['renewable costs']
        
        
        # Extract wind and solar costs
        solar_costs_yearly = combined_yearly_output['solar costs']
        wind_costs_yearly = combined_yearly_output['wind costs']
        other_costs_yearly = combined_yearly_output['other costs']
        
        # If annual hydrogen output is zero, replace with a value very close to zero
        if hydrogen_produced_yearly[1] == 0:
            hydrogen_produced_yearly[1] = 0.0001
        
        
        # Extract required variables
        total_capital_cost = combined_yearly_output['total costs'][0, :, :]
        renewables_cost = renewables_costs_yearly[0, :, :]
        solar_cost = solar_costs_yearly[0, :, :]
        wind_cost = wind_costs_yearly[0, :, :]
        electrolyser_cost = electrolyser_costs_yearly[0, :, :]
        configuration = combined_yearly_output['configuration']
        latitudes = combined_yearly_output.latitude.values
        longitudes = combined_yearly_output.longitude.values
        
        # Read the dimensions of the yearly output
        years = electrolyser_yearly_output.sizes['year'] - 1
        
        # If the size of the data is less than the lifetime, duplicate the data
        if years < lifetime:
            n_duplicates = round(lifetime / years)
            hydrogen_produced_yearly = self.extend_to_lifetime(hydrogen_produced_yearly, lifetime)
            electrolyser_costs_yearly = self.extend_to_lifetime(electrolyser_costs_yearly, lifetime)
            renewables_costs_yearly = self.extend_to_lifetime(renewables_costs_yearly, lifetime)
        

        # Add in Additional Costs - desalination etc
        water_costs = self.water_price * hydrogen_produced_yearly * 25
        desalination_costs = xr.where(geodata['offshore'] == 1, 0.491*self.renewables_capacity, 0)
        other_costs_yearly[0, :, :] = other_costs_yearly[0, :, :] + desalination_costs
        other_costs = water_costs + other_costs_yearly
        
        # Add in the cost of stack replacement 
        electrolyser_costs_yearly[self.stack_lifetime+1, :, :] = self.economic_profile_class.elec_capex * self.economic_profile_class.elec_stack_replacement * elec_capacity 
        
        # Discount renewables and electrolyser costs separately
        discounted_renew_costs = self.country_wacc_discounts(renewables_costs_yearly, solar_capex = int(solar_cost), wind_capex=int(wind_cost))
        discounted_elec_costs = self.country_wacc_discounts(electrolyser_costs_yearly, electrolyser="True", solar_capex = int(solar_cost), wind_capex=int(wind_cost))
        discounted_other_costs = self.country_wacc_discounts(other_costs_yearly, solar_capex = int(solar_cost), wind_capex=int(wind_cost))
        discounted_output = self.hydrogen_wacc_discounts(hydrogen_produced_yearly, int(electrolyser_cost), int(renewables_cost), solar_capex = int(solar_cost), wind_capex=int(wind_cost))
        discounted_costs = discounted_renew_costs + discounted_elec_costs + discounted_other_costs
            
            
        # Sum the discounted costs and hydrogen produced
        discounted_costs_sum = discounted_costs.sum(dim='year')
        hydrogen_produced_sum = discounted_output.sum(dim='year')
        discounted_elec_costs_sum = discounted_elec_costs.sum(dim='year')
        discounted_renew_costs_sum = discounted_renew_costs.sum(dim='year')
        
        # Calculate the average annual hydrogen production in tonnes per annum
        annual_hydrogen = hydrogen_produced_yearly.mean(dim='year') / 1000
        
        
        # Calculate the levelised costs, filtering to account for the locations that are too far from the shoreline
        levelised_cost_raw = np.divide(discounted_costs_sum, hydrogen_produced_sum)
        levelised_cost = xr.where(levelised_cost_raw == 0, np.nan, levelised_cost_raw)
        renewables_lcoh = np.divide(discounted_renew_costs_sum, hydrogen_produced_sum)
        electrolyser_lcoh = np.divide(discounted_elec_costs_sum, hydrogen_produced_sum)
        
        # Create dataset with results
        data_vars = {'levelised_cost': levelised_cost,
                     'levelised_cost_ren': renewables_lcoh,
                     'levelised_cost_elec': electrolyser_lcoh,
                     'hydrogen_production': annual_hydrogen / (self.renewables_capacity / 1000),
                     'electrolyser_capacity': xr.full_like(configuration, elec_capacity/self.renewables_capacity),
                     'total_capital_costs': total_capital_cost/(self.renewables_capacity / 1000),
                     'configuration': configuration,
                     'renewables_costs': renewables_cost/(self.renewables_capacity / 1000),
                     'solar_costs': solar_cost/(self.renewables_capacity / 1000),
                     'wind_costs': wind_cost/(self.renewables_capacity / 1000),
                     'electrolyser_costs': electrolyser_cost/(self.renewables_capacity / 1000)}
        coords = {'latitude': latitudes,
                  'longitude': longitudes}
        aggregated_results = xr.Dataset(data_vars=data_vars, coords=coords)
        
        
        return aggregated_results
    
    
    
    
    def save_results(self, output_folder, results, filename=None):

        # Get timestamp
        time_stamp = time.time()
        date_time = datetime.fromtimestamp(time_stamp)
        str_date_time = date_time.strftime("%d-%m-%Y-%H")
        start_year = years[0]
        end_year = years[-1]
        years_str = str(start_year) + '_' + str(end_year)

        # Output the file
        if filename is None:
            filename = 'unspecified_results_' 
        results.to_netcdf(output_folder + filename + '.nc')
        
    def save_specified_results(self, output_folder, lat_lon, results = None, renew_tech=None, solar_fraction=None, elec_tech=None, elec_capex=None, optimisation=None, return_filename=None):
        
        if renew_tech is None:
            filename_str = elec_tech + "_V4" + "_Solar" + f"{solar_fraction * 100:.0f}" + "%_" + str(elec_capex) + "USD_" + str(lat_lon[0]) + '_' + str(lat_lon[1]) + '_' + str(lat_lon[2]) + '_' + str(lat_lon[3]) 
        else: 
            filename_str = elec_tech + "_V4" + "_" + renew_tech + "_" + str(elec_capex) + "USD_" + str(lat_lon[0]) + '_' + str(lat_lon[1]) + '_' + str(lat_lon[2]) + '_' + str(lat_lon[3]) 
        
        if optimisation =="True":
            filename_str = filename_str + "_optimised"
            
        # Return just the string if required
        if return_filename is not None:
            filename = filename_str + ".nc"
            
            return filename
        
        # Output the file
        results.to_netcdf(output_folder + filename_str + '.nc')
    
    



#### Changes to be introduced in this version

# Set up folder paths
shared_folder_path = os.path.abspath(os.path.join(os. getcwd(), os.pardir)) + "/"
renewable_profiles_path = shared_folder_path + "/MERRA2_INPUTS/"
solar_path = renewable_profiles_path + "Solar_CF/"
wind_path = renewable_profiles_path + "Wind_CF/"
input_data_path = shared_folder_path + "/DATA/"
output_folder = shared_folder_path + "/V3_OUTPUT_FOLDER/"
    
# Record start time
start_time = time.time()

# Set the conditions for the model run
fract_diff = 0.1 # Set to the difference in solar fraction being run 
elec_tech = "Alkaline" # Set to either PEM or Alkaline
optimisation = "True" # Set to "True" (included only for debugging)
elec_capex = 1700 # Set to specified electrolyser CAPEX (1200 USD/kW for Alkaline, 1400 USD/kW for PEM)
num_cores = 28 # Set to number of cores to use for each Hydrogen Model case
    
# Record start time
start_time = time.time()


### Set Latitude and Longitude ###

# Set up for loop for each individual slice of 20 latitudes
for i in [13, 12, 11, 6, 5]:
    lat_lon=[-90, 90, -180+20*i, -160+20*i] 
    solar_fraction = 0
    # Specify renew_tech for solely solar and wind cases
    if solar_fraction == 0:
        renew_tech = "Wind" 
    elif solar_fraction ==1: 
        renew_tech = "Solar"
    else:
        renew_tech = None


    if renew_tech is None:
        filename_str = elec_tech + "_V4" + "_Solar" + f"{solar_fraction * 100:.0f}" + "%_" + str(elec_capex) + "USD_" + str(lat_lon[0]) + '_' + str(lat_lon[1]) + '_' + str(lat_lon[2]) + '_' + str(lat_lon[3])
    else: 
        filename_str = elec_tech + "_V4" + "_" + renew_tech + "_" + str(elec_capex) + "USD_" + str(lat_lon[0]) + '_' + str(lat_lon[1]) + '_' + str(lat_lon[2]) + '_' + str(lat_lon[3]) 

    if optimisation == "True":
        filename_str = filename_str + "_optimised"
    filename_to_check = filename_str + ".nc"
    file_exists = os.path.isfile(os.path.join(output_folder, filename_to_check))       
    # Exit if the filename already exists - if not then generate the HydrogenModel object and continue with running the code
    if file_exists:
        print(f"The file {filename_to_check} already exists in the output folder")
    else:
        ## Set up files class
        all_files_class = All_Files(lat_lon=lat_lon, solar_path = solar_path, wind_path=wind_path, solar_format = "SOLAR_CF.", wind_format= "WIND_CF.")

        ## Preprocess the files 
        solar_data, wind_data, years = all_files_class.preprocess_combine_yearly()
        solar_profile_array = solar_data['Solar']
        wind_profile_array = wind_data['CF']
        renewable_profile_array = solar_fraction*solar_profile_array + (1-solar_fraction)*wind_profile_array
        print("Files from Renewables Ninja read in, corrected and combined")


        # Initialise an HydrogenModel object
        model = HydrogenModel(dataset=renewable_profile_array, lifetime = 20, years=years, params_file_econ=(input_data_path + "economic_parameters.csv"), data_path = input_data_path, output_folder=output_folder, elec_tech=elec_tech, stack_lifetime=10, solar_fraction = solar_fraction, elec_capex = elec_capex, water_price = 0.002)


        # Calculate the levelised cost
        try:
            if optimisation == "True":
                print("Starting LCOH Optimisation")
                combined_results = model.global_parallel_calculation(num_cores)
                print("Optimisation Finished")
            else:
                print("Starting LCOH Calculation")
                combined_results = model.global_parallel_calculation(num_cores, "False")
                print("Calculation Finished")

            model.save_specified_results(output_folder = output_folder, results = combined_results, lat_lon = lat_lon,  renew_tech = renew_tech, solar_fraction = solar_fraction, elec_tech = elec_tech, elec_capex = elec_capex, optimisation = optimisation)
            print("Results Saved")

        # Catch the error if there is one and store in the output folder as a .txt file
        except Exception as e:
        # Handle the error here
            print("Error when calling the parallel LCOH calculator")
            error_message = str(e)

            # Capture the console output
            with open(output_folder + 'error_log_' + str(lat_lon[0]) + '_' + str(lat_lon[1]) + '_' + str(lat_lon[2]) + '_' + str(lat_lon[3]) + '.txt', 'w') as f:
                    f.write("Error Message:\n" + error_message + '\n')








