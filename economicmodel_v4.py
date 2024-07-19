import numpy as np
import xarray as xr
import glob
import pandas as pd
import pvlib
import math
import matplotlib.pyplot as plt
#import cartopy.crs as ccrs



class Economic_Profile:
    def __init__(self, renewables_capacity=None, solar_op_cost=None, onshore_op_cost=None, offshore_op_cost=None, wind_capex=None, solar_capex=None, renewables_data = None, renew_discount_rate=None, lifetime=None, land_foundations = None, geodata = None, electrolyser_capacity = None, turbine_rating=None, turbine_diameter=None, elec_capex=None, elec_op_cost=None, battery_cost=None, battery_duration=None, battery_power_relative=None, elec_stack_replacement=None):
       
       
        """
        Function to initialise the Economic Profile Class, which estimates the costs for the renewable and electrolyser portion of the plant. 
        
        Inputs: 
        Renewables_Capacity: Installed Capacity of Renewables (Solar AND Wind), in kW
        Solar_Fraction: Proportion of the renewables capacity made up of solar
        Solar Operating Cost: Assumption for Solar OPEX, as a % of total CAPEX per year
        Onshore Operating Cost: Assumption for Onshore Wind OPEX, as a % of total CAPEX per year
        Offshore Operating Cost: Assumption for Offshore OPEX, as a % of total CAPEX per year
        Wind Capex: Cost of a Wind Turbine, in USD/kW
        Solar Capex: Cost of a Solar PV plant, in USD/kW
        Renewable_Discount_Rate: Assumption for discount rate, where not available
        Renewables_Data: Hourly Capacity Factor data for renewables
        Lifetime: Lifetime of the Plant, in Years
        Land_Foundations: Capital Cost for Land-Based Wind Foundations
        Geodata: Geodata on depth, distance to shore and land/sea status
        Electrolyser Capacity: Capacity of the Electrolyser, in kW
        Turbine Rating: Power Rating of Modelled Turbine, in kW
        Turbine Diameter: Diameter of Modelled Turbine, in m
        Battery Cost: Cost of the Battery, in USD/kW
        Battery Duration: Length of time the battery can operate at full output
        Battery_Size_Relative: Size of Battery, relative to power output of renewables
        """
       

        
        # Read in Financial Assumptions
        self.solar_capex = solar_capex
        self.wind_capex = wind_capex
        self.elec_capex = elec_capex
        self.solar_op_cost = solar_op_cost
        self.onshore_op_cost = onshore_op_cost
        self.offshore_op_cost = offshore_op_cost
        self.elec_op_cost = elec_op_cost
        self.renew_discount_rate = renew_discount_rate
        self.land_foundations = land_foundations
        self.lifetime = lifetime
        self.battery_cost = battery_cost
        self.battery_duration = battery_duration
        self.battery_power_relative = battery_power_relative
        self.battery_size = self.battery_power_relative*self.battery_duration
        self.elec_stack_replacement = elec_stack_replacement
        # Read in General Assumptions
        self.electrolyser_capacity = electrolyser_capacity
        self.renewables_capacity = renewables_capacity
        
        # Read in Technical Assumption
        self.turbine_diameter = turbine_diameter
        self.turbine_rating = turbine_rating
        
        # Read in Input Parameters
        self.geodata = geodata
        
        # Initialise solar_fraction
        self.solar_fraction = 0
            

    def battery_smoothing(self, renewables_data):
    
        def moving_average_xr(data_array, window_size, dim):
            # Create a rolling window object
            rolling_window = data_array.rolling({dim: window_size}, center=True)

            # Calculate the mean within the rolling window
            moving_avg = rolling_window.mean()
            window_padding = math.floor(window_size/2)

            # Set the first and last two values to the original values
            moving_avg.values[:window_padding] = data_array.values[:window_padding]
            moving_avg.values[-1*window_padding:] = data_array.values[-1*window_padding:]

            return moving_avg
        
        # Take the moving average
        profile = renewables_data
        moving_average_profile = moving_average_xr(profile, self.battery_duration, dim='time')
        
        
        # Work out the difference
        difference = moving_average_profile - profile

        # Calculate the cumulative sum
        cumulative_diff = difference.cumsum(dim='time')


        # Set bounds on the cumulative sum
        cumulative_diff_ub = xr.where(cumulative_diff >=  self.battery_size, self.battery_size, cumulative_diff)
        cumulative_diff_bounded = xr.where(cumulative_diff_ub <= 0, 0, cumulative_diff)
        
        # Unravel the cumulative sum
        reversed_data = cumulative_diff_bounded.diff('time')
        appended_reversed_data = xr.concat([cumulative_diff_bounded.sel(time=cumulative_diff_bounded['time'][0])*0, reversed_data], dim='time')
        appended_data = appended_reversed_data.transpose('time',  'latitude','longitude',)
        unraveled_data = xr.DataArray(appended_data, coords=renewables_data.coords, dims=renewables_data.dims)
        moving_average_constrained = profile + unraveled_data
        
        ma_renewables_data = xr.DataArray(moving_average_constrained, coords=renewables_data.coords, dims=renewables_data.dims)
        
        return ma_renewables_data
    
    

    def get_foundation_cost(self, geodata):
        """ Method to calculate the foundation cost for the wind turbine based on the depth of water, 
        using relationships from Bosch et al 2019 """
        
        
        depth_data = geodata['depth']
        
        # Set up relationships with depth
        a_parameter = [201, 114.24, 0]
        b_parameter = [612.93, -2270, 773.85]
        c_parameter = [411464, 531738, 680651]
        cutoff_data = [0, 25, 55, 1000]

        # initialise an empty array
        foundation_costs = xr.zeros_like(depth_data)

        # Use relationships with depth to estimate the foundation costs
        for i in range(len(cutoff_data) - 1):
            a = a_parameter[i]
            b = b_parameter[i]
            c = c_parameter[i]
            cutoff_start = cutoff_data[i]
            cutoff_end = cutoff_data[i + 1]

            # Apply cost relationship where depth is greater than the cutoff depth and not NaN
            foundation_costs = xr.where((depth_data > cutoff_start) & (depth_data <= cutoff_end), a * depth_data ** 2 + b * depth_data + c, foundation_costs)
        
        # Apply relationships for onshore (set foundation cost to input) and offshore above the cutoff depth (>1000, N/A)
        foundation_costs = foundation_costs / self.renewables_capacity # convert all into USD/kW
        foundation_costs = xr.where(geodata['offshore'] == True, foundation_costs, self.land_foundations)
        foundation_costs = xr.where(depth_data > 1000, np.nan, foundation_costs)
        
        # NEED TO CONVERT FROM 2016 to 2023 PRICES
        foundation_costs = foundation_costs*1.180632

        return foundation_costs
    
    def get_transmission_cost(self, dist_data):
        """ Method to calculate the cost of electricity transmission (either through HVAC or HVDC) to shore, using
        relationships from the International Energy Agency's Wind Energy Outlook 2019 """
        dist = dist_data
        
        # Initialise empty arrays
        hvac = xr.zeros_like(dist)
        hvdc = xr.zeros_like(dist)      
        
        # Apply IEA relationships
        hvac = xr.where(dist > 0, (0.0085 * dist + 0.0568), 0) * 1000 # Conversion to USD/kW
        hvdc = xr.where(dist > 0, (0.0022 * dist + 0.3878), 0) * 1000 # Conversion to USD/kW
        transmission_costs = np.minimum(hvac, hvdc)
        #self.plot_data(transmission_costs, "Transmission Costs")
        
        # NEED TO CONVERT FROM 2019 to 2023 PRICES
        transmission_costs = transmission_costs * 1.120196
        
        return transmission_costs
    
    
    def get_interarray_costs(self, technology):
        """ Method to calculate the inter-array distance between wind turbines at each location and calculate the 
        cost of either AC cables or hydrogen pipelines between all of the wind turbines """
        
        # Get installed wind capacity
        wind_capacity = self.renewables_capacity
        turbine_rating = self.turbine_rating
        turbine_diameter = self.turbine_diameter
        
        # Calculate number of turbines
        n_turbines = wind_capacity / turbine_rating
        
        # Calculate interarray distance
        spacing = 7.5 * turbine_diameter / 1000 
        interarray_dist = n_turbines * spacing
        
        if technology == 'AC':
            interarray_cost = (0.0085 * interarray_dist + 0.0568) * 1000 * turbine_rating
        elif technology == 'Pipeline':
            interarray_dist = xr.DataArray(data = np.array(interarray_dist))
            interarray_cost = self.get_pipeline_cost(interarray_dist)
        
        # NEED TO CONVERT FROM 2019 to 2023 PRICES
        interarray_cost = interarray_cost*1.147126
        
        return interarray_cost
    
    def get_pipeline_cost(self, dist_data):
        """ Method to calculate the cost of a hydrogen pipeline, based on the IEA's Future of Hydrogen Report"""
        
        # Set up constants relating to the electrolyser
        dist = dist_data
        electrolyser_size = self.electrolyser_capacity
        hydrogen_LHV = 33.3
        electrolyser_efficiency = 0.7
        
        # Calculate hydrogen capacity in tH2/day
        hydrogen_capacity = electrolyser_size * electrolyser_efficiency * 8760 / hydrogen_LHV / 1000 / 365.25 
        
        # Calculate the cost using a Linear relationship  from the IEA's Future of Hydrogen modelling assumptions
        IEA_cost = 807.38 * hydrogen_capacity + 426066  
        
        # Initialise empty arrays for storage
        h_pipeline = xr.zeros_like(dist_data)     
        
        # Apply cost relationship for hydrogen pipeline
        h_pipeline = xr.where(dist > 0, dist * IEA_cost, 0)
        pipeline_costs = h_pipeline
        
        # NEED TO CONVERT FROM 2019 to 2023 PRICES
        pipeline_costs = pipeline_costs*1.168917
        
        return pipeline_costs
    
    
    def onshore_electrolysis(self, depth, dist_data):
        
        
        # Create a storage vector for the costs
        onshore_electrolysis_costs = xr.zeros_like(dist_data)
        
        # Calculate the offshore substation costs, taken from https://guidetoanoffshorewindfarm.com/wind-farm-costs 
        # and including the offshore substation total cost + installation cost
        offshore_substation_cost = 155 * 1.25 * self.renewables_capacity 
        
        
        # Calculate the onshore substation costs taken from https://guidetoanoffshorewindfarm.com/wind-farm-costs 
        # and including the offshore substation total cost + installation cost
        onshore_substation_cost = 55 * 1.25 * self.renewables_capacity
        substation_cost = offshore_substation_cost + onshore_substation_cost
        
        # Calculate the interarray cable costs
        interarray_cable_costs = self.get_interarray_costs('AC')
        
        # Calculate the transmission costs to land
        transmission_costs = self.get_transmission_cost(dist_data) * self.renewables_capacity
        
        # Sum all costs relating to the configuration
        total_cost = offshore_substation_cost + onshore_substation_cost + interarray_cable_costs + transmission_costs
        onshore_total_costs = xr.where(depth < 1000,  total_cost, np.nan)
        
        
        # Create an xarray dataset
        data_vars = {'pipeline_costs': xr.zeros_like(dist_data),
        'transmission_costs': transmission_costs,
        'interarray_costs': xr.full_like(dist_data, interarray_cable_costs),
        'other_costs': xr.full_like(dist_data, substation_cost)}

        coords = {'latitude': dist_data.latitude,'longitude': dist_data.longitude}

        onshore_cost_breakdown = xr.Dataset(data_vars=data_vars, coords=coords)
        
        return onshore_total_costs, onshore_cost_breakdown
    
    def offshore_electrolysis(self, depth, dist_data):
        
        # Create a storage vector                           
        offshore_electrolysis_costs = xr.zeros_like(dist_data)
        
        # Calculate the interarray cable costs
        interarray_cable_costs = self.get_interarray_costs('AC')
        
        # Calculate the offshore substation costs taken from https://guidetoanoffshorewindfarm.com/wind-farm-costs 
        # and including the offshore substation total cost + installation cost
        offshore_substation_costs = 155 * 1.267747 * self.renewables_capacity 
        
        # Calculate the cost of an offshore platform for electrolysis, which are taken from 
        # https://guidetoanoffshorewindfarm.com/wind-farm-costs as the cost for facilities and structure
        # of an offshore substation and the installation cost
        offshore_platform_costs = 115 * 1.267747 * self.electrolyser_capacity 
        
        # Calculate the pipeline costs
        pipeline_costs = self.get_pipeline_cost(dist_data)
        
        # Sum all costs
        total_costs = interarray_cable_costs + offshore_substation_costs + offshore_platform_costs + pipeline_costs
        offshore_total_costs = xr.where(depth < 1000, total_costs, np.nan)
        offshore_costs = offshore_platform_costs + offshore_substation_costs
        
        # Create an xarray dataset
        data_vars = {'pipeline_costs': pipeline_costs,
        'tranmission_costs': xr.zeros_like(dist_data),
        'interarray_costs': xr.full_like(dist_data, interarray_cable_costs),
        'other_costs': xr.full_like(dist_data, offshore_costs)}
        

        coords = {'latitude': dist_data.latitude,'longitude': dist_data.longitude}

        offshore_cost_breakdown = xr.Dataset(data_vars=data_vars, coords=coords)
        
                                   
        return offshore_total_costs, offshore_cost_breakdown
                                   
        
    def distributed_electrolysis(self, depth, dist_data):
        
        # Calculate the pipeline costs to shore
        pipeline_costs = self.get_pipeline_cost(dist_data)
        
        # Calculate the interarray pipeline costs
        interarray_pipeline_costs = self.get_interarray_costs('Pipeline')
        
        # Calculate the cost of a platform for central hydrogen collection, which are taken from 
        # https://guidetoanoffshorewindfarm.com/wind-farm-costs as the cost for facilities and structure
        # of an offshore substation and the installation cost
        offshore_platform_costs = 115 * 1.267747 * self.electrolyser_capacity
        
        # Sum all costs
        total_costs = pipeline_costs + interarray_pipeline_costs + offshore_platform_costs
        distributed_costs = xr.where(depth < 1000, total_costs, np.nan)
        
        # Create an xarray dataset
        data_vars = {'pipeline_costs': pipeline_costs,
        'transmission_costs': xr.zeros_like(dist_data),
        'interarray_costs': xr.full_like(dist_data, interarray_pipeline_costs),
        'other_costs': xr.full_like(dist_data, offshore_platform_costs)}

        coords = {'latitude': dist_data.latitude,'longitude': dist_data.longitude}

        distributed_costs_breakdown = xr.Dataset(data_vars=data_vars, coords=coords)
        
        return distributed_costs, distributed_costs_breakdown
        
    

        
        
        
        
        
    def configuration_analysis(self, geodata):
    
        # Get depth and distance data
        dist_data = geodata['distance']
        depth = geodata['depth']
        offshore = geodata['offshore']
        
        # Use cost relationship with foundations and transmission
        foundation_costs_unit = self.get_foundation_cost(geodata)
        
        # Sum the costs of turbine, transmission and foundation
        turbine_foundation_costs = foundation_costs_unit * self.renewables_capacity * (1 - self.solar_fraction)
        wind_turbine_costs = self.wind_capex * self.renewables_capacity * (1 - self.solar_fraction)
        nonconfig_costs = turbine_foundation_costs + wind_turbine_costs
        
        # Extract the total cost for each of the locations
        onshore_config_cost, onshore_breakdown = self.onshore_electrolysis(depth, dist_data)
        offshore_config_cost, offshore_breakdown = self.offshore_electrolysis(depth, dist_data)
        dist_config_cost, dist_breakdown = self.distributed_electrolysis(depth, dist_data)
        
        
        # Add the turbine and foundation costs to each configuration breakdown
        onshore_breakdown['turbine_costs'] = xr.full_like(onshore_breakdown['pipeline_costs'], wind_turbine_costs)
        onshore_breakdown['foundation_costs'] = turbine_foundation_costs
        offshore_breakdown['turbine_costs'] = xr.full_like(onshore_breakdown['pipeline_costs'], wind_turbine_costs)
        offshore_breakdown['foundation_costs'] = turbine_foundation_costs
        dist_breakdown['turbine_costs'] = xr.full_like(onshore_breakdown['pipeline_costs'], wind_turbine_costs)
        dist_breakdown['foundation_costs'] = turbine_foundation_costs
        
        # Calculate the cost for each of the configurations
        
        onshore_electrolysis_cost = xr.where(offshore == True, onshore_config_cost, 0)
        offshore_electrolysis_cost = xr.where(offshore == True, offshore_config_cost, 0)
        distributed_electrolysis_cost = xr.where(offshore == True, dist_config_cost, 0)
        
            
        # Calculate total capital costs for each of the configurations
        
        onshore_electrolysis_tc = nonconfig_costs + onshore_electrolysis_cost
        offshore_electrolysis_tc = nonconfig_costs + offshore_electrolysis_cost
        distributed_electrolysis_tc = nonconfig_costs + distributed_electrolysis_cost
        
        # Calculate the minimium cost for each grid point
        storage_array = xr.zeros_like(dist_data)
        min_costs_initial = np.minimum(onshore_electrolysis_tc, offshore_electrolysis_tc)
        min_costs = np.minimum(min_costs_initial, distributed_electrolysis_tc)
        #storage_array = xr.where(min_costs == onshore_electrolysis_tc, 'On.',
                            #xr.where(min_costs == offshore_electrolysis_tc, 'Off.', 'Distr.'))
        storage_array = xr.where(min_costs == onshore_electrolysis_tc, 1,
                            xr.where(min_costs == offshore_electrolysis_tc, 2, 3))
        #breakdown = xr.where(min_costs == onshore_electrolysis_tc, onshore_breakdown, xr.where(min_costs == offshore_breakdown, 2, dist_breakdown))
        #print(breakdown)
        #self.plot_data(min_costs, "Minimum Costs")
        #self.plot_data(storage_array, "Configuration")
        #df = storage_array.to_dataframe(name='values')
        
        # Create a dataset with the three possible capital expenditures 
        data_vars = {'turbine_foundation_costs': nonconfig_costs, 'minimum capital costs': min_costs, 'minimum cost configuration' : storage_array, 'onshore electrolysis': onshore_electrolysis_tc, 
                     'offshore electrolysis': offshore_electrolysis_tc, 'distributed electrolysis': distributed_electrolysis_tc}
        coords = {'latitude': geodata.latitude,
                  'longitude': geodata.longitude}
        configuration_capital_costs = xr.Dataset(data_vars=data_vars, coords=coords)
        
        
        # Return the dataset
        return configuration_capital_costs
        

    
    def calculate_capital_depth_distance(self, geodata):
        "Updates the cost of the wind farm for each location depending on the depth and distance to shore"


        # Use cost relationship with foundations and transmission
        foundation_costs_unit = self.get_foundation_cost(geodata)
        transmission_costs_unit = self.get_transmission_cost(geodata['distance'])
        
        # Sum the costs of turbine, transmission and foundation
        foundation_costs = foundation_costs_unit * self.renewables_capacity * (1 - self.solar_fraction)
        wind_turbine_costs = self.wind_capex * self.renewables_capacity * (1 - self.solar_fraction)
        transmission_costs = transmission_costs_unit * self.renewables_capacity * (1 - self.solar_fraction)
        total_costs = foundation_costs + transmission_costs + wind_turbine_costs
        self.plot_data(total_costs, "Total Capital Costs")

        # Save a capital cost for each location (lat/lon) and return this for use in the calculations
        data_vars = {'total capital costs': total_costs, 'foundation costs': foundation_costs, 
                     'wind turbine costs': wind_turbine_costs, 'transmission costs': transmission_costs}
        coords = {'latitude': geodata.latitude,
                  'longitude': geodata.longitude}
        capital_costs = xr.Dataset(data_vars=data_vars, coords=coords)
        return capital_costs

    
    def locational_operating_costs(self, capital_costs, renewables_data_yearly, cost_category, geodata=None):
        """Calculates the operating cost associated with the renewable or electrolyser capacity as a % of the CAPEX at
        each location"""
        
        # Calculate operating cost as a proportion of CAPEX
        if cost_category == 'Solar':
            operating_cost = capital_costs * self.solar_op_cost
        elif cost_category == "Wind":
            if geodata['offshore'] == 1:
                wind_operating_costs = self.offshore_op_cost
            else:
                wind_operating_costs = self.onshore_op_cost
            operating_cost = capital_costs * wind_operating_costs
        else:
            operating_cost = capital_costs * self.elec_op_cost
        
        # Assess size of renewables_profile_yearly and reproduce operating cost each year
        operating_cost_np = operating_cost.to_numpy()
        operating_costs_extended = operating_cost_np * np.ones_like(renewables_data_yearly)
        return operating_costs_extended
        
        
        
    
    
    def plot_data(self, data, name):
    
        # Set up data
        latitudes = data.latitude.values
        longitudes = data.longitude.values
        values = data.values

        # create the heatmap using pcolormesh
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        heatmap = ax.pcolormesh(longitudes, latitudes, values, transform=ccrs.PlateCarree(), cmap='plasma')
        fig.colorbar(heatmap, ax=ax, shrink=0.5)


        # set the extent and aspect ratio of the plot
        ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())
        aspect_ratio = (latitudes.max() - latitudes.min()) / (longitudes.max() - longitudes.min())
        ax.set_aspect(aspect_ratio)

        # add axis labels and a title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(name + ' heatmap')
        ax.coastlines()
        ax.stock_img()
        
        plt.show()



        

    def calculate_electrolyser_capex(self, geodata, capacity=None):
        "Calculates the capital cost associated with the electrolyser, 1.5x the capital cost for offshore locations"
        # Get offshore mask
        offshore_mask = geodata['offshore']
        landmask = geodata['land']
        
        # Check if capacity is specified, otherwise use default
        if capacity is not None:
            electrolyser_capacity = capacity
        else:
            electrolyser_capacity = self.electrolyser_capacity
        
        # Adjust capital expenditure for electrolyser when offshore
        #capex_adjustment = xr.where(offshore_mask == True, 1.5, 1)
        capex_adjustment = xr.where(np.isnan(landmask) == True, 1.5, 1)
        
        # Calculate locational capital expenditure relating to the electrolyser
        capital_cost = capex_adjustment * self.elec_capex * electrolyser_capacity
        
        return capital_cost
    
    
    
    def calculate_combined_capital_costs(self, renewables_profile, geodata, capacity=None):
        "Calculates the yearly cost for both renewables and the electrolyser components using cost relationships with water depth and distance to shore"
        
        # Read in the renewables_profile
        renewables_data_total = renewables_profile
        renewables_data_yearly = renewables_data_total.groupby('time.year').sum(dim='time')
        
        # Extract dimensions from the renewables_profile
        years = renewables_data_yearly.year
        latitudes = renewables_data_yearly.latitude
        longitudes = renewables_data_yearly.longitude

        # Need to account for CAPEX only in the 0th year
        lat_len = len(latitudes)
        lon_len = len(longitudes)
        years_len = len(years)
        new_year = [years[0]-1]
        years_appended = np.concatenate((new_year, years))
        zero_array_renew = np.zeros((1, int(lat_len), int(lon_len)))
        zero_array_elec = np.zeros((1, int(lat_len), int(lon_len)))
        
        # Create new arrays for storage
        renewables_array = xr.DataArray(renewables_data_yearly, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': years,
                                                'latitude': latitudes,
                                                'longitude': longitudes})
        renew_costs_array = xr.DataArray(zero_array_renew, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})
        elec_costs_array = xr.DataArray(zero_array_elec, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})
        
        
        # Calculate renewable capital costs using depth
        renew_config_costs = self.configuration_analysis(geodata)
        renew_capital_costs = renew_config_costs["minimum capital costs"]
        offshore_config = renew_config_costs["minimum cost configuration"]
        
        # Calculate electrolyser capital costs
        elec_capital_costs = self.calculate_electrolyser_capex(geodata, capacity)
        
        # Transfer capital costs across to the relevant cost arrays
        elec_costs_array[0, :, :] = elec_capital_costs
        renew_costs_array[0, :, :] = renew_capital_costs
        
        # Calculate the operating costs 
        renew_op_costs_array = xr.DataArray(self.locational_operating_costs(renew_capital_costs, renewables_data_yearly, 'Renew'), dims=('year', 'latitude', 'longitude'), coords={'year': years,'latitude': latitudes,'longitude': longitudes})
        elec_op_costs_array = xr.DataArray(self.locational_operating_costs(elec_capital_costs, renewables_data_yearly, 'Elec'), dims=('year', 'latitude', 'longitude'), coords={'year': years,'latitude': latitudes,'longitude': longitudes})
        
        
        # Combine capital and operating cost arrays
        renew_costs_combined = xr.concat([renew_costs_array, renew_op_costs_array], dim='year')
        elec_costs_combined = xr.concat([elec_costs_array, elec_op_costs_array], dim = 'year')
        total_costs_array = elec_costs_combined + renew_costs_combined
        renewables_array = xr.concat([renew_costs_array * 0, renewables_array], dim = 'year')

        
        # Create a dataset with all the arrays
        data_vars = {'renewable_electricity': renewables_array,
                     'renewable costs': renew_costs_array,
                     'electrolyser costs': elec_costs_array,
                     'total costs': total_costs_array,
                     'configuration': offshore_config}
        coords = {'year': years_appended,
                  'latitude': latitudes,
                  'longitude': longitudes}
        yearly_costs = xr.Dataset(data_vars=data_vars, coords=coords)
        return yearly_costs
    
    
    def calculate_solar_capital_costs(self, renewables_profile, geodata, capacity=None):
        "Calculates the yearly cost for both solar and the electrolyser components"
        
        # Read in the renewables_profile
        renewables_data_total = renewables_profile
        renewables_data_yearly = renewables_data_total.groupby('time.year').sum(dim='time')
        
        # Extract dimensions from the renewables_profile
        years = renewables_data_yearly.year
        latitudes = renewables_data_yearly.latitude
        longitudes = renewables_data_yearly.longitude

        # Need to account for CAPEX only in the 0th year
        lat_len = len(latitudes)
        lon_len = len(longitudes)
        years_len = len(years)
        new_year = [years[0]-1]
        years_appended = np.concatenate((new_year, years))
        zero_array_renew = np.zeros((1, int(lat_len), int(lon_len)))
        zero_array_elec = np.zeros((1, int(lat_len), int(lon_len)))
        
        # Create new arrays for storage
        renewables_array = xr.DataArray(renewables_data_yearly, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': years,
                                                'latitude': latitudes,
                                                'longitude': longitudes})
        renew_costs_array = xr.DataArray(zero_array_renew, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})
        elec_costs_array = xr.DataArray(zero_array_elec, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})
        
        
        # Calculate renewable capital costs using depth
        renew_capital_costs = xr.ones_like(geodata['land']) + self.solar_capex * self.renewables_capacity
        offshore_config = xr.zeros_like(geodata['land'])
        
        # Calculate electrolyser capital costs
        elec_capital_costs = self.calculate_electrolyser_capex(geodata, capacity)
        
        # Transfer capital costs across to the relevant cost arrays
        elec_costs_array[0, :, :] = elec_capital_costs
        renew_costs_array[0, :, :] = renew_capital_costs
        
        # Calculate the operating costs 
        renew_op_costs_array = xr.DataArray(self.locational_operating_costs(renew_capital_costs, renewables_data_yearly, 'Renew'), dims=('year', 'latitude', 'longitude'), coords={'year': years,'latitude': latitudes,'longitude': longitudes})
        elec_op_costs_array = xr.DataArray(self.locational_operating_costs(elec_capital_costs, renewables_data_yearly, 'Elec'), dims=('year', 'latitude', 'longitude'), coords={'year': years,'latitude': latitudes,'longitude': longitudes})

        
        # Combine capital and operating cost arrays
        renew_costs_combined = xr.concat([renew_costs_array, renew_op_costs_array], dim='year')
        elec_costs_combined = xr.concat([elec_costs_array, elec_op_costs_array], dim = 'year')
        total_costs_array = elec_costs_combined + renew_costs_combined
        renewables_array = xr.concat([renew_costs_array * 0, renewables_array], dim = 'year')

        
        # Create a dataset with all the arrays
        data_vars = {'renewable_electricity': renewables_array,
                     'renewable costs': renew_costs_array,
                     'electrolyser costs': elec_costs_array,
                     'total costs': total_costs_array,
                     'configuration': offshore_config}
        coords = {'year': years_appended,
                  'latitude': latitudes,
                  'longitude': longitudes}
        yearly_costs = xr.Dataset(data_vars=data_vars, coords=coords)
        return yearly_costs
    
    
    
    def calculate_fractional_costs(self, renewables_profile, geodata, solar_fraction, capacity=None):
        "Calculates the yearly cost for both solar and the electrolyser components"
        
        # Read in the renewables_profile
        renewables_data_total = renewables_profile
        renewables_data_yearly = renewables_data_total.groupby('time.year').sum(dim='time')
        
        # Extract dimensions from the renewables_profile
        years = renewables_data_yearly.year
        latitudes = renewables_data_yearly.latitude
        longitudes = renewables_data_yearly.longitude

        # Need to account for CAPEX only in the 0th year
        lat_len = len(latitudes)
        lon_len = len(longitudes)
        years_len = len(years)
        new_year = [years[0]-1]
        years_appended = np.concatenate((new_year, years))
        zero_array_renew = np.zeros((1, int(lat_len), int(lon_len)))
        zero_array_solar = np.zeros((1, int(lat_len), int(lon_len)))
        zero_array_wind = np.zeros((1, int(lat_len), int(lon_len)))
        zero_array_elec = np.zeros((1, int(lat_len), int(lon_len)))
        
        # Create new arrays for storage
        renewables_array = xr.DataArray(renewables_data_yearly, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': years,
                                                'latitude': latitudes,
                                                'longitude': longitudes})
        solar_costs_array = xr.DataArray(zero_array_solar, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})
        wind_costs_array = xr.DataArray(zero_array_wind, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})
        renew_costs_array = xr.DataArray(zero_array_renew, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})
        elec_costs_array = xr.DataArray(zero_array_elec, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})
        
        
        # Calculate solar capital costs using depth
        solar_capital_costs = xr.zeros_like(geodata['land']) + self.solar_capex * self.renewables_capacity
        offshore_config = xr.zeros_like(geodata['land'])
        
        # Calculate wind capital costs using depth
        wind_config_costs = self.configuration_analysis(geodata)
        wind_capital_costs = wind_config_costs["minimum capital costs"]
        
        # Calculate electrolyser and total renewable capital costs
        elec_capital_costs = self.calculate_electrolyser_capex(geodata, capacity)
        renew_capital_costs = wind_capital_costs  * (1-solar_fraction) + solar_capital_costs * solar_fraction
        
        # Transfer capital costs across to the relevant cost arrays
        elec_costs_array[0, :, :] = elec_capital_costs
        solar_costs_array[0, :, :] = solar_capital_costs  * (solar_fraction)
        wind_costs_array[0, :, :] = wind_capital_costs * (1 - solar_fraction)
        renew_costs_array[0, :, :] = renew_capital_costs
        
        # Calculate the operating costs 
        renew_op_costs_array = xr.DataArray(self.locational_operating_costs(renew_capital_costs, renewables_data_yearly, 'Wind'), dims=('year', 'latitude', 'longitude'), coords={'year': years,'latitude': latitudes,'longitude': longitudes})
        solar_op_costs_array = xr.DataArray(self.locational_operating_costs(solar_capital_costs, renewables_data_yearly, 'Solar'), dims=('year', 'latitude', 'longitude'), coords={'year': years,'latitude': latitudes,'longitude': longitudes})
        wind_op_costs_array = xr.DataArray(self.locational_operating_costs(wind_capital_costs, renewables_data_yearly, 'Wind'), dims=('year', 'latitude', 'longitude'), coords={'year': years,'latitude': latitudes,'longitude': longitudes})
        elec_op_costs_array = xr.DataArray(self.locational_operating_costs(elec_capital_costs, renewables_data_yearly, 'Elec'), dims=('year', 'latitude', 'longitude'), coords={'year': years,'latitude': latitudes,'longitude': longitudes})

        
        # Combine capital and operating cost arrays
        solar_costs_combined = xr.concat([solar_costs_array, solar_op_costs_array], dim='year')
        wind_costs_combined = xr.concat([wind_costs_array, wind_op_costs_array], dim='year')
        elec_costs_combined = xr.concat([elec_costs_array, elec_op_costs_array], dim = 'year')
        total_costs_array = elec_costs_combined + solar_costs_combined + wind_costs_combined
        renewables_array = xr.concat([solar_costs_array * 0, renewables_array], dim = 'year')

        
        # Create a dataset with all the arrays
        data_vars = {'renewable_electricity': renewables_array,
                     'solar costs': solar_costs_array,
                     'wind costs': wind_costs_array,
                     'renewable costs': renew_costs_array,
                     'electrolyser costs': elec_costs_array,
                     'total costs': total_costs_array,
                     'configuration': offshore_config}
        coords = {'year': years_appended,
                  'latitude': latitudes,
                  'longitude': longitudes}
        yearly_costs = xr.Dataset(data_vars=data_vars, coords=coords)
        return yearly_costs
    
    
    
    
    def calculate_fractional_costs_v2(self, renewables_profile, geodata, solar_fraction, capacity=None):
        """Calculates the yearly cost for both solar and the electrolyser components
        
        Updated to include: Desalination Costs, Battery Costs and the cost of stack replacement"""
        
        # Read in the renewables_profile
        renewables_data_total = renewables_profile
        renewables_data_yearly = renewables_data_total.groupby('time.year').sum(dim='time')
        
        # Extract dimensions from the renewables_profile
        years = renewables_data_yearly.year
        latitudes = renewables_data_yearly.latitude
        longitudes = renewables_data_yearly.longitude

        # Need to account for CAPEX only in the 0th year
        lat_len = len(latitudes)
        lon_len = len(longitudes)
        years_len = len(years)
        new_year = [years[0]-1]
        years_appended = np.concatenate((new_year, years))
        zero_array_renew = np.zeros((1, int(lat_len), int(lon_len)))
        zero_array_other = np.zeros((1, int(lat_len), int(lon_len)))
        zero_array_solar = np.zeros((1, int(lat_len), int(lon_len)))
        zero_array_wind = np.zeros((1, int(lat_len), int(lon_len)))
        zero_array_elec = np.zeros((1, int(lat_len), int(lon_len)))
        
        # Create new arrays for storage
        renewables_array = xr.DataArray(renewables_data_yearly, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': years,
                                                'latitude': latitudes,
                                                'longitude': longitudes})
        solar_costs_array = xr.DataArray(zero_array_solar, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})
        wind_costs_array = xr.DataArray(zero_array_wind, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})
        renew_costs_array = xr.DataArray(zero_array_renew, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})
        other_costs_array = xr.DataArray(zero_array_other, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})
        elec_costs_array = xr.DataArray(zero_array_elec, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})
        
        
        # Calculate solar capital costs using depth
        solar_capital_costs = xr.zeros_like(geodata['land']) + self.solar_capex * self.renewables_capacity
        offshore_config = xr.zeros_like(geodata['land'])
        
        # Calculate wind capital costs using depth
        wind_config_costs = self.configuration_analysis(geodata)
        turbine_foundation_costs = wind_config_costs["turbine_foundation_costs"]
        wind_capital_costs = wind_config_costs["minimum capital costs"]
        
        # Calculate other costs
        battery_capital_costs = self.battery_cost * self.battery_power_relative * self.renewables_capacity
        compressor_costs = 0.887*self.renewables_capacity
        #desalination_costs = 0.491*self.renewables_capacity
        other_capital_costs = battery_capital_costs + compressor_costs
        
        
        # Calculate electrolyser and total renewable capital costs
        elec_capital_costs = self.calculate_electrolyser_capex(geodata, capacity)
        renew_capital_costs = wind_capital_costs  * (1-solar_fraction) + solar_capital_costs * solar_fraction
        
        # Transfer capital costs across to the relevant cost arrays
        elec_costs_array[0, :, :] = elec_capital_costs
        solar_costs_array[0, :, :] = solar_capital_costs  * (solar_fraction)
        wind_costs_array[0, :, :] = turbine_foundation_costs * (1 - solar_fraction)
        renew_costs_array[0, :, :] = renew_capital_costs
        other_costs_array[0, :, :] = other_capital_costs
        
        # Calculate the operating costs 
        solar_op_costs_array = xr.DataArray(self.locational_operating_costs(solar_capital_costs*solar_fraction, renewables_data_yearly, 'Solar'), dims=('year', 'latitude', 'longitude'), coords={'year': years,'latitude': latitudes,'longitude': longitudes})
        wind_op_costs_array = xr.DataArray(self.locational_operating_costs(wind_capital_costs*(1-solar_fraction), renewables_data_yearly, 'Wind', geodata), dims=('year', 'latitude', 'longitude'), coords={'year': years,'latitude': latitudes,'longitude': longitudes})
        renew_op_costs_array = wind_op_costs_array + solar_op_costs_array
        elec_op_costs_array = xr.DataArray(self.locational_operating_costs(elec_capital_costs, renewables_data_yearly, 'Elec'), dims=('year', 'latitude', 'longitude'), coords={'year': years,'latitude': latitudes,'longitude': longitudes})
        
        # Combine capital and operating cost arrays
        solar_costs_combined = xr.concat([solar_costs_array, solar_op_costs_array], dim='year')
        wind_costs_combined = xr.concat([wind_costs_array, wind_op_costs_array], dim='year')
        elec_costs_combined = xr.concat([elec_costs_array, elec_op_costs_array], dim = 'year')
        other_costs_combined = xr.concat([other_costs_array, elec_op_costs_array*0], dim = 'year')
        total_costs_array = elec_costs_combined + solar_costs_combined + wind_costs_combined + other_costs_combined
        renewables_array = xr.concat([solar_costs_array * 0, renewables_array], dim = 'year')

        
        # Create a dataset with all the arrays
        data_vars = {'renewable_electricity': renewables_array,
                     'solar costs': solar_costs_array,
                     'wind costs': wind_costs_array,
                     'renewable costs': renew_costs_array,
                     'other costs': other_costs_array,
                     'electrolyser costs': elec_costs_array,
                     'total costs': total_costs_array,
                     'configuration': offshore_config}
        coords = {'year': years_appended,
                  'latitude': latitudes,
                  'longitude': longitudes}
        yearly_costs = xr.Dataset(data_vars=data_vars, coords=coords)
        return yearly_costs