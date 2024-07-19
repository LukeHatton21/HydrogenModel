import dask
import numpy
import xarray as xr
import os
import re
import glob
import numpy as np
import pandas as pd
    

class All_Files:
    def __init__(self, lat_lon, solar_path, wind_path, solar_format, wind_format):
        self.lat_min = lat_lon[0]
        self.lat_max = lat_lon[1]
        self.lon_min = lat_lon[2]
        self.lon_max = lat_lon[3]
        

        self.solar_path = solar_path
        self.wind_path = wind_path
        self.solar_format = solar_format
        self.wind_format = wind_format
    
    
    def extract_unique_numbers(self):
        def extract_numbers(filename):
            numbers = re.findall(r'\d{4}', filename)
            return numbers

        folder_path = self.solar_path
        unique_numbers = set()

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                numbers = extract_numbers(filename)
                unique_numbers.update(numbers)

        return list(unique_numbers)
    
    
    def preprocess_combine_monthly(self, year_override=None):
    
        # Read in relevant files by year   
        if year_override is None:
            years = sorted(self.extract_unique_numbers())
        else: 
            years = [2010, 2017]
        
        
        # For loop by years
        for yearcount, year in enumerate(years):
            
            file_list = sorted(glob.glob(self.filepath + self.name_format + str(year) + '*.nc'))
        
            # For loop by months
            for count, file in enumerate(file_list):
                
                # Open file
                ds = xr.open_dataset(file)

                
                # Select the relevant latitudes/longitudes limits
                reduced_ds = ds.sel(latitude=slice(self.lat_min, self.lat_max), longitude=slice(self.lon_min, self.lon_max))
                
                # Preprocess monthly files that have 30 or 28 days
                list = [3,5,8,10]
                if count == 1:
                    adjusted_ds = self.adjust_feb(reduced_ds)
                elif count in list:
                    adjusted_ds = self.adjust_30_days(reduced_ds)
                    
                else: 
                    adjusted_ds = reduced_ds

                    
                    
                
                # Merge yearly files together using a loop
                if count == 0:
                    combined_ds = adjusted_ds
                else:
                    combined_ds = xr.concat([combined_ds, adjusted_ds], dim='time')
            

            # Assign the new time coordinate array back to the dataset
            new_time = combined_ds.time.values - np.timedelta64(7, "h")
            combined_ds = combined_ds.assign_coords(time=new_time)
            
        
        
        
        
            # Combine annual datasets into one
            if yearcount == 0:
                yearly_combined_ds = combined_ds
            else:
                yearly_combined_ds = xr.concat([yearly_combined_ds, combined_ds], dim='time')
            #print("Combined File (Yearly):")
            #print(yearly_combined_ds)
        
        
        # Whilst data is still non-continuous, align data so that it is continuous for use in the model
        length = yearly_combined_ds['time'].size - 1
        start_date = yearly_combined_ds.time.values[0]
        end_date = start_date + np.timedelta64(length, "h")
        new_years = pd.date_range(start_date, end_date, freq="h")
        yearly_combined_ds = yearly_combined_ds.assign_coords(time=new_years)
        
        # Return total dataset
        return yearly_combined_ds, years   
    
    
    def preprocess_combine_yearly(self, year_override=None):
        
        
        # Read in relevant files by year   
        if year_override is None:
            years = sorted(self.extract_unique_numbers())
        else: 
            years = [2010, 2017]
        
        
        # For loop by years
        for yearcount, year in enumerate(years):
            
            solar_filename = sorted(glob.glob(self.solar_path + self.solar_format + str(year) + '*.nc'))
            wind_filename = sorted(glob.glob(self.wind_path + self.wind_format + str(year) + '*.nc'))    
            
            # Open file
            #ds = xr.open_dataset(filename[0], chunks={'lat': 10, 'lon': 10})
            solar_ds = xr.open_dataset(solar_filename[0])
            wind_ds = xr.open_dataset(wind_filename[0])
            
            # Get dim names and sizes
            solar_dim_names = []
            solar_dim_sizes = []
            wind_dim_names = []
            wind_dim_sizes = []
            
            for dim_name, dim_size, in solar_ds.dims.items():
                solar_dim_names.append(dim_name)
                solar_dim_sizes.append(dim_size)
            
            for dim_name, dim_size, in wind_ds.dims.items():
                wind_dim_names.append(dim_name)
                wind_dim_sizes.append(dim_size)

            if str('lat') in solar_dim_names:
                solar_ds = solar_ds.rename({'lat': 'latitude', 'lon': 'longitude'})
                solar_ds = solar_ds.transpose("time", "latitude", "longitude")
                
            if str('lat') in wind_dim_names:
                wind_ds = wind_ds.rename({'lat': 'latitude', 'lon': 'longitude'})
                wind_ds = wind_ds.transpose("time", "latitude", "longitude")
            
            # Select the relevant latitudes/longitudes limits
            solar_selected_ds = solar_ds.sel(latitude=slice(self.lat_min, self.lat_max), longitude=slice(self.lon_min, self.lon_max))
            wind_selected_ds = wind_ds.sel(latitude=slice(self.lat_min, self.lat_max), longitude=slice(self.lon_min, self.lon_max))
                
             # Merge yearly files together using a loop
            if yearcount == 0:
                solar_yearly_combined_ds = solar_selected_ds
                wind_yearly_combined_ds = wind_selected_ds
            else:
                solar_yearly_combined_ds = xr.concat([solar_yearly_combined_ds, solar_selected_ds], dim='time')
                wind_yearly_combined_ds = xr.concat([wind_yearly_combined_ds, wind_selected_ds], dim='time')
        
        
        # Whilst data is still non-continuous, align data so that it is continuous for use in the model
        length = solar_yearly_combined_ds['time'].size - 1
        start_date = solar_yearly_combined_ds.time.values[0]
        end_date = start_date + np.timedelta64(length, "h")
        new_years = pd.date_range(start_date, end_date, freq="h")
        solar_yearly_combined_ds = solar_yearly_combined_ds.assign_coords(time=new_years)
        wind_yearly_combined_ds = wind_yearly_combined_ds.assign_coords(time=new_years)
        
        # Return total dataset
        return solar_yearly_combined_ds, wind_yearly_combined_ds, years   
        
        
        
        
        
        
    def adjust_30_days(self, dataset):
    
        # Select first 30 days of the dataset only
        adjusted_data = dataset.isel(time=slice(None, 720))
        
        # Return adjusted dataset
        return adjusted_data
    
    
    
    def adjust_feb(self, dataset):
        
        # Select first 28 days of the dataset
        adjusted_data = dataset.isel(time=slice(None, 672))
        
        # Return adjusted dataset
        return adjusted_data
        
        

