import numpy as np
import xarray as xr
import glob
import pandas as pd
import pvlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import time

class All_Locations:
    # List of files and relevant locations
    def __init__(self, filename, path=None):
        self.path = path
        self.filename = filename
        if path is None:
            self.file_list = glob.glob(r'/Users/china_lcoh_model' + filename)
            for count, file in enumerate(self.file_list):
                if count == 0:
                    self.ds = xr.open_dataset(file)
            else:
                self.ds = xr.merge([self.ds, xr.open_dataset(file)])

# Class to generate a power generation profile from a weather profile
class Get_Renewables:
    def __init__(self, data, percentage_wind, turbine=None):
        "Initialises the Renewable_Profile class and sets up the solar model"
        self.data = data.ds
        # self.altitudes = data.altitude
        self.percentage_wind = percentage_wind
        if turbine is None:
            self.turbine = 'Vestas.V164.7000'
        else:
            self.turbine = turbine
        self.hourly_data = pd.to_datetime(self.data.time.values)

    def get_data(self, latitude, longitude):
        """Imports the data from the nc files and interprets them into wind and solar profiles"""
        ds = self.data
        self.latitude = latitude
        self.longitude = longitude
        # self.altitudes = ds.altitude
        v100 = ds.v100.loc[:, self.latitude, self.longitude].values
        u100 = ds.u100.loc[:, self.latitude, self.longitude].values
        s100 = [(u ** 2 + v ** 2) ** 0.5 for u, v in zip(u100, v100)]
        # t2m = ds.t2m.loc[:, self.latitude, self.longitude].values
        # ssrd = ds.ssrd.loc[:, self.latitude, self.longitude].values
        # v10 = ds.v10.loc[:, self.latitude, self.longitude].values
        # u10 = ds.u10.loc[:, self.latitude, self.longitude].values
        # s10 = [(u ** 2 + v ** 2) ** 0.5 for u, v in zip(u10, v10)]
        # v1 = np.array([s * np.log(1 / 0.03) / np.log(10 / 0.03) for s in s10])
        #return [self.get_wind_power(v100, u100)]  # , self.get_solar_power(ssrd, t2m, v1, ds.altitude),
        return [self.get_wind_power_using_RN(s100)]

    def get_wind_power(self, u100, v100):
        """Given u100 and v100 estimates wind power using the given wind turbine power curve - needs updating potentially"""
        measured_height = 100
        hub_height = 120
        rated = 3000  # installed capacity in kW
        cut_in = 3
        cut_out = 25
        power = []
        for u, v in zip(u100, v100):
            speed_measured = (u ** 2 + v ** 2) ** 0.5
            speed_hub = speed_measured * np.log(hub_height / 0.03) / np.log(measured_height / 0.03)
            if speed_hub < cut_in or speed_hub > cut_out:
                power.append(0)
            elif speed_hub < 7.5:
                power.append(2.785299 * speed_hub ** 3.161124 / rated)
            elif speed_hub < 11.5:
                power.append((-103.447526 * speed_hub ** 2 + 2319.060494 * speed_hub - 10004.69559) / rated)
            else:
                power.append(1)
        return np.array(power)
    
    def get_wind_power_using_RN(self, s100):
        
        # Get the turbine power curve
        turbine = self.turbine
        df = pd.read_csv('WindTurbinePowerCurve.csv')
        power_curve = df[turbine]
        
         # Create a storage array for the power
        power_array = np.zeros_like(s100)
        
        # Scale the wind speed to the wind speed at the hub height
        
        # Convert the speed into an array, multiply by 100 and round
        s100_arr = np.array(s100)
        s100_scaled = 100 * s100_arr
        s100_rounded = s100_scaled.round(decimals=0)
        s100_final = np.where(s100_rounded > 4000, 4000, s100_rounded)
        
        # Create a numpy iterator object and use indexing to convert from wind speed to power
        it = np.nditer(s100_final, flags=['f_index'])
        for speed in it:
            power_array[it.index] = power_curve[speed]
            
        # Return power_array
        return power_array

    def get_solar_power(self, ssrd, t2m, v1, altitude):
        """Uses PV_Lib to estimate solar power based on provided weather data"""
        """Note t2m to the function in Kelvin - function converts to degrees C!"""
        # Manipulate input data
        times = self.hourly_data.tz_localize('ETC/GMT')
        ssrd = pd.DataFrame(ssrd / 3600, index=times, columns=['ghi'])
        t2m = pd.DataFrame(t2m - 273.15, index=times, columns=['temp_air'])
        v1 = pd.DataFrame(v1, index=times, columns=['wind_speed'])

        # Set up solar farm design
        mc_location = pvlib.location.Location(latitude=self.latitude, longitude=self.longitude, altitude=altitude,
                                              name='NA')
        solpos = pvlib.solarposition.pyephem(times, latitude=self.latitude, longitude=self.longitude, altitude=altitude,
                                             pressure=101325, temperature=t2m.mean(), horizon='+0:00')
        mc = pvlib.modelchain.ModelChain(self.__pvwatts_system, mc_location, aoi_model='physical',
                                         spectral_model='no_loss')

        # Get the diffuse normal irradiance (dni) and diffuse horizontal irradiance (dhi) from the data; hence create a weather dataframe
        df_res = pd.concat([ssrd, t2m, v1, solpos['zenith']], axis=1)
        df_res['dni'] = pd.Series([pvlib.irradiance.disc(ghi, zen, i)['dni'] for ghi, zen, i in
                                   zip(df_res['ghi'], df_res['zenith'], df_res.index)], index=times).astype(float)
        df_res['dhi'] = df_res['ghi'] - df_res['dni'] * np.cos(np.radians(df_res['zenith']))
        weather = df_res.drop('zenith', axis=1)
        dc_power = mc.run_model(weather).dc / 240
        return np.array(dc_power)

# Class to call and run the Get_Renewables class for given latitude and longitudes, generating a renewable profile from the weather profile
class RenewableCalculator:
    def __init__(self, filename, outputfile, lat_lon=None, step=None):
        self.filename = filename
        self.outputfile = outputfile
        if step is not None:
            self.step = step
        else: 
            self.step = 0.25
        if lat_lon is not None:
            self.lat_min = lat_lon[0]
            self.lat_max = lat_lon[1]
            self.lon_min = lat_lon[2]
            self.lon_max = lat_lon[3]
        else:
            ds = xr.open_dataset('/Users/china_lcoh_model/' + filename)
            latitudes = ds.latitude.values
            longitudes = ds.longitude.values
            self.lat_min = latitudes[-1] + self.step
            self.lat_max = latitudes[0]
            self.lon_min = longitudes[0]
            self.lon_max = longitudes[-1] + self.step

            
       
       
        

    def run_renewable_calculator(self):
        lon_range = np.arange(self.lon_min, self.lon_max, self.step)
        lat_range = np.arange(self.lat_min, self.lat_max, self.step)
        data = All_Locations(self.filename)
        get_renewables_class = Get_Renewables(data=data, percentage_wind=1, turbine=None,)

        Solar = np.zeros((len(get_renewables_class.hourly_data), len(lat_range), len(lon_range)))
        Wind = np.zeros((len(get_renewables_class.hourly_data), len(lat_range), len(lon_range)))

        for count_lat, lat in enumerate(lat_range):
            for count_lon, lon in enumerate(lon_range):
                location_data = get_renewables_class.get_data(lat, lon)
                Solar[:, count_lat, count_lon] = location_data[0] * (1 - get_renewables_class.percentage_wind)
                Wind[:, count_lat, count_lon] = location_data[0] * (get_renewables_class.percentage_wind)

        Electricity = Solar + Wind

        # Convert latitude and longitude lists to tuples
        latitude = tuple(lat_range.tolist())
        longitude = tuple(lon_range.tolist())

        renewables_dataset = xr.Dataset(data_vars={'CF': (['time', 'latitude', 'longitude'], Wind)}, coords=dict(
    latitude=(['latitude'], lat_range.tolist()), longitude=(['longitude'], lon_range.tolist()),
    time=(['time'], get_renewables_class.hourly_data)), )
        renewables_dataset.to_netcdf(self.outputfile, mode='w')
        print(renewables_dataset)
        return renewables_dataset
    
    
    
    
filename = input("Input File Name: ")
outputfile = input("Output File Name: ")
start_time = time.time()
method = RenewableCalculator(filename, outputfile)
renewable_calculations = method.run_renewable_calculator()

# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

# Print elapsed time in seconds
print(f"Model took {elapsed_time:.2f} seconds to run")
