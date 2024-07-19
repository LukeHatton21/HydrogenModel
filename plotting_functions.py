import cartopy.crs as ccrs
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cartopy.feature as cfeature
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_data_shading(values, latitudes, longitudes, anchor=None, filename=None, increment=None, title=None, tick_values=None, cmap=None, extend=None, graphmarking=None, special_value=None, hatch_label=None, hatch_label_2=None, special_value_2=None, center_norm=None):      
    
    # create the heatmap using pcolormesh
    if anchor is None:
        anchor = 0.355
    fig = plt.figure(figsize=(30, 15), facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    if center_norm is None:
        heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.Normalize(vmin=tick_values[0], vmax=tick_values[-1]), transform=ccrs.PlateCarree(), cmap=cmap)
    else:
        heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.SymLogNorm(vmin = tick_values[0], vmax=tick_values[-1], linscale
=1, linthresh=1), transform=ccrs.PlateCarree(), cmap=cmap)
    
    # Check if there is a need for extension
    values_min = np.nanmin(values)
    values_max = np.nanmax(values)
    if values_min < tick_values[0]:
        extend = "min"
    if values_max > tick_values[-1]:
        extend = "max"
    if (values_max > tick_values[-1]) & (values_min < tick_values[0]):
        extend="both"
        
    axins = inset_axes(
    ax,
    width="1.5%",  
    height="80%",  
    loc="lower left",
    bbox_to_anchor=(1.05, 0., 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0,
)
    cb = fig.colorbar(heatmap, cax=axins, shrink=0.5, ticks=tick_values, format="%0.0f", extend=extend, anchor=(0, anchor))


    cb.ax.tick_params(labelsize=20)
    if title is not None:
        cb.ax.set_title(title, fontsize=25)
        
    # Add the special shading
    if special_value is not None:
        special_overlay = np.where(values == special_value, 1, np.nan)
        hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['/'], colors="silver", linewidth=0.15, transform=ccrs.PlateCarree())
        
    if special_value_2 is not None:
        special_overlay = np.where(values == special_value_2, 1, np.nan)
        hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['\\'], colors="gold", linewidth=0.15, transform=ccrs.PlateCarree())

    # set the extent and aspect ratio of the plot
    ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())
    aspect_ratio = (latitudes.max() - latitudes.min()) / (longitudes.max() - longitudes.min())
    ax.set_aspect(1)

    # add axis labels and a title
    ax.set_xlabel('Longitude', fontsize=30)
    ax.set_ylabel('Latitude', fontsize=30)
    borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m', facecolor='none')
    ax.add_feature(borders, edgecolor='gray', linestyle=':')
    ax.coastlines()
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('top')
    ax.coastlines()
    if graphmarking is not None:
        ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')
     
    hatch_patches=[]
    if special_value is not None and hatch_label is not None:
        hatch_patch_1 = Patch(facecolor='silver', edgecolor='black', hatch="/", label=hatch_label)
        hatch_patches.append(hatch_patch_1)

    if hatch_label_2 is not None:
        hatch_patch_2 = Patch(facecolor='gold', edgecolor='black', hatch="/", label=hatch_label_2)
        hatch_patches.append(hatch_patch_2)

    if hatch_patches:
        ax.legend(handles=hatch_patches, loc='lower left', fontsize=20)
    
    if filename is not None:
        plt.savefig(filename + ".png", bbox_inches="tight")
    
    return 



def produce_figure_3(solar_data, wind_data, hybrid_data, output_folder, technology):
    
    # Produce LCOH plots
    plot_data_shading(solar_data['levelised_cost'].sel(latitude=slice(-60, 90)), solar_data.latitude.sel(latitude=slice(-55, 90)), solar_data.longitude, tick_values=[2, 3, 4, 5, 6, 7, 8, 9, 10, 15], cmap="YlOrRd_r", center_norm="True", title="Solar\n LCOH\n (US$/kg)\n", filename=output_folder + "LCOH/Solar_LCOH_" + technology, graphmarking="a")
    plot_data_shading(wind_data['levelised_cost'].sel(latitude=slice(-60, 90)), wind_data.latitude.sel(latitude=slice(-55, 90)), wind_data.longitude, tick_values=[2, 3, 4, 5, 6, 7, 8, 9, 10, 15], cmap="YlGnBu_r", center_norm="True", title="Wind\n LCOH\n (US$/kg)\n", filename=output_folder + "LCOH/Wind_LCOH_" + technology, graphmarking="b")
    plot_data_shading(hybrid_data['levelised_cost'].sel(latitude=slice(-60, 90)), hybrid_data.latitude.sel(latitude=slice(-55, 90)), hybrid_data.longitude, tick_values=[2, 3, 4, 5, 6, 7, 8, 9, 10, 15], cmap="RdPu_r", center_norm="True", title="Hybrid\n LCOH\n (US$/kg)\n", filename=output_folder + "LCOH/Hybrid_LCOH_" + technology, graphmarking="c")
    optimal_sf = xr.where(np.isnan(solar_data['levelised_cost'])==True, np.nan, hybrid_data['Optimal_SF'])
    plot_data_shading(optimal_sf.sel(latitude=slice(-60, 90)), hybrid_data.latitude.sel(latitude=slice(-55, 90)), hybrid_data.longitude, tick_values=[0.1, 10, 20, 30, 40 ,50, 60, 70, 80, 90, 100], cmap="RdYlBu_r", title="Optimal\n Solar\n Fraction\n(%)\n", filename=output_folder + "LCOH/Hybrid_SF_" + technology, graphmarking="d")
    
    # Produce Electrolyser Capacity Plots
    plot_data_shading(solar_data['electrolyser_capacity'].sel(latitude=slice(-60, 90))*100, solar_data.latitude.sel(latitude=slice(-55, 90)), solar_data.longitude, tick_values=[0.1, 10, 20, 30, 40 ,50, 60, 70, 80, 90, 100], cmap="YlOrRd", title="Solar\n Electrolyser\n Sizing\nRatio (%)\n", filename=output_folder + "ELEC/Solar_Elec_" + technology, graphmarking="a")
    plot_data_shading(wind_data['electrolyser_capacity'].sel(latitude=slice(-60, 90))*100, wind_data.latitude.sel(latitude=slice(-55, 90)), wind_data.longitude, tick_values=[0.1, 10, 20, 30, 40 ,50, 60, 70, 80, 90, 100], cmap="YlGnBu", title="Wind\n Electrolyser\n Sizing\nRatio (%)\n", filename=output_folder + "ELEC/Wind_Elec_" + technology, graphmarking="b")
    plot_data_shading(hybrid_data['electrolyser_capacity'].sel(latitude=slice(-60, 90))*100, hybrid_data.latitude.sel(latitude=slice(-55, 90)), hybrid_data.longitude, tick_values=[0.1, 10, 20, 30, 40 ,50, 60, 70, 80, 90, 100], cmap="RdPu", title="Hybrid\n Electrolyser\n Sizing\nRatio (%)\n", filename=output_folder + "ELEC/Hybrid_Elec_" + technology, graphmarking="c")
    


def plot_binary_data(self, data, title, colour_1, colour_2, parameter_1, parameter_2, filename=None, graphmarking=None):

    sel_data = data.sel(latitude=slice(-60, 90))
    values = sel_data.values
    latitudes = sel_data.latitude.values
    longitudes = sel_data.longitude.values

    # Create the heatmap using pcolormesh
    fig = plt.figure(figsize=(20, 15.5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Define custom colors for 1 and 0
    custom_cmap = plt.cm.colors.ListedColormap([colour_1, colour_2])
    heatmap = ax.pcolormesh(longitudes, latitudes, values, transform=ccrs.PlateCarree(), cmap=custom_cmap)

    # Create a legend
    legend_elements = [plt.Line2D([0], [0], marker='s', color='w', label=parameter_1, markerfacecolor=colour_1, markersize=15),
                   plt.Line2D([0], [0], marker='s', color='w', label=parameter_2, markerfacecolor=colour_2, markersize=15)]
    legend = ax.legend(handles=legend_elements, loc='lower left', fontsize=20)

    # Add a title to the legend
    legend.set_title(title, prop={'size': 20})

    # Set the extent and aspect ratio of the plot
    ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())

    # Add axis labels and a title
    ax.set_xlabel('Longitude', fontsize=30)
    ax.set_ylabel('Latitude', fontsize=30)
    borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m', facecolor='none')
    ax.add_feature(borders, edgecolor='gray', linestyle=':')
    ax.coastlines()
    if graphmarking is not None:
        ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')
    if filename is not None:
        plt.savefig(filename + '.png')
    plt.show()

    

    
output_folder = "/Users/lukehatton/Green Hydrogen 2024/PLOTS/FINAL/"
produce_figure_3(results_model.PEM_100, results_model.PEM_0, results_model.PEM_cheapest_sf, output_folder, "PEM")
produce_figure_3(results_model.ALK_100, results_model.ALK_0, results_model.ALK_cheapest_sf, output_folder, "ALK")






import cartopy.crs as ccrs
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cartopy.feature as cfeature
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_data_shading(values, latitudes, longitudes, anchor=None, filename=None, increment=None, title=None, tick_values=None, cmap=None, extend=None, graphmarking=None, special_value=None, hatch_label=None, hatch_label_2=None, special_value_2=None, center_norm=None, decimals=None):      
    
    # create the heatmap using pcolormesh
    if anchor is None:
        anchor = 0.355
    fig = plt.figure(figsize=(30, 15), facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    if center_norm is None:
        heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.Normalize(vmin=tick_values[0], vmax=tick_values[-1]), transform=ccrs.PlateCarree(), cmap=cmap)
    else:
        heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.SymLogNorm(vmin = tick_values[0], vmax=tick_values[-1], linscale
=1, linthresh=1), transform=ccrs.PlateCarree(), cmap=cmap)
    
    # Check if there is a need for extension
    values_min = np.nanmin(values)
    values_max = np.nanmax(values)
    if values_min < tick_values[0]:
        extend = "min"
    if values_max > tick_values[-1]:
        extend = "max"
    if (values_max > tick_values[-1]) & (values_min < tick_values[0]):
        extend="both"
        
    axins = inset_axes(
    ax,
    width="1.5%",  
    height="80%",  
    loc="lower left",
    bbox_to_anchor=(1.05, 0., 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0,
)
    if decimals is None:
        decimals = "%0.0f"
    cb = fig.colorbar(heatmap, cax=axins, shrink=0.5, ticks=tick_values, format=decimals, extend=extend, anchor=(0, anchor))


    cb.ax.tick_params(labelsize=20)
    if title is not None:
        cb.ax.set_title(title, fontsize=25)
        
    # Add the special shading
    if special_value is not None:
        special_overlay = np.where(values == special_value, 1, np.nan)
        hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['/'], colors="silver", linewidth=0.15, transform=ccrs.PlateCarree())
        
    if special_value_2 is not None:
        special_overlay = np.where(values == special_value_2, 1, np.nan)
        hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['\\'], colors="gold", linewidth=0.15, transform=ccrs.PlateCarree())

    # set the extent and aspect ratio of the plot
    ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())
    aspect_ratio = (latitudes.max() - latitudes.min()) / (longitudes.max() - longitudes.min())
    ax.set_aspect(1)

    # add axis labels and a title
    ax.set_xlabel('Longitude', fontsize=30)
    ax.set_ylabel('Latitude', fontsize=30)
    borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m', facecolor='none')
    ax.add_feature(borders, edgecolor='gray', linestyle=':')
    ax.coastlines()
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('top')
    ax.coastlines()
    if graphmarking is not None:
        ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')
     
    hatch_patches=[]
    if special_value is not None and hatch_label is not None:
        hatch_patch_1 = Patch(facecolor='silver', edgecolor='black', hatch="/", label=hatch_label)
        hatch_patches.append(hatch_patch_1)

    if hatch_label_2 is not None:
        hatch_patch_2 = Patch(facecolor='gold', edgecolor='black', hatch="/", label=hatch_label_2)
        hatch_patches.append(hatch_patch_2)

    if hatch_patches:
        ax.legend(handles=hatch_patches, loc='lower left', fontsize=20)
    
    if filename is not None:
        plt.savefig(filename + ".png", bbox_inches="tight")
    
    return 


# Read in PEM and Alkaline cheapest
PEM_cheapest = results_model.PEM_cheapest_sf
ALK_cheapest = results_model.ALK_cheapest_sf

# Read in PEM values
PEM_solar = results_model.PEM_100
PEM_wind = results_model.PEM_0

# Read in PEM values
ALK_solar = results_model.ALK_100
ALK_wind = results_model.ALK_0

# Set efficiency for each technology
efficiency_pem = 0.68
efficiency_alk = 0.65

# Calculate the max electricity price
PEM_cheapest = results_model.get_max_electricity_price(PEM_cheapest, efficiency_pem)
ALK_cheapest = results_model.get_max_electricity_price(ALK_cheapest, efficiency_alk)

# Calculate the carbon intensity
PEM_cheapest = results_model.calculate_embodied_carbon(PEM_cheapest, "Hybrid", "PEM")
ALK_cheapest = results_model.calculate_embodied_carbon(ALK_cheapest, "Hybrid", "ALK")

# Calculate the carbon intensity
PEM_solar  = results_model.calculate_embodied_carbon(PEM_solar , "Solar", "PEM")
PEM_wind = results_model.calculate_embodied_carbon(PEM_wind, "Wind", "PEM")
ALK_solar  = results_model.calculate_embodied_carbon(ALK_solar , "Solar", "ALK")
ALK_wind = results_model.calculate_embodied_carbon(ALK_wind, "Wind", "ALK")

# Calculate the max carbon intensity
PEM_cheapest = results_model.get_max_ci(PEM_cheapest, efficiency_pem, 1)
ALK_cheapest = results_model.get_max_ci(ALK_cheapest, efficiency_alk, 1)

def produce_figures_4_8_9(data, wind_data, solar_data, output_folder, technology):
    
    
    # Plot figure 4
    plot_data_shading(data['electricity_price_max'].sel(latitude=slice(-60, 90))*1000, data.latitude.sel(latitude=slice(-60, 90)), data.longitude, tick_values=[50, 100, 200, 300, 400, 500], center_norm="True", cmap="plasma_r", title="Max Grid\n Import \n Price\n (US$/MWh)\n", filename=output_folder + "Elec_Price_"+technology)
    
    # Plot figure 8
    plot_data_shading(solar_data['carbon_intensity'].sel(latitude=slice(-60, 90)), solar_data.latitude.sel(latitude=slice(-60, 90)), solar_data.longitude, tick_values=[0.01, 0.5, 1, 2.5, 5], center_norm="True", cmap="BuGn", graphmarking="a", decimals="%0.1f", title="Embodied\n Carbon\n Intensity\n (kgCO\u2082/kgH\u2082)\n", filename=output_folder + "Solar_CI_"+technology)
    plot_data_shading(wind_data['carbon_intensity'].sel(latitude=slice(-60, 90)), wind_data.latitude.sel(latitude=slice(-60, 90)), wind_data.longitude, tick_values=[0.01, 0.5, 1, 2.5, 5], center_norm="True", cmap="BuGn", graphmarking="b", decimals="%0.1f", title="Embodied\n Carbon\n Intensity\n (kgCO\u2082/kgH\u2082)\n", filename=output_folder + "Wind_CI_"+technology)
    plot_data_shading(data['carbon_intensity'].sel(latitude=slice(-60, 90)), data.latitude.sel(latitude=slice(-60, 90)), data.longitude, tick_values=[0.01, 0.5, 1, 2.5, 5], center_norm="True", cmap="BuGn", graphmarking="c", decimals="%0.1f", title="Embodied\n Carbon\n Intensity\n (kgCO\u2082/kgH\u2082)\n", filename=output_folder + "Hybrid_CI_"+technology)
    
    
    
    
    # Plot figure 9
    plot_data_shading(data['electricity_ci_max'].sel(latitude=slice(-60, 90))*1000, data.latitude.sel(latitude=slice(-60, 90)), data.longitude, tick_values=[0.00001, 5, 10, 25, 50, 100, 200], center_norm="True", cmap="plasma_r", special_value = 999000, hatch_label="Locations exceeding\n 1kgCO\u2082/kgH\u2082", title="Max\n Grid\n Intensity\n (gCO\u2082/kWh)\n", filename=output_folder + "Elec_CI_"+technology)

# Call function
output_folder = "/Users/lukehatton/Green Hydrogen 2024/PLOTS/FINAL/CI_GRID/"
produce_figures_4_8_9(PEM_cheapest, PEM_wind, PEM_solar, output_folder, "PEM")
produce_figures_4_8_9(ALK_cheapest, ALK_wind, ALK_solar, output_folder, "ALK")





import cartopy.crs as ccrs
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cartopy.feature as cfeature
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_data_shading(values, latitudes, longitudes, anchor=None, filename=None, increment=None, title=None, tick_values=None, cmap=None, extend=None, graphmarking=None, special_value=None, hatch_label=None, hatch_label_2=None, special_value_2=None, center_norm=None, decimals=None):      
    
    # create the heatmap using pcolormesh
    if anchor is None:
        anchor = 0.355
    fig = plt.figure(figsize=(30, 15), facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    if center_norm is None:
        heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.Normalize(vmin=tick_values[0], vmax=tick_values[-1]), transform=ccrs.PlateCarree(), cmap=cmap)
    else:
        heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.SymLogNorm(vmin = tick_values[0], vmax=tick_values[-1], linscale
=1, linthresh=1), transform=ccrs.PlateCarree(), cmap=cmap)
    
    # Check if there is a need for extension
    values_min = np.nanmin(values)
    values_max = np.nanmax(values)
    if extend is None:
        if values_min < tick_values[0]:
            extend = "min"
        if values_max > tick_values[-1]:
            extend = "max"
        if (values_max > tick_values[-1]) & (values_min < tick_values[0]):
            extend="both"
   
        
    axins = inset_axes(
    ax,
    width="1.5%",  
    height="80%",  
    loc="lower left",
    bbox_to_anchor=(1.05, 0., 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0,
)
    if decimals is None:
        decimals = "%0.0f"
    cb = fig.colorbar(heatmap, cax=axins, shrink=0.5, ticks=tick_values, format=decimals, extend=extend, anchor=(0, anchor))


    cb.ax.tick_params(labelsize=20)
    if title is not None:
        cb.ax.set_title(title, fontsize=25)
        
    # Add the special shading
    if special_value is not None:
        special_overlay = np.where(values == special_value, 1, np.nan)
        hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['/'], colors="silver", linewidth=0.15, transform=ccrs.PlateCarree())
        
    if special_value_2 is not None:
        special_overlay = np.where(values == special_value_2, 1, np.nan)
        hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['\\'], colors="gold", linewidth=0.15, transform=ccrs.PlateCarree())

    # set the extent and aspect ratio of the plot
    ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())
    aspect_ratio = (latitudes.max() - latitudes.min()) / (longitudes.max() - longitudes.min())
    ax.set_aspect(1)

    # add axis labels and a title
    ax.set_xlabel('Longitude', fontsize=30)
    ax.set_ylabel('Latitude', fontsize=30)
    borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m', facecolor='none')
    ax.add_feature(borders, edgecolor='gray', linestyle=':')
    ax.coastlines()
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('top')
    ax.coastlines()
    if graphmarking is not None:
        ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')
     
    hatch_patches=[]
    if special_value is not None and hatch_label is not None:
        hatch_patch_1 = Patch(facecolor='silver', edgecolor='black', hatch="/", label=hatch_label)
        hatch_patches.append(hatch_patch_1)

    if hatch_label_2 is not None:
        hatch_patch_2 = Patch(facecolor='gold', edgecolor='black', hatch="/", label=hatch_label_2)
        hatch_patches.append(hatch_patch_2)

    if hatch_patches:
        ax.legend(handles=hatch_patches, loc='lower left', fontsize=20)
    
    if filename is not None:
        plt.savefig(filename + ".png", bbox_inches="tight")
    
    return 


# Read in PEM and Alkaline cheapest
PEM_cheapest = results_model.PEM_cheapest_sf
ALK_cheapest = results_model.ALK_cheapest_sf

# Read in PEM values
PEM_solar = results_model.PEM_100
PEM_wind = results_model.PEM_0

# Read in PEM values
ALK_solar = results_model.ALK_100
ALK_wind = results_model.ALK_0

# Calculate CAPEX reductions to get to US3/kg
PEM_cheapest = results_model.calculate_total_capex_reduction(PEM_cheapest, 3)
ALK_cheapest = results_model.calculate_total_capex_reduction(ALK_cheapest, 3)

# Calculate CAPEX reductions to get to US3/kg
PEM_solar  = results_model.calculate_total_capex_reduction(PEM_solar , 3)
PEM_wind  = results_model.calculate_total_capex_reduction(PEM_wind , 3)

# Calculate CAPEX reductions to get to US3/kg
ALK_solar = results_model.calculate_total_capex_reduction(ALK_solar, 3)
ALK_wind = results_model.calculate_total_capex_reduction(ALK_wind, 3)



def produce_figures_5_6(data, wind_data, solar_data, output_folder, technology):
    
    
    # Plot figure 5
    plot_data_shading(solar_data['capex_reduction_3'].sel(latitude=slice(-60, 90)), solar_data.latitude.sel(latitude=slice(-60, 90)), solar_data.longitude, tick_values=[0.01, 25, 50, 75, 100], cmap="YlOrRd", graphmarking="a", title="CAPEX\n Reductions\n to reach\n US$3/kg (%)\n", filename=output_folder + "Solar_CAPEX_"+technology, special_value=999, hatch_label="Locations already \n below US$3/kg", extend="neither")
    plot_data_shading(wind_data['capex_reduction_3'].sel(latitude=slice(-60, 90)), wind_data.latitude.sel(latitude=slice(-60, 90)), wind_data.longitude, tick_values=[0.01, 25, 50, 75, 100], cmap="YlGnBu", graphmarking="b", title="CAPEX\n Reductions\n to reach\n US$3/kg (%)\n", filename=output_folder + "Wind_CAPEX_"+technology, special_value=999, hatch_label="Locations already \n below US$3/kg", extend="neither")
    plot_data_shading(data['capex_reduction_3'].sel(latitude=slice(-60, 90)), data.latitude.sel(latitude=slice(-60, 90)), data.longitude, tick_values=[0.01, 25, 50, 75, 100], cmap="PuRd", graphmarking="c", title="CAPEX\n Reductions\n to reach\n US$3/kg (%)\n", filename=output_folder + "Hybrid_CAPEX_"+technology, special_value=999, hatch_label="Locations already \n below US$3/kg", extend="neither")
    


# Call function
output_folder = "/Users/lukehatton/Green Hydrogen 2024/PLOTS/FINAL/CAPEX/"
produce_figures_5_6(PEM_cheapest, PEM_wind, PEM_solar, output_folder, "PEM")
produce_figures_5_6(ALK_cheapest, ALK_wind, ALK_solar, output_folder, "ALK")



import cartopy.crs as ccrs
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cartopy.feature as cfeature
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_data_shading(values, latitudes, longitudes, anchor=None, filename=None, increment=None, title=None, tick_values=None, cmap=None, extend=None, graphmarking=None, special_value=None, hatch_label=None, hatch_label_2=None, special_value_2=None, center_norm=None, decimals=None):      
    
    # create the heatmap using pcolormesh
    if anchor is None:
        anchor = 0.355
    fig = plt.figure(figsize=(30, 15), facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    if center_norm is None:
        heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.Normalize(vmin=tick_values[0], vmax=tick_values[-1]), transform=ccrs.PlateCarree(), cmap=cmap)
    else:
        heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.SymLogNorm(vmin = tick_values[0], vmax=tick_values[-1], linscale
=1, linthresh=1), transform=ccrs.PlateCarree(), cmap=cmap)
    
    # Check if there is a need for extension
    values_min = np.nanmin(values)
    values_max = np.nanmax(values)
    if extend is None:
        if values_min < tick_values[0]:
            extend = "min"
        if values_max > tick_values[-1]:
            extend = "max"
        if (values_max > tick_values[-1]) & (values_min < tick_values[0]):
            extend="both"
   
        
    axins = inset_axes(
    ax,
    width="1.5%",  
    height="80%",  
    loc="lower left",
    bbox_to_anchor=(1.05, 0., 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0,
)
    if decimals is None:
        decimals = "%0.0f"
    cb = fig.colorbar(heatmap, cax=axins, shrink=0.5, ticks=tick_values, format=decimals, extend=extend, anchor=(0, anchor))


    cb.ax.tick_params(labelsize=20)
    if title is not None:
        cb.ax.set_title(title, fontsize=25)
        
    # Add the special shading
    if special_value is not None:
        special_overlay = np.where(values == special_value, 1, np.nan)
        hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['/'], colors="silver", linewidth=0.15, transform=ccrs.PlateCarree())
        
    if special_value_2 is not None:
        special_overlay = np.where(values == special_value_2, 1, np.nan)
        hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['\\'], colors="gold", linewidth=0.15, transform=ccrs.PlateCarree())

    # set the extent and aspect ratio of the plot
    ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())
    aspect_ratio = (latitudes.max() - latitudes.min()) / (longitudes.max() - longitudes.min())
    ax.set_aspect(1)

    # add axis labels and a title
    ax.set_xlabel('Longitude', fontsize=30)
    ax.set_ylabel('Latitude', fontsize=30)
    borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m', facecolor='none')
    ax.add_feature(borders, edgecolor='gray', linestyle=':')
    ax.coastlines()
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('top')
    ax.coastlines()
    if graphmarking is not None:
        ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')
     
    hatch_patches=[]
    if special_value is not None and hatch_label is not None:
        hatch_patch_1 = Patch(facecolor='silver', edgecolor='black', hatch="/", label=hatch_label)
        hatch_patches.append(hatch_patch_1)

    if hatch_label_2 is not None:
        hatch_patch_2 = Patch(facecolor='gold', edgecolor='black', hatch="/", label=hatch_label_2)
        hatch_patches.append(hatch_patch_2)

    if hatch_patches:
        ax.legend(handles=hatch_patches, loc='lower left', fontsize=20)
    
    if filename is not None:
        plt.savefig(filename + ".png", bbox_inches="tight")
    
    return 


# Read in PEM and Alkaline cheapest
PEM_cheapest = results_model.PEM_cheapest_sf
ALK_cheapest = results_model.ALK_cheapest_sf

# Read in PEM values
PEM_solar = results_model.PEM_100
PEM_wind = results_model.PEM_0

# Read in PEM values
ALK_solar = results_model.ALK_100
ALK_wind = results_model.ALK_0

# Calculate CAPEX reductions to get to US3/kg
PEM_cheapest = results_model.calculate_required_waccs(PEM_cheapest, 3, "Hybrid")
ALK_cheapest = results_model.calculate_required_waccs(ALK_cheapest, 3, "Hybrid")

# Calculate CAPEX reductions to get to US3/kg
PEM_solar  = results_model.calculate_required_waccs(PEM_solar, 3, "Solar")
PEM_wind  = results_model.calculate_required_waccs(PEM_wind, 3, "Wind")

# Calculate CAPEX reductions to get to US3/kg
ALK_solar  = results_model.calculate_required_waccs(ALK_solar, 3, "Solar")
ALK_wind  = results_model.calculate_required_waccs(ALK_wind, 3, "Wind")

    
def produce_figures_6(data, wind_data, solar_data, output_folder, technology): 
    # Plot figure 5
    plot_data_shading(solar_data['Required_WACC'].sel(latitude=slice(-60, 90)), solar_data.latitude.sel(latitude=slice(-60, 90)), solar_data.longitude, tick_values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cmap="YlOrRd", graphmarking="a", title="Nominal\nWACC\nfor US$3/kg \nby 2030 (%)\n", filename=output_folder + "Solar_WACC_"+technology, special_value=999, hatch_label="LCOH exceeds \nUS$3/kg even at\n 0% WACC")
    plot_data_shading(wind_data['Required_WACC'].sel(latitude=slice(-60, 90)), wind_data.latitude.sel(latitude=slice(-60, 90)), wind_data.longitude, tick_values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cmap="YlGnBu", graphmarking="b", title="Nominal\nWACC\nfor US$3/kg \nby 2030 (%)\n", filename=output_folder + "Wind_WACC_"+technology, special_value=999, hatch_label="LCOH exceeds \nUS$3/kg even at\n 0% WACC")
    plot_data_shading(data['Required_WACC'].sel(latitude=slice(-60, 90)), data.latitude.sel(latitude=slice(-60, 90)), data.longitude, tick_values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cmap="PuRd", graphmarking="c", title="Nominal\nWACC\nfor US$3/kg \nby 2030 (%)\n", filename=output_folder + "Hybrid_WACC_"+technology, special_value=999, hatch_label="LCOH exceeds \nUS$3/kg even at\n 0% WACC")
    


# Call function
output_folder = "/Users/lukehatton/Green Hydrogen 2024/PLOTS/FINAL/WACC/"
produce_figures_6(PEM_cheapest, PEM_wind, PEM_solar, output_folder, "PEM")
produce_figures_6(ALK_cheapest, ALK_wind, ALK_solar, output_folder, "ALK")


import cartopy.crs as ccrs
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cartopy.feature as cfeature
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_data_shading(values, latitudes, longitudes, anchor=None, filename=None, increment=None, title=None, tick_values=None, cmap=None, extend=None, graphmarking=None, special_value=None, hatch_label=None, hatch_label_2=None, special_value_2=None, center_norm=None):      
    
    # create the heatmap using pcolormesh
    if anchor is None:
        anchor = 0.355
    fig = plt.figure(figsize=(30, 15), facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
    # Add the special shading
    if special_value is not None:
        special_overlay = np.where(values == special_value, 1, np.nan)
        hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['/'], colors="silver", linewidth=0.15, transform=ccrs.PlateCarree())
        
    if special_value_2 is not None:
        special_overlay = np.where(values == special_value_2, 1, np.nan)
        hatching = ax.contourf(longitudes, latitudes, special_overlay, hatches=['\\'], colors="gold", linewidth=0.15, transform=ccrs.PlateCarree())

    # set the extent and aspect ratio of the plot
    ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())
    aspect_ratio = (latitudes.max() - latitudes.min()) / (longitudes.max() - longitudes.min())
    ax.set_aspect(1)

    # add axis labels and a title
    ax.set_xlabel('Longitude', fontsize=30)
    ax.set_ylabel('Latitude', fontsize=30)
    borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m', facecolor='none')
    ax.add_feature(borders, edgecolor='gray', linestyle=':')
    ax.coastlines()
    ax.coastlines()
    if graphmarking is not None:
        ax.text(0.02, 0.94, graphmarking, transform=ax.transAxes, fontsize=20, fontweight='bold')
     
    hatch_patches=[]
    if special_value is not None and hatch_label is not None:
        hatch_patch_1 = Patch(facecolor='silver', edgecolor='black', hatch="/", label=hatch_label)
        hatch_patches.append(hatch_patch_1)

    if hatch_label_2 is not None:
        hatch_patch_2 = Patch(facecolor='gold', edgecolor='black', hatch="/", label=hatch_label_2)
        hatch_patches.append(hatch_patch_2)

    if hatch_patches:
        ax.legend(handles=hatch_patches, loc='lower left', fontsize=20)
    
    if filename is not None:
        plt.savefig(filename + ".png", bbox_inches="tight")
    
    return 

# Get cheapest values
cheapest_solar = xr.where(np.isnan(results_model.PEM_100['levelised_cost'])==True, np.nan, xr.where(results_model.PEM_100['levelised_cost'] < results_model.ALK_100['levelised_cost'],1111,999))
cheapest_wind = xr.where(np.isnan(results_model.PEM_0['levelised_cost'])==True, np.nan, xr.where(results_model.PEM_0['levelised_cost'] < results_model.ALK_0['levelised_cost'],111,999))
cheapest_hybrid = xr.where(np.isnan(results_model.PEM_100['levelised_cost'])==True, np.nan, xr.where(cheapest_PEM['levelised_cost'] < cheapest_ALK['levelised_cost'],111,999))

# Plt data
plot_data_shading(cheapest_solar, cheapest_solar.latitude, cheapest_solar.longitude, special_value=111, hatch_label = "PEM is cheaper", special_value_2 = 999, hatch_label_2 = "ALK is cheaper")
plot_data_shading(cheapest_wind, cheapest_wind.latitude, cheapest_wind.longitude, special_value=111, hatch_label = "PEM is cheaper", special_value_2 = 999, hatch_label_2 = "ALK is cheaper")
plot_data_shading(cheapest_hybrid, cheapest_hybrid.latitude, cheapest_hybrid.longitude, special_value=111, hatch_label = "PEM is cheaper", special_value_2 = 999, hatch_label_2 = "ALK is cheaper")