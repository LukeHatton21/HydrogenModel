import numpy as np
import pandas as pd
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)


class AbatementEstimator:
    def __init__(self, ember_data, IFI_data, GEM_data, country_grids, country_mapping):
        """
        Initialise the abatement estimator with all data needed to compute
        location-specific power-system and end-use abatement metrics.

        Parameters
        ----------
        ember_data : str
            Path to Ember yearly dataset CSV.
        IFI_data : str
            Path to IFI grid factors CSV.
        GEM_data : str
            Path to Global Energy Monitor power project CSV.
        country_grids : xarray.Dataset
            Country index grid in model spatial resolution (must include 'country').
        country_mapping : str
            Path to country mapping CSV (must include country codes and index mapping).
        """
        self.ember_data = pd.read_csv(ember_data)
        self.lifetime = 20
        self.year = 2023
        self.IFI_data = pd.read_csv(IFI_data)
        self.gem_data = pd.read_csv(GEM_data)
        self.country_grids = country_grids
        self.country_mapping = pd.read_csv(country_mapping)

    def calculate_IFI_abatement(self, IFI_data, results, country_emissions, future=None):
        """
        Add IFI-based emissions factor outputs and hydrogen-normalised abatement to results.

        Parameters
        ----------
        IFI_data : xarray.Dataset
            Spatial IFI dataset containing emissions-factor components.
        results : xarray.Dataset
            Model output dataset containing at least electricity_production and hydrogen_production.
        country_emissions : xarray.DataArray
            Spatial country-level total emissions aligned with results grid.
        future : str or None, optional
            If provided, writes outputs to future-labelled variables.

        Returns
        -------
        xarray.Dataset
            Input results with added IFI emissions metrics and abatement variables.
        """
        tCO2_MW = IFI_data["IFI_Emissions_Factor"] / 1e+6 * results['electricity_production'] * self.lifetime
        if future is None:
            results["abatement_MW"] = tCO2_MW / results["hydrogen_production"] / self.lifetime
        else:
            results["future_abatement_MW"] = tCO2_MW / results["hydrogen_production"] / self.lifetime

        if future is None:
            results["IFI_Emissions_Factor"] = IFI_data["IFI_Emissions_Factor"]
            results["Future_Build_Margin"] = IFI_data["Build_Margin"]
        else:
            results["IFI_Emissions_Factor_Future"] = IFI_data["IFI_Emissions_Factor"]
            results["Build_Margin"] = IFI_data["Build_Margin_Adjusted"]

        results["Operating_Margin"] = IFI_data["VRE_Energy"]
        results["Additionality_Factor"] = IFI_data["Additionality_Factor"]
        results["emissions"] = country_emissions
        results["grid_intensity"] = IFI_data["Electricity_CI"]
        return results

    def extract_emissions(self, ember_data):
        """
        Map country-level total power-sector emissions from Ember to the model grid.

        Parameters
        ----------
        ember_data : pandas.DataFrame
            Ember dataset loaded in memory.

        Returns
        -------
        xarray.Dataset
            Gridded dataset with emissions variable by latitude/longitude.
        """
        ci_data = ember_data.loc[(ember_data["Variable"] == "Total emissions") & (ember_data["Year"] == self.year)]
        ci_data_subset = ci_data[["Country code", "Value"]]
        ci_data_mapping = pd.merge(self.country_mapping, ci_data_subset, how="left", on="Country code")

        country_grids = self.country_grids.rename({"country": "index"})
        country_df = country_grids.to_dataframe().reset_index()
        country_df = country_df.merge(ci_data_mapping.rename(columns={"Value": "emissions"}), how="left", on="index")
        country_ds = country_df.drop_duplicates(subset=["latitude", "longitude"]).set_index(["latitude", "longitude"]).to_xarray()
        country_ds = country_ds.assign_coords({"latitude": country_ds.latitude, "longitude": country_ds.longitude})
        return country_ds

    def extract_ci(self):
        """
        Build a country-level electricity carbon intensity table with simple year-to-year gap filling.

        Gap filling uses previous available values from sequential years.

        Returns
        -------
        pandas.DataFrame
            Country-code table with column 'Electricity_CI'.
        """
        def gap_fill_ci(data, ember_data, year):
            """
            Fill missing carbon intensity values in `data` using a target `year` slice from Ember.
            """
            generation_subset = ember_data.loc[(ember_data["Unit"] == "gCO2/kWh") & (ember_data["Year"] == year)]
            previous_data = generation_subset[["Country code", "Value"]]
            previous_data = previous_data.rename(columns={"Value": "Current_Value"})
            data = data.merge(previous_data, how="left", on=["Country code"])
            data['Current_Value'] = data['Current_Value'].fillna(data['Value'])
            data = data.drop(columns="Value")
            data = data.rename(columns={"Current_Value": "Value"})
            data["Year"] = year
            return data

        ember_data = self.ember_data
        ci_data = ember_data.loc[(ember_data["Unit"] == "gCO2/kWh") & (ember_data["Year"] == 2021)]
        ci_data = gap_fill_ci(ci_data, ember_data, 2022)
        ci_data = gap_fill_ci(ci_data, ember_data, 2023)
        ci_data = gap_fill_ci(ci_data, ember_data, 2024)
        ci_data = gap_fill_ci(ci_data, ember_data, 2025)

        extracted_data = ci_data[["Country code", "Value"]].dropna(subset="Country code").rename(columns={"Value": "Electricity_CI"})
        return extracted_data

    def extract_IFI(self, fraction, additionality=None):
        """
        Construct gridded IFI emissions-factor dataset from national IFI and Ember inputs.

        Parameters
        ----------
        fraction : float
            Weighting applied to build margin in IFI emissions factor calculation.
        additionality : any, optional
            If not None, applies additionality adjustment using GEM fossil/renewable pipeline.

        Returns
        -------
        xarray.Dataset
            Spatial IFI dataset including emissions factor and intermediate components.
        """
        ifi_data = self.IFI_data[["Country code", "VRE_Energy", "Build_Margin"]].copy()

        if additionality is not None:
            gem = self.gem_data[["ISO", "Type", "Capacity (MW)"]]
            gem["Capacity (MW)"] = gem["Capacity (MW)"].astype(float)
            grouped_gem = gem.groupby(["ISO", "Type"])["Capacity (MW)"].agg("sum").reset_index()

            renewables_gem = grouped_gem.loc[grouped_gem["Type"].isin(["solar", "wind"])].rename(columns={"Capacity (MW)": "Renewable_Capacity"})
            renewables_gem = renewables_gem[["ISO", "Renewable_Capacity"]].groupby("ISO").agg("sum").reset_index()
            oil_gas_gem = grouped_gem.loc[grouped_gem["Type"] == "oil/gas"].rename(columns={"Capacity (MW)": "Fossil_Capacity"})

            merged_gem = pd.merge(renewables_gem, oil_gas_gem, how="outer", on="ISO")
            merged_gem["Fossil_Renewable"] = merged_gem["Fossil_Capacity"] / merged_gem["Renewable_Capacity"]
            merged_gem["Country code"] = merged_gem["ISO"]

            ifi_data = ifi_data.merge(merged_gem[["Country code", "Fossil_Renewable"]], how="left", on="Country code")
            ifi_data = ifi_data.merge(merged_gem[["Country code", "Fossil_Capacity"]], how="left", on="Country code")
            ifi_data["Additionality_Factor"] = ifi_data["Fossil_Renewable"].clip(upper=1, lower=0.1).fillna(1)
        else:
            ifi_data.loc[:, "Additionality_Factor"] = 1

        ember_CI = self.extract_ci()
        ifi_data = ifi_data.merge(ember_CI, how="left", on="Country code")

        ifi_data.loc[ifi_data["VRE_Energy"] < 10, "VRE_Energy"] = ifi_data["Electricity_CI"]
        ifi_data.loc[ifi_data["Build_Margin"] < 10, "Build_Margin"] = np.nanmedian(ifi_data["Build_Margin"])

        ifi_data["Build_Margin_Adjusted"] = ifi_data["Build_Margin"]

        if fraction == 0.25:
            ifi_data["IFI_Emissions_Factor"] = ifi_data["Additionality_Factor"] * (
                ifi_data["VRE_Energy"] * (1 - fraction) + fraction * ifi_data["Build_Margin"]
            )
        else:
            ifi_data["IFI_Emissions_Factor"] = ifi_data["Additionality_Factor"] * (
                ifi_data["VRE_Energy"] * (1 - fraction) + fraction * ifi_data["Build_Margin_Adjusted"]
            )

        ifi_data_mapping = pd.merge(self.country_mapping, ifi_data, how="left", on="Country code")

        country_grids = self.country_grids.rename({"country": "index"})
        country_df = country_grids.to_dataframe().reset_index()
        country_df = country_df.merge(ifi_data_mapping, how="left", on="index")
        country_ds = country_df.drop_duplicates(subset=["latitude", "longitude"]).set_index(["latitude", "longitude"]).to_xarray()
        country_ds = country_ds.assign_coords({"latitude": country_ds.latitude, "longitude": country_ds.longitude})
        return country_ds

    def heat_pump_abatement_cost(self, elec_gas_price_ratio, cop_hp, eff_gb, hp_om_ratio=0.02, gb_om_ratio=0.02,
                                 hp_installed_cost=6000, gb_installed_cost=2000, discount_rate=0.08,
                                 lifetime_years=15, gas_price=0.06, ef_elec=0, ef_gas=0.203, flh=8760):
        """
        Compute heat-pump vs gas-boiler abatement economics and emissions metrics.

        Returns
        -------
        dict
            Includes abatement cost ($/tCO2), emissions saved per renewable electricity unit,
            and intermediate cost/emissions terms.
        """
        r = discount_rate
        n = lifetime_years
        crf = r * (1 + r) ** n / ((1 + r) ** n - 1)

        hp_annual_fixed = hp_installed_cost * crf + hp_installed_cost * hp_om_ratio
        gb_annual_fixed = gb_installed_cost * crf + gb_installed_cost * gb_om_ratio

        hp_fixed_per_kwh_heat = hp_annual_fixed / flh
        gb_fixed_per_kwh_heat = gb_annual_fixed / flh

        p_gas = gas_price
        p_elec = elec_gas_price_ratio * p_gas

        hp_var_per_kwh_heat = p_elec / cop_hp
        gb_var_per_kwh_heat = p_gas / eff_gb

        lcoh_hp = hp_fixed_per_kwh_heat + hp_var_per_kwh_heat
        lcoh_gb = gb_fixed_per_kwh_heat + gb_var_per_kwh_heat

        em_hp = ef_elec / cop_hp
        em_gb = ef_gas / eff_gb

        delta_cost = lcoh_hp - lcoh_gb
        delta_emissions = em_gb - em_hp

        ren_elec_used_kwh_per_kwh_heat = 1 / cop_hp
        em_saved_per_kwh_ren = (delta_emissions / ren_elec_used_kwh_per_kwh_heat
                                if abs(ren_elec_used_kwh_per_kwh_heat) >= 1e-12 else float("nan"))
        em_saved_per_mwh_ren = em_saved_per_kwh_ren * 1000

        if abs(delta_emissions) < 1e-12:
            abatement_cost = float("inf") if delta_cost > 0 else float("-inf") if delta_cost < 0 else float("nan")
            interpretation = "Undefined (near-zero emissions difference)."
        else:
            abatement_cost = (delta_cost / delta_emissions) * 1000
            interpretation = ("Negative abatement cost (cost-saving emissions reduction)."
                              if abatement_cost < 0 else
                              "Positive abatement cost (paying per tCO2 reduced).")

        return {
            "abatement_cost_usd_per_tco2": abatement_cost,
            "interpretation": interpretation,
            "lcoh_hp_usd_per_kwh_heat": lcoh_hp,
            "lcoh_gb_usd_per_kwh_heat": lcoh_gb,
            "em_hp_kgco2_per_kwh_heat": em_hp,
            "em_gb_kgco2_per_kwh_heat": em_gb,
            "delta_cost_usd_per_kwh_heat": delta_cost,
            "delta_emissions_kgco2_per_kwh_heat": delta_emissions,
            "ren_elec_used_kwh_per_kwh_heat": ren_elec_used_kwh_per_kwh_heat,
            "emissions_saved_kgco2_per_kwh_ren_elec": em_saved_per_kwh_ren,
            "emissions_saved_kgco2_per_mwh_ren_elec": em_saved_per_mwh_ren,
        }

    def ev_abatement_cost_ttw(self, ev_installed_cost, ice_installed_cost, elec_liquid_price_ratio,
                              wheel_energy_demand_kwh_per_km, eta_ev_ttw, eta_ice_ttw, fuel_type="petrol",
                              ev_om_ratio=0.02, ice_om_ratio=0.02, discount_rate=0.08, lifetime_years=12,
                              annual_km=15000, liquid_fuel_price_per_liter=1.2,
                              lhv_petrol_kwh_per_liter=8.9, lhv_diesel_kwh_per_liter=9.8,
                              ci_elec_kg_per_kwh=0.0, ci_petrol_kg_per_kwh=0.182, ci_diesel_kg_per_kwh=0.27):
        """
        Compute EV vs ICE tank-to-wheel abatement economics and renewable-electricity emissions leverage.

        Returns
        -------
        dict
            Includes abatement cost ($/tCO2), emissions saved per renewable electricity unit,
            and detailed per-km cost/emissions breakdown.
        """
        if not (0 < eta_ev_ttw <= 1 and 0 < eta_ice_ttw <= 1):
            raise ValueError("eta_ev_ttw and eta_ice_ttw must be in (0, 1].")
        if wheel_energy_demand_kwh_per_km <= 0:
            raise ValueError("wheel_energy_demand_kwh_per_km must be > 0.")
        if annual_km <= 0:
            raise ValueError("annual_km must be > 0.")

        r = discount_rate
        n = lifetime_years
        crf = r * (1 + r) ** n / ((1 + r) ** n - 1)

        ev_annual_fixed = ev_installed_cost * crf + ev_installed_cost * ev_om_ratio
        ice_annual_fixed = ice_installed_cost * crf + ice_installed_cost * ice_om_ratio

        ev_fixed_per_km = ev_annual_fixed / annual_km
        ice_fixed_per_km = ice_annual_fixed / annual_km

        ft = fuel_type.strip().lower()
        if ft == "petrol":
            ci_liquid = ci_petrol_kg_per_kwh
            lhv_kwh_per_liter = lhv_petrol_kwh_per_liter
        elif ft == "diesel":
            ci_liquid = ci_diesel_kg_per_kwh
            lhv_kwh_per_liter = lhv_diesel_kwh_per_liter
        else:
            raise ValueError("fuel_type must be 'petrol' or 'diesel'.")

        liquid_price_per_kwh = liquid_fuel_price_per_liter / lhv_kwh_per_liter
        elec_price_per_kwh = elec_liquid_price_ratio * liquid_price_per_kwh

        ev_input_kwh_per_km = wheel_energy_demand_kwh_per_km / eta_ev_ttw
        ice_input_kwh_per_km = wheel_energy_demand_kwh_per_km / eta_ice_ttw

        ev_var_per_km = ev_input_kwh_per_km * elec_price_per_kwh
        ice_var_per_km = ice_input_kwh_per_km * liquid_price_per_kwh

        tco_ev_per_km = ev_fixed_per_km + ev_var_per_km
        tco_ice_per_km = ice_fixed_per_km + ice_var_per_km

        em_ev_per_km = ev_input_kwh_per_km * ci_elec_kg_per_kwh
        em_ice_per_km = ice_input_kwh_per_km * ci_liquid

        delta_cost = tco_ev_per_km - tco_ice_per_km
        delta_emissions = em_ice_per_km - em_ev_per_km

        ren_elec_used_kwh_per_km = ev_input_kwh_per_km
        em_saved_per_kwh_ren = (delta_emissions / ren_elec_used_kwh_per_km
                                if abs(ren_elec_used_kwh_per_km) >= 1e-12 else float("nan"))
        em_saved_per_mwh_ren = em_saved_per_kwh_ren * 1000

        if abs(delta_emissions) < 1e-12:
            abatement_cost = float("inf") if delta_cost > 0 else float("-inf") if delta_cost < 0 else float("nan")
            interpretation = "Undefined (near-zero emissions difference)."
        else:
            abatement_cost = (delta_cost / delta_emissions) * 1000
            if abatement_cost < 0 and delta_emissions > 0:
                interpretation = "Negative abatement cost (cost-saving emissions reduction)."
            elif delta_emissions > 0:
                interpretation = "Positive abatement cost (paying per tCO2 reduced)."
            else:
                interpretation = "EV increases emissions under these assumptions."

        return {
            "abatement_cost_usd_per_tco2": abatement_cost,
            "interpretation": interpretation,
            "tco_ev_usd_per_km": tco_ev_per_km,
            "tco_ice_usd_per_km": tco_ice_per_km,
            "em_ev_kgco2_per_km": em_ev_per_km,
            "em_ice_kgco2_per_km": em_ice_per_km,
            "delta_cost_usd_per_km": delta_cost,
            "delta_emissions_kgco2_per_km": delta_emissions,
            "ren_elec_used_kwh_per_km": ren_elec_used_kwh_per_km,
            "emissions_saved_kgco2_per_kwh_ren_elec": em_saved_per_kwh_ren,
            "emissions_saved_kgco2_per_mwh_ren_elec": em_saved_per_mwh_ren,
            "ev_input_kwh_per_km": ev_input_kwh_per_km,
            "ice_input_kwh_per_km": ice_input_kwh_per_km,
            "assumed_elec_price_usd_per_kwh": elec_price_per_kwh,
            "assumed_liquid_price_usd_per_kwh": liquid_price_per_kwh
        }

    def _build_end_use_abatement_factors(self):
        """
        Create scenario factors for end-use emissions saved per renewable electricity consumed.

        Returns
        -------
        pandas.DataFrame
            Table with columns Min/Mean/Max for:
            Electric vehicles, Heat pumps, and Industry.
        """
        results_df = pd.DataFrame(columns=["Min", "Mean", "Max"], index=["Electric vehicles", "Heat pumps", "Industry"])
        results_df.index.name = "End-Use"
        results_df = results_df.reset_index()

        avg_hp = self.heat_pump_abatement_cost(elec_gas_price_ratio=3, cop_hp=4.3, eff_gb=0.865)
        min_hp = self.heat_pump_abatement_cost(elec_gas_price_ratio=1.5, cop_hp=5.5, eff_gb=0.81)
        max_hp = self.heat_pump_abatement_cost(elec_gas_price_ratio=4.5, cop_hp=3.1, eff_gb=0.91)
        results_df.loc[results_df["End-Use"] == "Heat pumps", "Mean"] = avg_hp["emissions_saved_kgco2_per_kwh_ren_elec"]
        results_df.loc[results_df["End-Use"] == "Heat pumps", "Max"] = min_hp["emissions_saved_kgco2_per_kwh_ren_elec"]
        results_df.loc[results_df["End-Use"] == "Heat pumps", "Min"] = max_hp["emissions_saved_kgco2_per_kwh_ren_elec"]

        avg_ind = self.heat_pump_abatement_cost(elec_gas_price_ratio=3, cop_hp=2, eff_gb=0.95)
        min_ind = self.heat_pump_abatement_cost(elec_gas_price_ratio=1.5, cop_hp=3, eff_gb=0.95)
        max_ind = self.heat_pump_abatement_cost(elec_gas_price_ratio=4.5, cop_hp=0.95, eff_gb=0.95)
        results_df.loc[results_df["End-Use"] == "Industry", "Mean"] = avg_ind["emissions_saved_kgco2_per_kwh_ren_elec"]
        results_df.loc[results_df["End-Use"] == "Industry", "Max"] = min_ind["emissions_saved_kgco2_per_kwh_ren_elec"]
        results_df.loc[results_df["End-Use"] == "Industry", "Min"] = max_ind["emissions_saved_kgco2_per_kwh_ren_elec"]

        avg_ev = self.ev_abatement_cost_ttw(
            ev_installed_cost=33000, ice_installed_cost=30000, elec_liquid_price_ratio=1.5,
            wheel_energy_demand_kwh_per_km=0.1, eta_ev_ttw=0.65, eta_ice_ttw=0.2
        )
        min_ev = self.ev_abatement_cost_ttw(
            ev_installed_cost=33000, ice_installed_cost=30000, elec_liquid_price_ratio=1,
            wheel_energy_demand_kwh_per_km=0.1, eta_ev_ttw=0.8, eta_ice_ttw=0.14
        )
        max_ev = self.ev_abatement_cost_ttw(
            ev_installed_cost=33000, ice_installed_cost=30000, elec_liquid_price_ratio=2,
            wheel_energy_demand_kwh_per_km=0.1, eta_ev_ttw=0.5, eta_ice_ttw=0.26
        )
        results_df.loc[results_df["End-Use"] == "Electric vehicles", "Mean"] = avg_ev["emissions_saved_kgco2_per_kwh_ren_elec"]
        results_df.loc[results_df["End-Use"] == "Electric vehicles", "Max"] = min_ev["emissions_saved_kgco2_per_kwh_ren_elec"]
        results_df.loc[results_df["End-Use"] == "Electric vehicles", "Min"] = max_ev["emissions_saved_kgco2_per_kwh_ren_elec"]

        return results_df

    def _add_end_use_abatement_layers(self, results_with_abatement, factors_df, case="Mean"):
        """
        Add end-use abatement estimates (EV, heat pumps, industry) to the gridded results.

        Parameters
        ----------
        results_with_abatement : xarray.Dataset
            Gridded model results containing renewable_electricity and hydrogen_production.
        factors_df : pandas.DataFrame
            End-use factors table from `_build_end_use_abatement_factors`.
        case : str, default "Mean"
            Which scenario column to use ("Min", "Mean", or "Max").

        Returns
        -------
        xarray.Dataset
            Input dataset with added abatement_EV, abatement_HP, abatement_IND variables.
        """
        ev_factor = float(factors_df.loc[factors_df["End-Use"] == "Electric vehicles", case].values[0])
        hp_factor = float(factors_df.loc[factors_df["End-Use"] == "Heat pumps", case].values[0])
        ind_factor = float(factors_df.loc[factors_df["End-Use"] == "Industry", case].values[0])

        results_with_abatement["abatement_EV"] = (
            results_with_abatement["renewable_electricity"] * ev_factor
            / results_with_abatement["hydrogen_production"] / 1000
        )
        results_with_abatement["abatement_HP"] = (
            results_with_abatement["renewable_electricity"] * hp_factor
            / results_with_abatement["hydrogen_production"] / 1000
        )
        results_with_abatement["abatement_IND"] = (
            results_with_abatement["renewable_electricity"] * ind_factor
            / results_with_abatement["hydrogen_production"] / 1000
        )
        return results_with_abatement

    def calculate_abatement_potential(self, results):
        """
        Run full abatement  pipeline for a gridded results dataset.

        Steps
        -----
        1) Build IFI factors for present and future scenarios.
        2) Map country emissions to grid.
        3) Add IFI abatement and emissions-factor fields.
        4) Add end-use abatement estimates (EV/HP/Industry, mean case).

        Parameters
        ----------
        results : xarray.Dataset
            Model results dataset for one technology/supply case.

        Returns
        -------
        xarray.Dataset
            Results enriched with IFI and end-use abatement variables.
        """
        country_IFI = self.extract_IFI(fraction=0.25)
        country_IFI_future = self.extract_IFI(fraction=0.75)

        country_emissions = self.extract_emissions(self.ember_data)["emissions"]

        results_with_abatement = self.calculate_IFI_abatement(country_IFI, results, country_emissions)
        results_with_abatement = self.calculate_IFI_abatement(
            country_IFI_future, results_with_abatement, country_emissions, "Future"
        )

        end_use_factors = self._build_end_use_abatement_factors()
        results_with_abatement = self._add_end_use_abatement_layers(
            results_with_abatement, end_use_factors, case="Mean"
        )

        return results_with_abatement