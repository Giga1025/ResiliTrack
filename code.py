import pandas as pd
import numpy as np
import pyreadr
import geopandas as gpd
import geopandas as gpd
import folium
from shapely.geometry import mapping
import matplotlib.pyplot as plt
from functools import reduce
import os
from scipy.signal import find_peaks
# from IPython.display import display
"""
Census blocks are uniquely defined by a 15-character geocode: SSCCCTTTTTTBBBB where:
SS state FIPS code . CCC county FIPS code . TTTTTT census tract code . BBBB census block code
 
Port Arthur geocode: SS - 48, FIPS - 245, census tract - 

1. origin_census_block_group: the source cbg -----  cbg where the people started
2. device_count: the number of devices residing in the origin cbg ------ total number of ppl in that cbg on that day
3. destination_cbg: the destination cbg ------ the cbg where they went to
4. destination_device_count: the raw visits from source cbg to destination cbg ----- total number ppl from origin cbg who went the dest cbg
5. year: year ---- here its just 2019
6. uid: the specific date of the year 2019 (1 indicates 01/01/2019) ----- heres its from 0 to 365 
7. from_cnt: the source county ---- fron county code
8. to_cnt: the destination county ----  to county code
"""


def load_rdata(filepath):
    """Load .RData file and return the main dataframe."""
    data = pyreadr.read_r(filepath)
    for key in data.keys():
        print(key)
        return data[key]

def prepare_mobility_data(df):
    """Add 'date' column and sort by date."""
    df["date"] = pd.to_datetime('2019-01-01') + pd.to_timedelta(df['uid'] - 1, unit='D')
    return df.sort_values(by='date')

def check_cbgs_presence(df, target_cbg="482459900000"):
    """Check if a specific CBG is present in the mobility data."""
    unique_cbgs = df["origin_census_block_group"].unique()
    if target_cbg in unique_cbgs:
        print("yes")
    else:
        print("no")

def load_county_cbgs(shapefile_path, county_fips):
    """Load block groups for specified counties."""
    cbgs = gpd.read_file(shapefile_path)
    return cbgs[cbgs["COUNTYFP"].isin(county_fips)].reset_index(drop=True)

def load_port_arthur_geometry(places_shapefile_path):
    """Load Port Arthur's geometry from the places shapefile."""
    places = gpd.read_file(places_shapefile_path)
    return places[places["NAME"] == "Port Arthur"]

def get_port_arthur_cbgs(cbgs, port_arthur_geom, exclude_cbg="482459900000"):
    """Find CBGs intersecting Port Arthur geometry, optionally excluding one."""
    cbgs = cbgs.to_crs(epsg=4326)
    port_arthur_geom = port_arthur_geom.to_crs(epsg=4326)
    pa_cbgs = cbgs[cbgs.intersects(port_arthur_geom.geometry.iloc[0])].copy()
    cbg_list = pa_cbgs["GEOID"].tolist()
    if exclude_cbg in cbg_list:
        cbg_list.remove(exclude_cbg)
    return cbg_list

def compute_baseline(series, start="2019-07-01", end="2019-08-31"):
    window = series[(series.index >= start) & (series.index <= end)]
    return window.mean()


def get_t0(date = "2019-09-17"):
    return pd.to_datetime(date)


def get_t1(series, t0, baseline, max_recovery_days=60):
    first_dip_tab = series[(series.index >= t0) & (series <= baseline)]

    if not first_dip_tab.empty:
        first_dip = first_dip_tab.index[0]
        recovery_window = series[(series.index >= first_dip) & (series.index <= first_dip + pd.to_timedelta(f"{max_recovery_days} days"))]

        recovered = recovery_window[recovery_window >= baseline]   # These are the values that are greater than basline after first dip
        not_recovered = recovery_window[recovery_window < baseline]
        values = not_recovered.values
        peaks,_ = find_peaks(values)
        if not recovered.empty:
            return recovered.index[0], baseline
        else:
            # Never returned to baseline → assign max point as new t1
            below_baseline = recovery_window[recovery_window < baseline]
            values = below_baseline.values
            peaks, _ = find_peaks(values)
            if len(peaks) > 0:
                peak_values = values[peaks]
                sorted_indices = np.argsort(peak_values)[::-1]
                best_peak = peaks[sorted_indices[0]]
                if best_peak == 0:
                    if len(sorted_indices) > 1:
                        second_best_peak = peaks[sorted_indices[1]]
                        return below_baseline.index[second_best_peak], values[second_best_peak]
                    else:
                        # Only one peak and it's the first → fallback to max
                        return below_baseline.idxmax(), below_baseline.max()
                else:
                    return below_baseline.index[best_peak], values[best_peak]
            else:
                # No peaks → fallback
                return recovery_window.idxmax(), recovery_window.max()
    else:
        fallback_t1 = t0 + pd.to_timedelta(f"{max_recovery_days} days")
        return fallback_t1, series.get(fallback_t1, np.nan)
    
def get_tD(series, t0, t1):
    if pd.isna(t1) or t1 < t0:
        return None, None
    window = series[(series.index >= t0) & (series.index <= t1)]
    if window.empty:
        return None, None
    return window.idxmin(), window.min()

def compute_triangle_area(series, baseline, t0, t1):
    if pd.isna(t1) or t1 < t0:
        return 0
    window = series[(series.index >= t0) & (series.index <= t1)]
    x_days = (window.index - t0).days
    return np.trapezoid(np.maximum(baseline - window, 0), x=x_days)
    
def compute_resilience_score(triangle_area, baseline, t0, t1):
    if pd.isna(t1) or t1 < t0 or pd.isna(baseline):
        return np.nan, 0, "Invalid"
    duration_days = (t1 - t0).days
    if duration_days == 0 or pd.isna(baseline):
        return np.nan, 0
    total_area = baseline * duration_days
    resilience_score = 1 - (triangle_area / total_area)
    if resilience_score >= 0.8:
        category = "High"
    elif resilience_score >= 0.5:
        category = "Medium"
    elif resilience_score >= 0:
        category = "Low"
    else:
        category = "Negative"
    return resilience_score, total_area, category

def vulnerability(tD_value:float, tD_date, t0_value: float, t0_date):
    num = tD_value - t0_value
    deno = (tD_date - t0_date).days
    return num/deno if deno !=0 else np.nan

def robustness(t1_value:float, t1_date, tD_value:float, tD_date):
    num = t1_value - tD_value
    deno = (t1_date - tD_date).days
    return num / deno if deno !=0 else np.nan

def generate_results_dict(cbg, t0, t0_value, tD_value, tD_date, t1_value, t1_date,
                          baseline, vulnerability_score, robustness_score,
                          triangle_area, total_area, resilience_score, category):
    return {
        "cbg_id": cbg,
        "t0": t0.strftime("%Y-%m-%d"),
        "t0_value": t0_value,
        "tD": float(tD_value),
        "tD_date": tD_date,
        "t1": t1_date.strftime("%Y-%m-%d") if t1_date != 0 else None,
        "t1_value": t1_value,
        "baseline": float(baseline),
        "vulnerability": vulnerability_score,
        "robustness": robustness_score,
        "triangle_area": float(triangle_area),
        "baseline_area": float(total_area),
        "resilience_score": resilience_score,
        "category" : category
    }
def process_stream(stream, cbg, label, t0, baseline):
    t1_date, t1_value = get_t1(stream, t0, baseline)
    tD_date, tD_value = get_tD(stream, t0, t1_date)
    t0_value = stream.loc[t0]

    vuln = vulnerability(tD_value, tD_date, t0_value, t0)
    robust = robustness(t1_value, t1_date, tD_value, tD_date)

    triangle_area = compute_triangle_area(stream, baseline, t0, t1_date)
    resilience_score, total_area, category = compute_resilience_score(triangle_area, baseline, t0, t1_date)

    return generate_results_dict(
        cbg, t0, t0_value, tD_value, tD_date, t1_value, t1_date,
        baseline, vuln, robust,
        triangle_area, total_area, resilience_score, category
    )


def preprocess_cbg(cbg, pa_data):
    df = pa_data[pa_data["origin_census_block_group"] == cbg].copy()
    df['date'] = pd.to_datetime(df['date'])
    df['destination_device_count'] = pd.to_numeric(df['destination_device_count'], errors='coerce')

    own_daily = df[df['origin_census_block_group'] == df['destination_cbg']].groupby('date')['destination_device_count'].sum()
    outward_daily = df[df['origin_census_block_group'] != df['destination_cbg']].groupby('date')['destination_device_count'].sum()
    inward = pa_data[(pa_data["destination_cbg"] == cbg) & (pa_data["origin_census_block_group"] != cbg)].copy()
    inward["destination_device_count"] = pd.to_numeric(inward["destination_device_count"], errors="coerce")
    inward["date"] = pd.to_datetime(inward["date"])
    inward_daily = inward.groupby("date")["destination_device_count"].sum()

    # print(inward_daily.head())
    device_count_daily = df.groupby('date')['device_count'].first()

    mobility = pd.DataFrame({'Own': own_daily, 'Outward': outward_daily, 'Inward': inward_daily, "Overall_count": device_count_daily}).fillna(0)
    mobility = mobility.rolling(window=3, center=True).mean().interpolate().bfill().ffill()
    return mobility


def plot_resilience_categories(port_arthur_cbgs, resilience_df):
    resilience_df["cbg_id"] = resilience_df["cbg_id"].astype(str)

    port_arthur_cbgs = port_arthur_cbgs.merge(
        resilience_df[["cbg_id", "category"]],
        left_on="GEOID",
        right_on="cbg_id",
        how="left"
    )

    category_colors = {
        "High": "green",
        "Medium": "orange",
        "Low": "red"
    }
    port_arthur_cbgs["color"] = port_arthur_cbgs["category"].map(category_colors).fillna("gray")

    center = [
        port_arthur_cbgs.geometry.bounds[["miny", "maxy"]].mean().mean(),
        port_arthur_cbgs.geometry.bounds[["minx", "maxx"]].mean().mean()
    ]
    m = folium.Map(location=center, zoom_start=10)

    folium.GeoJson(
        port_arthur_cbgs,
        style_function=lambda feature: {
            "fillColor": feature["properties"]["color"],
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.6
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["GEOID", "category"],
            aliases=["CBG", "Category"]
        )
    ).add_to(m)

    return m

def plot_single_stream_with_baseline(cbg, mobility, baseline, stream, output_dir, disaster_start = None, disaster_end = None):
    stream_folder_map = {
        "Own": "own",
        "Outward": "outward",
        "Inward": "inward",
        "Overall_count": "overall"
    }

    folder = stream_folder_map[stream]
    plot_dir = os.path.join(output_dir, "plots", folder)
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(mobility.index, mobility[stream], label=f"{stream} visits", color="blue")
    plt.axhline(y=baseline, color='red', linestyle='--', label='Baseline')
    if disaster_start and disaster_end:
        plt.axvspan(disaster_start, disaster_end, color='gray', alpha=0.3, label='Disaster window')

    plt.title(f"{stream} - CBG: {cbg}")
    plt.xlabel("Date")
    plt.ylabel("Visit Count")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(plot_dir, f"{cbg}.png")
    plt.savefig(save_path)
    plt.close()

def plot_combined_streams_per_cbg(cbg, mobility, output_dir):
    plot_dir = os.path.join(output_dir, "plots", "combined_no_baseline")
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    for stream, color in zip(["Own", "Outward", "Inward", "Overall_count"], ["blue", "green", "orange", "purple"]):
        plt.plot(mobility.index, mobility[stream], label=stream, color=color)

    plt.title(f"All Streams - CBG: {cbg}")
    plt.xlabel("Date")
    plt.ylabel("Visit Count")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(plot_dir, f"{cbg}.png")
    plt.savefig(save_path)
    plt.close()




if __name__ == "__main__":
    os.makedirs("output_files",exist_ok = True)
    rdata_path = "portarthur_sd_df_2019.rdata"
    county_shapefile = "tl_2019_48_bg/tl_2019_48_bg.shp"
    places_shapefile = "tl_2019_48_place/tl_2019_48_place.shp"
    counties = ['245', '199', '361']  # Jefferson, Hardin, Orange

    pa_data = load_rdata(rdata_path)
    pa_data = prepare_mobility_data(pa_data)
    check_cbgs_presence(pa_data)

    jef_har_org_cntys = load_county_cbgs(county_shapefile, counties)
    port_arthur = load_port_arthur_geometry(places_shapefile)

    port_arthur_cbgs_list = get_port_arthur_cbgs(jef_har_org_cntys, port_arthur)
    cbg_df = pd.DataFrame(port_arthur_cbgs_list, columns=["GEOID"])
    cbg_df["GEOID"] = cbg_df["GEOID"].astype(str)
    cbg_df.to_excel("output_files/port_arthur_cbgs.xlsx", index=False)

    # print("Port Arthur CBGs:", port_arthur_cbgs_list)

    #---------------------------------------------------IMELDA-----------------------------------------------------------
    output_dir = "output_files/Imdela"
    os.makedirs(output_dir, exist_ok=True)
    results_own, results_out, results_in, results_device= [],[],[],[]

    for cbg in port_arthur_cbgs_list:
        mobility = preprocess_cbg(cbg, pa_data)
        baselines = {stream: compute_baseline(mobility[stream]) for stream in ['Own', 'Outward', 'Inward',"Overall_count"]}
        mobility = mobility[(mobility.index >= '2019-09-01') & (mobility.index <= '2019-12-31')]
        t0 = get_t0()
        disaster_start = t0
        disaster_end = pd.to_datetime("2019-09-21")

        for stream, result_list in zip(['Own', 'Outward', 'Inward', "Overall_count"], [results_own, results_out, results_in, results_device]):
            result = process_stream(mobility[stream], cbg, stream, t0, baselines[stream])
            result_list.append(result)
            plot_single_stream_with_baseline(cbg=cbg,mobility=mobility,baseline=baselines[stream],stream=stream,output_dir=output_dir, disaster_start = disaster_start, disaster_end = disaster_end)
            plot_combined_streams_per_cbg(cbg=cbg,mobility=mobility,output_dir=output_dir)

    resilience_df_own = pd.DataFrame(results_own)
    resilience_df_out = pd.DataFrame(results_out)
    resilience_df_in = pd.DataFrame(results_in)
    resilience_df_overall = pd.DataFrame(results_device)

    own = resilience_df_own[["cbg_id", "resilience_score"]].rename(columns={"resilience_score": "resilience_own"})
    out = resilience_df_out[["cbg_id", "resilience_score"]].rename(columns={"resilience_score": "resilience_outward"})
    inward = resilience_df_in[["cbg_id", "resilience_score"]].rename(columns={"resilience_score": "resilience_inward"})
    overall = resilience_df_overall[["cbg_id", "resilience_score"]].rename(columns={"resilience_score": "resilience_overall"})

    dfs = [own, out, inward, overall]
    resilience_table = reduce(lambda left, right: pd.merge(left, right, on="cbg_id", how="outer"), dfs)

    resilience_table["resilience_avg"] = resilience_table[["resilience_own", "resilience_outward", "resilience_inward", "resilience_overall"]].mean(axis=1)
    resilience_table = resilience_table.sort_values("cbg_id").reset_index(drop=True)

    # print(resilience_table.sort_values(by = "resilience_avg", ascending= True))

    resilience_df_own.to_excel(f"{output_dir}/resilience_own.xlsx", index=False)
    resilience_df_out.to_excel(f"{output_dir}/resilience_out.xlsx", index=False)
    resilience_df_in.to_excel(f"{output_dir}/resilience_in.xlsx", index=False)
    resilience_df_overall.to_excel(f"{output_dir}/resilience_overall.xlsx", index=False)
    resilience_table.to_excel(f"{output_dir}/resilience_table.xlsx", index=False)


    #-----------------------------------------------------TPC---------------------------------------------------------
    output_dir = "output_files/TPC"
    os.makedirs(output_dir, exist_ok=True)

    results_own_tpc, results_out_tpc, results_in_tpc, results_device_tpc = [], [], [], []

    for cbg in port_arthur_cbgs_list:
        mobility_tpc = preprocess_cbg(cbg, pa_data)
        baselines_tpc = {
            stream_tpc: compute_baseline(mobility_tpc[stream_tpc], start="2019-09-26", end="2019-11-26")
            for stream_tpc in ['Own', 'Outward', 'Inward', "Overall_count"]
        }

        mobility_tpc = mobility_tpc[(mobility_tpc.index >= '2019-11-01') & (mobility_tpc.index <= '2019-12-31')]
        t0_tpc = get_t0("2019-11-27")

        for stream_tpc, result_list_tpc in zip(['Own', 'Outward', 'Inward', "Overall_count"],
                                            [results_own_tpc, results_out_tpc, results_in_tpc, results_device_tpc]):
            t0 = get_t0("2019-11-27")
            disaster_start = t0
            disaster_end = pd.to_datetime("2019-11-29")
            result_tpc = process_stream(mobility_tpc[stream_tpc], cbg, stream_tpc, t0_tpc, baselines_tpc[stream_tpc])
            result_list_tpc.append(result_tpc)
            plot_single_stream_with_baseline(cbg=cbg,mobility=mobility_tpc,baseline=baselines_tpc[stream_tpc],stream=stream_tpc,output_dir=output_dir, disaster_start=disaster_start, disaster_end=disaster_end)
            plot_combined_streams_per_cbg(cbg=cbg,mobility=mobility_tpc,output_dir=output_dir)


    resilience_df_own_tpc = pd.DataFrame(results_own_tpc)
    resilience_df_out_tpc = pd.DataFrame(results_out_tpc)
    resilience_df_in_tpc = pd.DataFrame(results_in_tpc)
    resilience_df_overall_tpc = pd.DataFrame(results_device_tpc)


    resilience_df_own_tpc.sort_values(by= "resilience_score", ascending=True)

    own_tpc = resilience_df_own_tpc[["cbg_id", "resilience_score"]].rename(columns={"resilience_score": "resilience_own_tpc"})
    out_tpc = resilience_df_out_tpc[["cbg_id", "resilience_score"]].rename(columns={"resilience_score": "resilience_outward_tpc"})
    inward_tpc = resilience_df_in_tpc[["cbg_id", "resilience_score"]].rename(columns={"resilience_score": "resilience_inward_tpc"})
    overall_tpc = resilience_df_overall_tpc[["cbg_id", "resilience_score"]].rename(columns={"resilience_score": "resilience_overall_tpc"})

    dfs_tpc = [own_tpc, out_tpc, inward_tpc, overall_tpc]
    resilience_table_tpc = reduce(lambda left, right: pd.merge(left, right, on="cbg_id", how="outer"), dfs_tpc)

    resilience_table_tpc["resilience_avg_tpc"] = resilience_table_tpc[
        ["resilience_own_tpc", "resilience_outward_tpc", "resilience_inward_tpc", "resilience_overall_tpc"]
    ].mean(axis=1)

    resilience_table_tpc = resilience_table_tpc.sort_values("cbg_id").reset_index(drop=True)
    # print(resilience_table_tpc.sort_values(by = "resilience_avg_tpc", ascending= True))
    resilience_df_own_tpc.to_excel(f"{output_dir}/resilience_own_tpc.xlsx", index=False)
    resilience_df_out_tpc.to_excel(f"{output_dir}/resilience_out_tpc.xlsx", index=False)
    resilience_df_in_tpc.to_excel(f"{output_dir}/resilience_in_tpc.xlsx", index=False)
    resilience_df_overall_tpc.to_excel(f"{output_dir}/resilience_overall_tpc.xlsx", index=False)
    resilience_table_tpc.to_excel(f"{output_dir}/resilience_table_tpc.xlsx", index=False)


