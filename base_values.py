import functools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import geonamescache 
import requests
from dotenv import load_dotenv

load_dotenv()

class BaseValues:

    QUICKSTATS_API_KEY = os.getenv("QUICKSTATS_API_KEY")
    QUICKSTATS_BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"

    _city_list = [
    "Birmingham",
    "Mobile",
    "Montgomery",
    "Phoenix",
    "Tucson",
    "Flagstaff",
    "Little Rock",
    "Los Angeles",
    "Sacramento",
    "San Diego",
    "San Francisco",
    "Fresno",
    "San Jose",
    "Denver",
    "Colorado Springs",
    "Hartford",
    "New York City",
    "Bridgeport",
    "Philadelphia",
    "Dover",
    "Washington",
    "Jacksonville",
    "Miami",
    "Orlando",
    "Tampa",
    "Tallahassee",
    "Atlanta",
    "Savannah",
    "Augusta",
    "Boise",
    "Chicago",
    "St. Louis",
    "Springfield",
    "Indianapolis",
    "Fort Wayne",
    #--"Evansville",
    "Des Moines",
    "Kansas City",
    "Wichita",
    "Topeka",
    "Cincinnati",
    "Louisville",
    "Lexington",
    "Baton Rouge",
    "Lake Charles",
    "New Orleans",
    "Shreveport",
    "Portland",
    "Baltimore",
    "Annapolis",
    "Boston",
    "Worcester",
    "Detroit",
    "Grand Rapids",
    "Lansing",
    "Minneapolis",
    "Saint Paul",
    "Jackson",
    "Billings",
    "Omaha",
    "Lincoln",
    "Las Vegas",
    "Reno",
    "Manchester",
    "Concord", #manually have to set coordinates
    "Albany",
    "Buffalo",
    "Rochester",
    "Syracuse",
    "Charlotte",
    "Greensboro",
    "Raleigh",
    "Wilmington",
    "Bismarck",
    "Cleveland",
    "Columbus",
    "Dayton",
    "Toledo",
    "Oklahoma City",
    "Tulsa",
    #--"Norman",
    "Eugene",
    "Pittsburgh",
    "Harrisburg",
    "Providence",
    "Charleston",
    "Greenville",
    "Columbia",
    "Sioux Falls",
    "Memphis",
    "Nashville",
    "Knoxville",
    "Chattanooga",
    "Austin",
    "Beaumont",
    "Corpus Christi",
    "Dallas",
    "El Paso",
    "Houston",
    "Laredo",
    "San Antonio",
    "Lubbock",
    "Salt Lake City",
    "Provo",
    "Richmond",
    "Virginia Beach",
    "Roanoke",
    "Seattle",
    "Spokane",
    "Milwaukee",
    #--"Madison",
    "Cheyenne",
    "Albuquerque"
    ]
    commodities = ["CORN", "SOYBEANS", "WHEAT", "COTTON", "HAY"]
    _edges = [
    ("Seattle", "Spokane"),
    ("Seattle", "Portland"),
    ("Portland", "Eugene"),
    ("Portland", "Sacramento"),
    ("Sacramento", "Reno"),
    ("Sacramento", "San Francisco"),
    ("San Francisco", "San Jose"),
    ("San Francisco", "Fresno"),
    ("Fresno", "Los Angeles"),
    ("Los Angeles", "San Diego"),
    ("Los Angeles", "Las Vegas"),
    ("Reno", "Salt Lake City"),
    ("Salt Lake City", "Provo"),
    ("Salt Lake City", "Boise"),
    ("Salt Lake City", "Denver"),
    ("Boise", "Billings"),
    ("Billings", "Cheyenne"),
    ("Cheyenne", "Denver"),
    ("Denver", "Colorado Springs"),
    ("Las Vegas", "Phoenix"),
    ("Phoenix", "Flagstaff"),
    ("Phoenix", "Tucson"),
    ("Flagstaff", "Albuquerque"),
    ("Albuquerque", "El Paso"),
    ("El Paso", "Lubbock"),
    ("El Paso", "San Antonio"),
    ("El Paso", "Denver"),
    ("Lubbock", "Dallas"),
    ("Dallas", "Oklahoma City"),
    ("Dallas", "Houston"),
    ("Dallas", "Austin"),
    ("Austin", "San Antonio"),
    ("San Antonio", "Corpus Christi"),
    ("Houston", "Beaumont"),
    ("Houston", "Laredo"),
    ("Houston", "Lake Charles"),
    ("Beaumont", "Lake Charles"),
    ("Lake Charles", "Baton Rouge"),
    ("Baton Rouge", "New Orleans"),
    ("New Orleans", "Mobile"),
    ("Mobile", "Birmingham"),
    ("Birmingham", "Montgomery"),
    ("Birmingham", "Atlanta"),
    ("Shreveport", "Dallas"),
    ("Shreveport", "Baton Rouge"),
    ("Oklahoma City", "Tulsa"),
    ("Tulsa", "Kansas City"),
    ("Kansas City", "Topeka"),
    ("Kansas City", "Wichita"),
    ("Kansas City", "Omaha"),
    ("Omaha", "Lincoln"),
    ("Omaha", "Sioux Falls"),
    ("Sioux Falls", "Bismarck"),
    ("Bismarck", "Billings"),
    ("Omaha", "Des Moines"),
    ("Des Moines", "Minneapolis"),
    ("Minneapolis", "Saint Paul"),
    ("Minneapolis", "Milwaukee"),
    ("Milwaukee", "Chicago"),
    ("Chicago", "Detroit"),
    ("Chicago", "Indianapolis"),
    ("Chicago", "St. Louis"),
    ("St. Louis", "Springfield"),
    ("Indianapolis", "Fort Wayne"),
    ("Indianapolis", "Louisville"),
    ("Louisville", "Lexington"),
    ("Louisville", "Cincinnati"),
    ("Cincinnati", "Dayton"),
    ("Dayton", "Columbus"),
    ("Columbus", "Cleveland"),
    ("Cleveland", "Toledo"),
    ("Toledo", "Detroit"),
    ("Detroit", "Grand Rapids"),
    ("Grand Rapids", "Lansing"),
    ("Memphis", "St. Louis"),
    ("Memphis", "Nashville"),
    ("Memphis", "Jackson"),
    ("Memphis", "Little Rock"),
    ("Nashville", "Knoxville"),
    ("Knoxville", "Chattanooga"),
    ("Nashville", "Atlanta"),
    ("Atlanta", "Augusta"),
    ("Atlanta", "Savannah"),
    ("Savannah", "Jacksonville"),
    ("Jacksonville", "Tallahassee"),
    ("Jacksonville", "Orlando"),
    ("Orlando", "Tampa"),
    ("Tampa", "Miami"),
    ("Charlotte", "Greensboro"),
    ("Greensboro", "Raleigh"),
    ("Raleigh", "Wilmington"),
    ("Charlotte", "Columbia"),
    ("Columbia", "Charleston"),
    ("Columbia", "Greenville"),
    ("Charlotte", "Atlanta"),
    ("Charlotte", "Richmond"),
    ("Richmond", "Roanoke"),
    ("Richmond", "Washington"),
    ("Richmond", "Virginia Beach"),
    ("Washington", "Baltimore"),
    ("Baltimore", "Annapolis"),
    ("Baltimore", "Philadelphia"),
    ("Philadelphia", "Harrisburg"),
    ("Philadelphia", "Pittsburgh"),
    ("Philadelphia", "New York City"),
    ("Pittsburgh", "Cleveland"),
    ("New York City", "Hartford"),
    ("Hartford", "Bridgeport"),
    ("Hartford", "Providence"),
    ("Providence", "Boston"),
    ("Boston", "Worcester"),
    ("New York City", "Albany"),
    ("Albany", "Buffalo"),
    ("Buffalo", "Rochester"),
    ("Rochester", "Syracuse"),
    ("Albany", "Boston"),
    ("Dover", "Philadelphia"),
    ("Manchester", "Concord"),
    ("Manchester", "Boston"),
    ("Concord", "Boston")
    ]
   
    CITY_TO_FAF5 = {
    # --- West Coast ---
    "Seattle": 531, "Spokane": 539, "Portland": 411, "Eugene": 419,
    "Sacramento": 62, "San Francisco": 64, "San Jose": 64, "Fresno": 65,
    "Los Angeles": 61, "San Diego": 63,

    # --- Mountain West ---
    "Reno": 329, "Salt Lake City": 491, "Provo": 499, "Boise": 160,
    "Billings": 300, "Cheyenne": 560, "Denver": 81, "Colorado Springs": 89,

    # --- Southwest ---
    "Las Vegas": 321, "Phoenix": 41, "Tucson": 42, "Flagstaff": 49,
    "Albuquerque": 350, "El Paso": 485,

    # --- Texas ---
    "Lubbock": 489, "Dallas": 484, "Austin": 481, "San Antonio": 488,
    "Corpus Christi": 483, "Houston": 486, "Beaumont": 482, "Laredo": 487,

    # --- Gulf Coast ---
    "Lake Charles": 222, "Baton Rouge": 221, "New Orleans": 223,
    "Mobile": 12, "Birmingham": 11, "Montgomery": 19, "Shreveport": 229,

    # --- Central Plains ---
    "Oklahoma City": 401, "Tulsa": 402, "Kansas City": 291, "Topeka": 209,
    "Wichita": 202, "Omaha": 311, "Lincoln": 319, "Sioux Falls": 460,
    "Bismarck": 380, "Des Moines": 190,

    # --- Midwest core ---
    "Minneapolis": 271, "Saint Paul": 271, "Milwaukee": 551, "Chicago": 171,
    "Detroit": 261, "Indianapolis": 182, "St. Louis": 292, "Springfield": 179,
    "Fort Wayne": 183, "Louisville": 212, "Lexington": 219, "Cincinnati": 211,

    # --- Ohio Valley / Great Lakes ---
    "Dayton": 394, "Columbus": 393, "Cleveland": 392, "Toledo": 399,
    "Grand Rapids": 262, "Lansing": 269,

    # --- Southeast ---
    "Memphis": 471, "Nashville": 472, "Jackson": 280, "Knoxville": 473,
    "Chattanooga": 479, "Atlanta": 131, "Augusta": 139, "Savannah": 132,
    "Jacksonville": 121, "Tallahassee": 129, "Orlando": 123, "Tampa": 124,
    "Miami": 122,"Little Rock": 50,

    # --- Carolinas / Mid-Atlantic ---
    "Charlotte": 371, "Greensboro": 372, "Raleigh": 373, "Wilmington": 379,
    "Columbia": 459, "Charleston": 451, "Greenville": 452, "Richmond": 511,
    "Roanoke": 519, "Washington": 111, "Virginia Beach": 512,

    # --- East Coast Corridor ---
    "Baltimore": 241, "Annapolis": 249, "Philadelphia": 421, "Harrisburg": 429,
    "Pittsburgh": 422, "New York City": 363, "Hartford": 91, "Bridgeport": 99,
    "Providence": 441, "Boston": 251, "Worcester": 259, "Albany": 361,
    "Buffalo": 362, "Rochester": 364, "Syracuse": 369,

    # --- Delaware / New England ---
    "Dover": 109, "Manchester": 339, "Concord": 339,
    }

    state_comm_pair = [("CORN","IA"),("CORN","IL"),("CORN","NE"),
                   ("COTTON","TX"),("COTTON","GA"),("COTTON","AR"),
                   ("HAY", "TX"),("HAY","OK"),("HAY","NE"),
                   ("SOYBEANS","MN"),("SOYBEANS","IL"),("SOYBEANS","IA"),
                   ("WHEAT","KS"),("WHEAT","SD"),("WHEAT","MT")
                  ]

    state_city_commodity_mapping =  {
        ("CORN", "IA"): "Des Moines",
        ("CORN", "IL"): "Chicago",
        ("CORN", "NE"): "Omaha",

        ("COTTON", "TX"): "Lubbock",
        ("COTTON", "GA"): "Savannah",
        ("COTTON", "AR"): "Little Rock",

        ("HAY", "TX"): "San Antonio",
        ("HAY", "OK"): "Oklahoma City",
        ("HAY", "NE"): "Omaha",

        ("SOYBEANS", "MN"): "Saint Paul",
        ("SOYBEANS", "IL"): "Chicago",
        ("SOYBEANS", "IA"): "Des Moines",

        ("WHEAT", "KS"): "Wichita",
        ("WHEAT", "SD"): "Sioux Falls",
        ("WHEAT", "MT"): "Billings",
        }


    #Required to create initial price list 
    BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
    commodities = ["CORN", "SOYBEANS", "WHEAT", "COTTON", "HAY"]

    #Data is in propriety units 
    #Create a conversion table 
    UNIT_MAP = {
        "CORN":     "BU",
        "SOYBEANS": "BU",
        "WHEAT":    "BU",
        "COTTON":   "480 LB BALES",
        "HAY":      "TONS",
        }

    conversion_table = {"CORN": 37.714,
                   "WHEAT":33.333,
                   "SOYBEANS":33.333,
                   "COTTON": 4.167,
                   "HAY": 1}
    CITY_STATE_GROUPINGS = {
    "CORN": {
        "Des Moines": "IA",
        "Chicago":    "IL",
        "Omaha":      "NE",
    },
    "COTTON": {
        "Lubbock":     "TX",
        "Savannah":    "GA",
        "Little Rock": "AR",
    },
    "HAY": {
        "San Antonio":   "TX",
        "Oklahoma City": "OK",
        "Omaha":         "NE",
    },
    "SOYBEANS": {
        "Saint Paul": "MN",
        "Chicago":    "IL",
        "Des Moines": "IA",
    },
    "WHEAT": {
        "Wichita":     "KS",
        "Sioux Falls": "SD",
        "Billings":    "MT",
    },
        }
    
    def __init__(self):
        script_dir = Path(__file__).resolve().parent
        resources_dir = script_dir.parent / "resources"

        self.faf5_path = resources_dir / "FAF5.7.1.csv"
        self.total_truck_flow_path = resources_dir / "FAF5 Total Truck Flows by Commodity_2022.csv"
        self.link_id_path = resources_dir / "NTAD_Freight_Analysis_Framework_Network_Links_5033135967609690907.csv"
        self.demand_path = resources_dir / "synthetic_monthly_demand_long.csv"

        self.faf_df = pd.read_csv(self.faf5_path)
        self.truck_df = self.faf_df[self.faf_df["dms_mode"] == 1]
        self.highway_df = pd.read_csv(self.total_truck_flow_path)
        self.link_ids = pd.read_csv(self.link_id_path)
        self.synthetic_demand = pd.read_csv(self.demand_path)


    @property
    def city_list(self):
        return self._city_list
    #using Pythonic caching , computed just once 
    
    @functools.cached_property
    def edges(self):
        #bi-directional edge list
        return self._edges + [(b,a) for a,b in self._edges]
    
    @functools.cached_property
    def city_dist(self):
        gc = geonamescache.GeonamesCache()
        dist_dict = {}
        def get_coords(city_name):
            city_matches = gc.get_cities_by_name(city_name)
            if not city_matches:
                return None
            us_cities=[]
            for match in city_matches:
                for geoname_id, info in match.items():
                    if info['countrycode'] == "US":
                        us_cities.append(info)
            if not us_cities:
                value_add.append(city_name)
                return None
            #Select largest city by population
            major_city = max(us_cities, key = lambda x: x['population'])
            
            if major_city =='Concord':
                return -71.53757, 43.20814
            return major_city['longitude'],major_city['latitude']

        def haversine_distance(lat1,lon1,lat2,lon2):
            R = 3958.8 # Return distance in miles use r = 6371.0 for kilometers
            lat1,lon1,lat2,lon2 = map(np.radians, [lat1,lon1,lat2,lon2])
            a = np.sin((lat2 - lat1) / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1) /2)**2
            return R * 2 * np.arctan2(np.sqrt(a),np.sqrt(1-a))
        
        for city_a,city_b in self.edges:
            log_a,lat_a = get_coords(city_a)
            log_b,lat_b = get_coords(city_b)
            dist = haversine_distance(lat_a,log_a,lat_b,log_b)
            dist_dict[(city_a,city_b)] = dist
        return dist_dict

    @functools.cached_property 
    def cap_dict(self):
        city_capacity = {}
        for x,y in self.edges:
            temp_df = self.truck_df[(self.truck_df['dms_orig'] == self.CITY_TO_FAF5[x]) & (self.truck_df['dms_dest'] == self.CITY_TO_FAF5[y]) & (self.truck_df['sctg2'].isin([2,3,4])) 
                &(self.truck_df['trade_type'] == 1)].groupby(["dms_orig","dms_dest"]).mean(numeric_only =True).reset_index()
        if not temp_df.empty:
            city_capacity[(x,y)] = float(temp_df["tons_2024"].iloc[0])
        else:
            city_capacity[(x,y)] = 0
        
        #Manually set edges with no FAF record
        def avg_capacity_for(road_name,states):
            subset = self.link_ids[
                (self.link_ids["Road_Name"] == road_name)
                & (self.link_ids["STATE"].isin(states) if isinstance(states, (list, tuple, set))
                   else self.link_ids["STATE"] == states)
            ]
            ids = subset["ID"].tolist()
            hwy_subset = self.highway_df[self.highway_df["ID"].isin(ids)]
            return hwy_subset["AB Farm Products-Tons_22 All"].mean()

        denver_cap_avg = avg_capacity_for("I 25", ["NM","CO"])
        laredo_cap_avg = avg_capacity_for("US 59", "TX")
        richmond_cap_avg = avg_capacity_for("I 95", ["DC", "VA"])
        dover_cap_avg = avg_capacity_for("DE 1", "DE")
        el_paso_cap_avg = avg_capacity_for("I 10", "TX")

        city_capacity[("El Paso", "San Antonio")] = el_paso_cap_avg * 0.75
        city_capacity[("El Paso", "Denver")] = denver_cap_avg * 0.75
        city_capacity[("Houston", "Laredo")] = laredo_cap_avg * 0.75
        city_capacity[("Richmond", "Washington")] = richmond_cap_avg * 0.75
        city_capacity[("Dover", "Philadelphia")] = dover_cap_avg * 0.75
        #Make it double sided for modeling purposes going forward 
        city_capacity[("San Antonio", "El Paso")] = el_paso_cap_avg * 0.75
        city_capacity[("Denver", "El Paso")] = denver_cap_avg * 0.75
        city_capacity[("Laredo", "Houston")] = laredo_cap_avg * 0.75
        city_capacity[("Washington", "Richmond")] = richmond_cap_avg * 0.75
        city_capacity[("Philadelphia", "Dover")] = dover_cap_avg * 0.75

        return city_capacity
    @functools.cached_property
    def demand_dict(self):
        #creating dictionary for initial demand data 
        unique_city = self.synthetic_demand["city"].unique()
        unique_city_commodity = self.synthetic_demand.drop_duplicates(subset=['commodity','city'])
        commodity_city_pairs = unique_city_commodity[["commodity","city"]]
        #multiply by 12 for yearly demand since data is monthly
        demand_dict = {}
        for index, row in commodity_city_pairs.iterrows():
            demand_dict[(row["city"],row["commodity"].upper())] = float(self.synthetic_demand.loc[(self.synthetic_demand["commodity"] == row["commodity"]) & (self.synthetic_demand["city"] == row["city"]), "demand_estimate"].iloc[0]) * 12
        for (city_a,comm),unit in demand_dict.items():
            demand_dict[(city_a,comm)] = unit * self.conversion_table[comm]
        return demand_dict

    @functools.cached_property
    def supply_dict(self):
        def fetch_state_production(commodity: str, city: str, year: int = 2025) -> pd.DataFrame:
    
            state_alpha = self.CITY_STATE_GROUPINGS[commodity][city]
            params = {
            "key":                   self.QUICKSTATS_API_KEY,
            "commodity_desc":        commodity,
            "statisticcat_desc":     "PRODUCTION",
            "unit_desc":             self.UNIT_MAP[commodity],
            "freq_desc":             "ANNUAL",
            "reference_period_desc": "YEAR",
            "agg_level_desc":        "STATE",
            "state_alpha":           state_alpha,
            "year":                  year,
            "format":                "JSON",
            }
            resp = requests.get(self.QUICKSTATS_BASE_URL, params=params)
            result = resp.json()
            if "data" not in result:
                print(f"No data for {commodity} / {city} ({state_alpha}, {year}):", result)
                return pd.DataFrame()

            df = pd.DataFrame(result["data"])
            df = df[["commodity_desc", "state_alpha", "year", "Value", "unit_desc"]]
            df["Value"] = df["Value"].str.replace(",", "").str.strip()
            df = df[~df["Value"].isin(["(D)", "(Z)", "(NA)", ""])]
            df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
            df = df.rename(columns={
                "commodity_desc": "commodity",
                "state_alpha":    "state",
                "Value":          "production",
            })
            df["city"] = city
            return df


        def fetch_all_state_production(year: int = 2025) -> pd.DataFrame:
            frames = []
            for commodity, cities in self.CITY_STATE_GROUPINGS.items():
                for city in cities:
                    df = fetch_state_production(commodity, city, year=year)
                    if not df.empty:
                        frames.append(df)
            if not frames:
                    return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)


        production_panel = fetch_all_state_production(year=2025)
        supply_dict = {}
        for index, row in production_panel.iterrows():
            supply_dict[(row["city"],row["commodity"])] = float(production_panel.loc[(production_panel["commodity"] == row["commodity"]) & (production_panel["city"] == row["city"]), "production"].iloc[0])
        #apply conversion
        for (city_a,comm),unit in supply_dict.items():
            supply_dict[(city_a,comm)] = unit * self.conversion_table[comm]
        return supply_dict
    
    @functools.cached_property
    def price_dict(self):
        def fetch_monthly_price(commodity: str,year_start: int = 2020,year_end: int = 2024) -> pd.DataFrame:
            params = {
                "key":                self.QUICKSTATS_API_KEY,
                "commodity_desc":     commodity,
                "statisticcat_desc":  "PRICE RECEIVED",
                "freq_desc":          "MONTHLY",
                "agg_level_desc":     "STATE",
                "year__GE":           year_start,
                "year__LE":           year_end,
                "format":             "JSON",
                }

            resp   = requests.get(self.QUICKSTATS_BASE_URL, params=params)
            result = resp.json()

            if "data" not in result:
                print(f"No monthly price for {commodity}:", result)
                return pd.DataFrame()

            df = pd.DataFrame(result["data"])
            df = df[["commodity_desc", "state_alpha", "year",
             "reference_period_desc", "Value", "unit_desc"]]

            df["Value"] = (df["Value"]
                   .str.replace(",", "")
                   .str.strip())
            df = df[~df["Value"].isin(["(D)", "(Z)", "(NA)", ""])]
            df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

            df = df.rename(columns={
             "commodity_desc":        "commodity",
            "state_alpha":           "state",
             "reference_period_desc": "month",
                "Value":                 "price_usd",
             })

            return df
        
        price_frames = []
        for comm in self.commodities:
            df = fetch_monthly_price(comm, year_start=2020, year_end=2024)
            price_frames.append(df)

        price_panel = pd.concat(price_frames, ignore_index=True)

        comm_price_state = {}

        for comm,state in self.state_comm_pair:
            comm_price_state[comm,state] = (price_panel.loc[(price_panel["commodity"] == comm) & (price_panel["state"] == state), "unit_desc"].iloc[0],
                float(price_panel.loc[(price_panel["commodity"] == comm) & (price_panel["state"] == state), "price_usd"].iloc[0]))
        final_comm_price = {}
        for key, value in comm_price_state.items():
            unit,price = value 
            comm,state = key
            city = self.state_city_commodity_mapping.get(key)
            final_comm_price[(city,comm)]= price
        final_comm_price[('Little Rock', 'COTTON')] = 217.0

        for (city_a,comm),price in final_comm_price.items():
            final_comm_price[(city_a,comm)] = price * self.conversion_table[comm]
        return final_comm_price


    


