import numpy as np 
import pandas as pd 


def build_location_table():
    apple_data_path = "data/applemobilitytrends-2020-05-03.csv"
    google_data_path = "data/Google_Global_Mobility_Report.csv"
    us_cities_path = "data/uscities.csv"
    world_cities_path = "data/worldcities.csv"
    google_df = pd.read_csv(google_data_path)
    apple_df = pd.read_csv(apple_data_path)
    us_cities_df = pd.read_csv(us_cities_path)
    world_cities_df = pd.read_csv(world_cities_path)
    us_google_df = google_df.loc[google_df['country_region'] == "United States"]
    states = us_google_df.sub_region_1.unique()
    google_counties = us_google_df.sub_region_2.unique()
    regions_apple_df = apple_df.loc[apple_df['geo_type'] == "country/region"]
    us_apple_df = apple_df.loc[apple_df['region'] == "United States"]
    sub_regions_apple_df = apple_df.loc[apple_df['geo_type'] == "sub-region"]
    cities_apple_df = apple_df.loc[apple_df['geo_type'] == "city"]
    states_apple_df = sub_regions_apple_df.loc[sub_regions_apple_df["region"].isin(states)]

    apple_counties = {}
    for city in cities_apple_df.region.unique():
        # cleaning up inconsistency in the Apple naming convetion
        if city == 'Birmingham - Alabama':
            city = 'Birmingham'
        elif city == 'New York City':
            city = 'New York'
        elif city == 'Saint Petersburg - Clearwater (Florida)':
            city = 'Saint Petersburg'
        elif city == 'San Francisco - Bay Area':
            city = 'San Francisco'
        elif city == 'Washington DC':
            city = 'District of Columbia'
        city_df = us_cities_df.loc[us_cities_df["city"] == city]
        world_city_df = world_cities_df.loc[world_cities_df["city_ascii"] == city] # need to use city_ascii column to address places like Zürich
        country = ""
        state = ""
        county = ""
        # filter out cities that have a larger city outside the US with the same name assuming apple was refering to that one
        if not city_df.empty and not world_city_df.empty:
            # check if largest city of this name is in the US
            world_city_df = world_city_df.sort_values(by=['population'], ascending=False)
            if world_city_df.iloc[0]["country"] == "United States":
                country = world_city_df.iloc[0]["country"]
                # find the state containing this city with the largest population
                city_df = city_df.sort_values(by=['population'], ascending=False)
                state = city_df.iloc[0]["state_name"]
                county = city_df.iloc[0]["county_name"]
                print("{} is most populated in {}, {}".format(city, county, state))
                apple_counties[county] = (state, city)
        elif not city_df.empty and world_city_df.empty:
            # find the most populated of the cities in the US with the same name
            city_df.sort_values(by=['population'])
            state = city_df.iloc[0]["state_name"]
            county = city_df.iloc[0]["county_name"]
            print("{} is most populated in {}, {}".format(city, county, state))
            apple_counties[county] = (state, city)

    location_table = pd.DataFrame(columns=["country", "state", "county", "city", "google", "apple"])
    # construct location table starting with all counties present in google dataset
    for county in google_counties:
        county_stripped = ""
        if type(county) == type(""): # there appears to be at least 1 float in the county column?
            county_stripped = county.split(" ")[0] # remove the word county from the google counties
        state = us_google_df.loc[us_google_df["sub_region_2"] == county]
        states = state["sub_region_1"].values
        if len(states) != 0:
            state = states[0]
        else:
            state = "N/A"
        # check if county is also present in apple dataset
        apple_indicator = 0
        city = "N/A"
        if county in apple_counties.keys():
            apple_indicator = 1
            city = apple_counties[county][1]
        elif county_stripped in apple_counties.keys():
            apple_indicator = 1
            city = apple_counties[county_stripped][1]
        location_table = location_table.append({
            "country": "US",
            "state":  state,
            "county": county,
            "city": city,
            "google": 1,
            "apple": apple_indicator
            }, ignore_index=True)
    # append remaining counties from apple dataset that are not present in google dataset if any
    for county, state_city in apple_counties.items():
        county_appended = "{} County".format(county)
        state, city = state_city
        if location_table[(location_table['county'] == county)].empty and location_table[(location_table['county'] == county_appended)].empty:
            # add a new row with apple flag high and google flag low if it wasnt found in google dataset
            location_table = location_table.append({
                "country": "US",
                "state":  state,
                "county": county_appended,
                "city": city,
                "google": 0,
                "apple": 1
            }, ignore_index=True)

    location_table.to_csv("data/location_table.csv")
        


build_location_table()
