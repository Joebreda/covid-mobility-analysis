import numpy as np 
import pandas as pd 
import csv
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import datetime

def plot_king_county():
    #load and plot covid cases data
    covid_data_dict = {}
    skip_date_for_plot = 0  #skip some early dates that have 0 cases
    county_key_word = "King, Washington, US"


    with open('data/time_series_covid19_confirmed_US.csv','r') as csvfile:
        data = csv.reader(csvfile, delimiter = ',')
        header = next(data)
        for row in data:
            covid_data_dict[row[10]] = np.array(row[11+skip_date_for_plot:], int)

    date_string = [date[:-3] for date in header[11+skip_date_for_plot:]]

    county_covid_cases = covid_data_dict[county_key_word]

    xticks_date_index = range(0,len(county_covid_cases),5)
    xticks_date = [date_string[i] for i in xticks_date_index]

    fig = plt.figure(figsize=(24,8))
    plt.plot(county_covid_cases)
    plt.xticks(xticks_date_index, xticks_date, rotation=20) 
    plt.xlabel("date")
    plt.ylabel("confirmed cases")
    plt.title("number of covid cases in " + county_key_word)
    plt.show()

def total_cases():
    covid_cases_path = 'data/time_series_covid19_confirmed_US.csv'
    covid_cases_df = pd.read_csv(covid_cases_path)

    locations = covid_cases_df.Combined_Key.values
    countries = []
    states = []
    counties = []
    for location in locations:
        location = location.split(",")
        if len(location) == 3:
            counties.append(location[0])
            states.append(location[1].replace(" ", "")) # remove spaces from county names
            countries.append(location[2].replace(" ", ""))
        else:
            counties.append(location[0])
            states.append("N/A")
            countries.append(location[1])
    country_col = pd.DataFrame({'country': countries})
    state_col = pd.DataFrame({'state': states})
    county_col = pd.DataFrame({'county': counties})
    total_cases = country_col.join(state_col)
    total_cases = total_cases.join(county_col)

    columns = covid_cases_df.columns
    dates = columns[11:].values


    # subtract previous day count from current day count to get daily new cases
    for i, cur_date in enumerate(dates):
        cur_date_data = covid_cases_df[cur_date].values
        dated_total_cases_col = pd.DataFrame({cur_date: cur_date_data})
        total_cases = total_cases.join(dated_total_cases_col)

    return total_cases


def new_cases_timeseries_table():
    covid_cases_path = 'data/time_series_covid19_confirmed_US.csv'
    covid_cases_df = pd.read_csv(covid_cases_path)

    locations = covid_cases_df.Combined_Key.values
    countries = []
    states = []
    counties = []
    for location in locations:
        location = location.split(",")
        if len(location) == 3:
            counties.append(location[0])
            states.append(location[1].replace(" ", "")) # remove spaces from county names
            countries.append(location[2].replace(" ", ""))
        else:
            counties.append(location[0])
            states.append("N/A")
            countries.append(location[1])
    country_col = pd.DataFrame({'country': countries})
    state_col = pd.DataFrame({'state': states})
    county_col = pd.DataFrame({'county': counties})
    new_cases_df = country_col.join(state_col)
    new_cases_df = new_cases_df.join(county_col)

    columns = covid_cases_df.columns
    dates = columns[11:].values


    # subtract previous day count from current day count to get daily new cases
    for i, cur_date in enumerate(dates):
        if i == 0:
            prev_date = cur_date
            continue # skip the first date
        cur_date_data = covid_cases_df[cur_date].values
        pre_date_data = covid_cases_df[prev_date].values
        #print(cur_date, prev_date, cur_date_data)

        num_new_cases = cur_date_data - pre_date_data
        num_new_cases = num_new_cases.clip(min=0) # assume any negative new cases are error in data entry and set as 0
        dated_new_cases_col = pd.DataFrame({cur_date: num_new_cases})
        new_cases_df = new_cases_df.join(dated_new_cases_col)
        prev_date = cur_date

    #print(new_cases_df)
    return new_cases_df


def new_deaths_timeseries_table():
    covid_deaths_path = 'data/time_series_covid19_deaths_US.csv'
    covid_deaths_df = pd.read_csv(covid_deaths_path)

    locations = covid_deaths_df.Combined_Key.values
    countries = []
    states = []
    counties = []
    for location in locations:
        location = location.split(",")
        if len(location) == 3:
            counties.append(location[0])
            states.append(location[1].replace(" ", "")) # remove spaces from county names
            countries.append(location[2].replace(" ", ""))
        else:
            counties.append(location[0])
            states.append("N/A")
            countries.append(location[1])
    country_col = pd.DataFrame({'country': countries})
    state_col = pd.DataFrame({'state': states})
    county_col = pd.DataFrame({'county': counties})
    new_deaths_df = country_col.join(state_col)
    new_deaths_df = new_deaths_df.join(county_col)

    columns = covid_deaths_df.columns
    dates = columns[11:].values


    # subtract previous day count from current day count to get daily new cases
    for i, cur_date in enumerate(dates):
        if i == 0:
            prev_date = cur_date
            continue # skip the first date
        cur_date_data = covid_deaths_df[cur_date].values
        pre_date_data = covid_deaths_df[prev_date].values
        #print(cur_date, prev_date, cur_date_data)

        num_new_deaths = cur_date_data - pre_date_data
        num_new_deaths = num_new_deaths.clip(min=0) # assume any negative new cases are error in data entry and set as 0
        dated_new_deaths_col = pd.DataFrame({cur_date: num_new_deaths})
        new_deaths_df = new_deaths_df.join(dated_new_deaths_col)
        prev_date = cur_date

    #print(new_deaths_df)
    return new_deaths_df


def google_county_naming_2_covid_county_naming(county_name):
    stripped_county_name = str(county_name).replace("New York County", "New York City")
    stripped_county_name = str(stripped_county_name).replace(" County", "")
    stripped_county_name = str(stripped_county_name).replace(" Parish", "")
    stripped_county_name = str(stripped_county_name).replace(" Borough", "")
    stripped_county_name = str(stripped_county_name).replace("Doña Ana", "Dona Ana")
    return stripped_county_name

def get_common_start_and_end_dates(mobility_table, google_df):
    mobility_counties = mobility_table.county.values
    mobility_counties = mobility_counties[1:] # remove the NAN thats in the first row
    mobility_states = mobility_table.state.values
    mobility_states = mobility_states[1:] # remove the NAN thats in the first row
    
    latest_start_date = None 
    latest_start_date_string = None
    earliest_end_date = None 
    earliest_end_date_string = None
    # determine the common start and end dates for each single county 
    for state, county in zip(mobility_states, mobility_counties):
        this_state = google_df.loc[google_df['sub_region_1'] == str(state)]
        this_county = this_state.loc[this_state['sub_region_2'] == str(county)]
        date_span = this_county.date.values
        #print("for county ", county, " we have ", date_span[0], date_span[len(date_span)-1])
        start_date = date_span[0]
        end_date = date_span[len(date_span)-1]
        start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        if latest_start_date == None:
            latest_start_date = start_datetime
        if earliest_end_date == None:
            earliest_end_date = end_datetime
        
        if end_datetime <= earliest_end_date:
            earliest_end_date_string = end_date
            earliest_end_date = end_datetime
            
        if start_datetime >= latest_start_date:
            latest_start_date_string = start_date
            latest_start_date = start_datetime
        
    print("we have data for all locations from ", latest_start_date_string, " to ", earliest_end_date_string)
    return earliest_end_date_string, earliest_end_date, latest_start_date_string, latest_start_date

def format_mobility_like_covid_data():

    new_cases_df = new_cases_timeseries_table()
    new_deaths_df = new_deaths_timeseries_table()

    mobility_table_path = 'data/location_table.csv'
    mobility_table = pd.read_csv(mobility_table_path)
    mobility_counties = mobility_table.county.values
    mobility_counties = mobility_counties[1:] # remove the NAN thats in the first row
    mobility_states = mobility_table.state.values
    mobility_states = mobility_states[1:] # remove the NAN thats in the first row

    google_data_path = "data/Google_Global_Mobility_Report.csv"
    google_df = pd.read_csv(google_data_path)
    google_df = google_df.loc[google_df['country_region'] == "United States"]

    earliest_end_date_string, earliest_end_date, latest_start_date_string, latest_start_date = get_common_start_and_end_dates(mobility_table, google_df)
    
    # convert 1 master dataset into 6 sub datasets, reformatted to look like covid data
    retail_rows = []
    grocery_rows = []
    park_rows = []
    transit_rows = []
    workplaces_rows = []
    residential_rows = []
    # loop across each state county pair and add their columns to the 6 new dataframes as timeseries rows
    for state, county in zip(mobility_states, mobility_counties):
        this_state = google_df.loc[google_df['sub_region_1'] == str(state)]
        this_county = this_state.loc[this_state['sub_region_2'] == str(county)]
        this_county['date'] =  pd.to_datetime(this_county['date'], format='%Y-%m-%d')
        this_county_in_common_dates = this_county[(this_county['date'] >= latest_start_date) & (this_county['date'] <= earliest_end_date)]

        print(this_county_in_common_dates)
        print(type(this_county_in_common_dates))

        # convert back to strings for column names
        this_county_in_common_dates = this_county_in_common_dates['date'].dt.strftime('%Y/%m/%d')


        # almost all state, county pairs contain 28 days - omit all that have less than 28 for simplicity 
        dates = list(this_county_in_common_dates.date.values)

        if dates[0] == latest_start_date_string and dates[len(dates)-1] == earliest_end_date_string and len(dates) == 28:
            col_dates = dates
            print(dates)
        if len(dates) != 28:
            continue
            #print(dates[0], dates[len(dates)-1])
            #print(len(dates))
            #print(dates)

        # reformat name to match those present in the covid dataset
        county = google_county_naming_2_covid_county_naming(county)

        # separate 6 columns from google dataset to 6 separated 
        retail_row = this_county_in_common_dates.retail_and_recreation_percent_change_from_baseline.values
        retail_rows.append([state, county] + list(retail_row))
        '''
        grocery_row = this_county_in_common_dates.grocery_and_pharmacy_percent_change_from_baseline.values
        grocery_rows.append([state, county] + list(grocery_row))

        park_row = this_county_in_common_dates.parks_percent_change_from_baseline.values
        park_rows.append([state, county] + list(park_row))

        transit_row = this_county_in_common_dates.transit_stations_percent_change_from_baseline.values
        transit_rows.append([state, county] + list(transit_row))

        workplaces_row = this_county_in_common_dates.workplaces_percent_change_from_baseline.values
        workplaces_rows.append([state, county] + list(workplaces_row))

        residential_row = this_county_in_common_dates.residential_percent_change_from_baseline.values
        residential_rows.append([state, county] + list(residential_row))
        '''

    retail_df = pd.DataFrame(retail_rows, columns=[["state", "county"] + col_dates])
    '''
    grocery_df = pd.DataFrame(grocery_rows, columns=[["state", "county"] + col_dates])
    park_df = pd.DataFrame(park_rows, columns=[["state", "county"] + col_dates])
    transit_df = pd.DataFrame(transit_rows, columns=[["state", "county"] + col_dates])
    workplaces_df = pd.DataFrame(workplaces_rows, columns=[["state", "county"] + col_dates])
    residential_df = pd.DataFrame(residential_rows, columns=[["state", "county"] + col_dates])

    retail_df.to_csv("data/split_google_metrics/retail.csv")
    grocery_df.to_csv("data/split_google_metrics/grocery.csv")
    park_df.to_csv("data/split_google_metrics/park.csv")
    transit_df.to_csv("data/split_google_metrics/transit.csv")
    workplaces_df.to_csv("data/split_google_metrics/workplaces.csv")
    residential_df.to_csv("data/split_google_metrics/residential.csv")
    print("split google mobility data columns into 6 files with similar structure to covid data")
    '''
    print(retail_df)

    
format_mobility_like_covid_data()






