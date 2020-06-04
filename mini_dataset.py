import numpy as np 
import pandas as pd 
import csv
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import datetime
from correlate_mobility_and_cases import new_cases_timeseries_table
from correlate_mobility_and_cases import total_cases
from correlate_mobility_and_cases import google_county_naming_2_covid_county_naming
from scipy.stats import spearmanr
import math


def google_date_2_covid_date(google_date):
    date_time = datetime.datetime.strptime(google_date, '%Y-%m-%d')
    covid_date = datetime.datetime.strftime(date_time, '%-m/%-d/%y')
    return covid_date

def first_row_2_header(df):
    new_header = df.iloc[0] #grab the first row for the header
    df = df[1:] #take the data less the header row
    df.columns = new_header #set the header row as the df header
    return df

def create_mini_dataset():
    new_cases = new_cases_timeseries_table()

    mobility_table = pd.read_csv('data/location_table.csv')
    mobility_counties = mobility_table.county.values
    mobility_counties = mobility_counties[1:] # remove the NAN thats in the first row
    mobility_states = mobility_table.state.values
    mobility_states = mobility_states[1:] # remove the NAN thats in the first row

    google_df = pd.read_csv("data/Google_Global_Mobility_Report.csv")
    google_df = google_df.loc[google_df['country_region'] == "United States"]



    latest_start_date = None 
    latest_start_date_string = None
    earliest_end_date = None 
    earliest_end_date_string = None
    # determine the common start and end dates for each single county 


    #for state, county in zip(mobility_states, mobility_counties):
    #    this_state = google_df.loc[google_df['sub_region_1'] == str(state)]
    #    this_county = this_state.loc[this_state['sub_region_2'] == str(county)]

    state = "Washington"
    this_state = google_df.loc[google_df['sub_region_1'] == str(state)]

    # convert 1 master dataset into 6 sub datasets, reformatted to look like covid data
    retail_rows = []
    grocery_rows = []
    park_rows = []
    transit_rows = []
    workplaces_rows = []
    residential_rows = []

    for county in mobility_counties:
        this_county = this_state.loc[this_state['sub_region_2'] == str(county)]
        this_county['datetimes'] =  pd.to_datetime(this_county['date'], format='%Y-%m-%d')

        if this_county.empty:
            continue
        col_dates = this_county.date.values
        col_dates = list(col_dates)
        if len(col_dates) < 72:
            continue

        # reformat name to match those present in the covid dataset
        county = google_county_naming_2_covid_county_naming(county)

        print("for county ", county, " we have ", col_dates[0], col_dates[len(col_dates)-1], len(col_dates))
        start_date = col_dates[0]
        end_date = col_dates[len(col_dates)-1]
        start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        start_date = datetime.datetime.strftime(start_datetime, '%m/%d/%y')
        end_date = datetime.datetime.strftime(end_datetime, '%m/%d/%y')
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
        


        # separate 6 columns from google dataset to 6 separated 
        retail_row = this_county.retail_and_recreation_percent_change_from_baseline.values
        retail_rows.append([state, county] + list(retail_row))
        
        grocery_row = this_county.grocery_and_pharmacy_percent_change_from_baseline.values
        grocery_rows.append([state, county] + list(grocery_row))

        park_row = this_county.parks_percent_change_from_baseline.values
        park_rows.append([state, county] + list(park_row))

        transit_row = this_county.transit_stations_percent_change_from_baseline.values
        transit_rows.append([state, county] + list(transit_row))

        workplaces_row = this_county.workplaces_percent_change_from_baseline.values
        workplaces_rows.append([state, county] + list(workplaces_row))

        residential_row = this_county.residential_percent_change_from_baseline.values
        residential_rows.append([state, county] + list(residential_row))


    for i in range(len(col_dates)):
        col_dates[i] = google_date_2_covid_date(col_dates[i])

    retail_rows = [["state", "county"] + col_dates] + retail_rows

    retail_rows = [["state", "county"] + col_dates] + retail_rows
    grocery_rows = [["state", "county"] + col_dates] + grocery_rows
    park_rows = [["state", "county"] + col_dates] + park_rows
    transit_rows = [["state", "county"] + col_dates] + transit_rows
    workplaces_rows = [["state", "county"] + col_dates] + workplaces_rows
    residential_rows = [["state", "county"] + col_dates] + residential_rows

    retail_df = first_row_2_header(pd.DataFrame(retail_rows))
    grocery_df = first_row_2_header(pd.DataFrame(grocery_rows))
    park_df = first_row_2_header(pd.DataFrame(park_rows))
    transit_df = first_row_2_header(pd.DataFrame(transit_rows))
    workplaces_df = first_row_2_header(pd.DataFrame(workplaces_rows))
    residential_df = first_row_2_header(pd.DataFrame(residential_rows))

    save_path = "data/washington_counties"
    retail_df.to_csv("{}/retail.csv".format(save_path))
    grocery_df.to_csv("{}/grocery.csv".format(save_path))
    park_df.to_csv("{}/park.csv".format(save_path))
    transit_df.to_csv("{}/transit.csv".format(save_path))
    workplaces_df.to_csv("{}/workplaces.csv".format(save_path))
    residential_df.to_csv("{}/residential.csv".format(save_path))
    print("split google mobility data columns into 6 files with similar structure to covid data")

        
    print("we have data for all locations from ", latest_start_date_string, " to ", earliest_end_date_string)



def spearman_correlation_on_mini():
    new_cases = new_cases_timeseries_table()
    total_cases_df = total_cases()

    washington_workplace_mobility = pd.read_csv('data/washington_counties/workplaces.csv')
    dates = list(washington_workplace_mobility.columns[3:])


    plot_county_names = []
    plot_corr = []
    p_vals = []
    for state, county in zip(washington_workplace_mobility.state.values, washington_workplace_mobility.county.values):
        
        covid_county_timeseries = total_cases_df[(total_cases_df['state'] == state) & (total_cases_df['county'] == county)]
        workplace_mobility = washington_workplace_mobility[(washington_workplace_mobility['state'] == state) & (washington_workplace_mobility['county'] == county)]

        print(state, county)

        this_county_covid_total_cases = covid_county_timeseries[dates].values[0]
        this_county_workplace_mobility = workplace_mobility.values[0][3:]


        # lowpass filtering
        '''
        print(len(this_county_workplace_mobility))
        window_size = 7
        workplace_mobility_series = pd.Series(this_county_workplace_mobility)
        windows = workplace_mobility_series.rolling(window_size)
        moving_averages = windows.mean()
        this_county_workplace_mobility = moving_averages.tolist()
        this_county_workplace_mobility = this_county_workplace_mobility[window_size - 1:]
        this_county_covid_total_cases = this_county_covid_total_cases[window_size - 1:]
        dates = dates[window_size - 1:]
        '''

        spearman_results = spearmanr(this_county_covid_total_cases, this_county_workplace_mobility) 
        corr = spearman_results.correlation
        p = spearman_results.pvalue

        if not math.isnan(corr):
            print(corr, p)
            plot_county_names.append(county)
            plot_corr.append(-1*corr)
            p_vals.append(p)


    min_p_val = min(p_vals)
    max_p_val = max(p_vals)

    print(min_p_val, max_p_val)

    width = 0.35
    x = np.arange(len(plot_county_names))  # the label locations
    fig = plt.figure(figsize=(15,5),facecolor='w') 
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.bar(x - width/2, plot_corr, width)
    plt.title('Spearman Correlation of Daily Negative Mobility and Covid Cases by County in Washington State')
    plt.xticks(x, plot_county_names, fontsize=8, rotation=45)
    plt.xlabel('Washington State Counties')
    plt.ylabel('Spearman Correlation Coefficient')
    plt.show()

    

def dtw():
    from dtw import dtw,accelerated_dtw

    new_cases = new_cases_timeseries_table()
    total_cases_df = total_cases()

    washington_workplace_mobility = pd.read_csv('data/washington_counties/workplaces.csv')
    dates = list(washington_workplace_mobility.columns[3:])


    plot_county_names = []
    plot_corr = []
    p_vals = []
    for state, county in zip(washington_workplace_mobility.state.values, washington_workplace_mobility.county.values):
        
        covid_county_timeseries = total_cases_df[(total_cases_df['state'] == state) & (total_cases_df['county'] == county)]
        workplace_mobility = washington_workplace_mobility[(washington_workplace_mobility['state'] == state) & (washington_workplace_mobility['county'] == county)]

        print(state, county)

        this_county_covid_total_cases = covid_county_timeseries[dates].interpolate().values[0]
        this_county_workplace_mobility = workplace_mobility.interpolate().values[0][3:]

        if county == 'King':
            d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(this_county_workplace_mobility, this_county_covid_total_cases, dist='euclidean')

            plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
            plt.plot(path[0], path[1], 'w')
            plt.xlabel('Subject1')
            plt.ylabel('Subject2')
            plt.title(f'DTW Minimum Path with minimum distance: {np.round(d,2)}')
            plt.show()


#create_mini_dataset()
'''
states = ['Washington', 'New York', 'Massachusetts', 'California']
counties = ['King', 'New York City', 'Sufolk', 'San Francisco']
for state in states:
    for county in counties:
        spearman_correlation_on_mini(state, county)
        '''
spearman_correlation_on_mini()
#dtw()
