import numpy as np
import csv
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from dtw import *
from scipy import stats
from dtaidistance import dtw as dtw_visualize
from dtaidistance import dtw_visualisation as dtwvis

def moving_average(data, n=7) :
    ret = np.cumsum(data, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def load_all_covid_confirmed_cases_data():
    covid_data_dict = {}
    fips_dict = {}
    covid_skip_start_date = 24  #skip some early dates to have same date length with mobility data
    covid_skip_end_date = -5
    with open('time_series_covid19_confirmed_US.csv','r') as csvfile:
        data = csv.reader(csvfile, delimiter = ',')
        header = next(data)
        for row in data:
            covid_data_dict[row[10]] = np.array(row[11+covid_skip_start_date:covid_skip_end_date], int)
            fips_dict[row[10]] = float(row[4])
    return covid_data_dict,fips_dict

def covid_cases_for_county(covid_data_dict, county_key_word):
    county_covid_cases = covid_data_dict[county_key_word]
    county_covid_cases_delta = np.diff(county_covid_cases)

    fig = plt.figure(figsize=(24,8))
    plt.plot(county_covid_cases_delta)
    plt.xlabel("date")
    plt.ylabel("new death cases")
    plt.title("number of new covid cases in " + county_key_word)
    fig.savefig('all_us_counties_results/' + county_key_word + '_new_confirmed_cases.png')
    
    county_covid_cases_delta = moving_average(county_covid_cases_delta)
    county_covid_cases_delta = stats.zscore(county_covid_cases_delta)
    fig = plt.figure(figsize=(24,8))
    plt.plot(county_covid_cases_delta)
    plt.xlabel("date")
    plt.ylabel("new death cases")
    plt.title("filtered number of new covid cases in " + county_key_word)
    fig.savefig('all_us_counties_results/' + county_key_word + 'filtered_new_confirmed_cases.png')

    return county_covid_cases_delta
  
def compute_dtw(fips, county_name, current_county_covid_cases_delta, major_county_covid_cases_delta, mobility_data, mobility_data_type_index):
    mobility_data_type_name = ['recreation', 'groecry', 'park', 'transit', 'work', 'residential']
    mobility_data = moving_average(mobility_data)
    mobility_data = stats.zscore(mobility_data)

    distance_with_itself = dtw_visualize.distance(current_county_covid_cases_delta, mobility_data)
    distance_with_major_city = dtw_visualize.distance(major_county_covid_cases_delta, mobility_data)

    fig = plt.figure(figsize=(24,8))
    plt.plot(current_county_covid_cases_delta)
    plt.xlabel("date")
    plt.ylabel("new death cases")
    plt.title("in DTW filtered number of new covid cases in " + county_name)
    fig.savefig('all_us_counties_results/' + county_name + '_filtered_new_confirmed_cases_dtw.png')

    d, paths = dtw_visualize.warping_paths(current_county_covid_cases_delta, mobility_data, window=25, psi=2)
    best_path = dtw_visualize.best_path(paths)
    dtwvis.plot_warpingpaths(current_county_covid_cases_delta, mobility_data, paths, best_path, filename="all_us_counties_results/dtw_" + county_name +".png")
    
    with open('all_us_counties_results/confirmed_cases_vs_dtw_distance.csv','a') as csvfile:
        csvfile.write(str(fips) +","+ county_name + "," + str(distance_with_itself) + "," + str(distance_with_major_city) + "\n")



def main():
    mobility_data_type_index = 5
    filtered = False
    covid_data_dict,fips_dict = load_all_covid_confirmed_cases_data()
    NYC_covid_cases_delta = covid_cases_for_county(covid_data_dict, "New York City, New York, US")
    
    current_county = ""
    mobility_array = np.zeros(101)
    mobility_array_index = 0
    
    with open('Google_US_Mobility_Report_toyset.csv','r') as csvfile:
        data = csv.reader(csvfile, delimiter = ',')
        header = next(data)
        for line in data:
            if line[3]:
                county_key = ' '.join(line[3].split(' ')[:-1]) +', ' + line[2] + ', US'
                if county_key == current_county:
                    try:
                        mobility_array[mobility_array_index] = int(line[5+mobility_data_type_index])
                    except ValueError:  #the mobility data doesn't have this info for this county
                        pass
                    mobility_array_index += 1
                    if mobility_array_index>101:
                        print("mobility_array_index exceed 101!")
                        mobility_array_index = 0
                else:
                    if current_county:
                        try:
                            current_county_covid_cases_delta = covid_cases_for_county(covid_data_dict, current_county)
                            fips = fips_dict[current_county]
                            fig = plt.figure(figsize=(24,8))
                            plt.plot(mobility_array)
                            plt.xlabel("date")
                            plt.ylabel("mobility")
                            plt.title("mobility in " + current_county)
                            fig.savefig('all_us_counties_results/' + current_county + '_mobility.png')
                            compute_dtw(fips, current_county, current_county_covid_cases_delta, NYC_covid_cases_delta,  mobility_array, mobility_data_type_index)
                        except:
                            pass
                    mobility_array_index = 0
                    mobility_array = np.zeros(101)
                    #some of the mobility data don't have residential mobility
                    try:
                        mobility_array[mobility_array_index] = int(line[5+mobility_data_type_index])
                    except ValueError:
                        pass
                    mobility_array_index += 1                
                current_county = county_key



if __name__ == "__main__":
    main()





