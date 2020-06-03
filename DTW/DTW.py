import numpy as np
import csv
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from dtw import *
from scipy import stats
from dtaidistance import dtw as dtw_visualize
from dtaidistance import dtw_visualisation as dtwvis



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# def moving_average_filter(data):
    



def load_and_plot_covid_confirmed_cases_data(county_key_word, filtering=False, filter_thres=0.3):
    covid_data_dict = {}
    covid_skip_start_date = 24  #skip some early dates to have same date length with mobility data
    covid_skip_end_date = -5
    with open('time_series_covid19_confirmed_US.csv','r') as csvfile:
        data = csv.reader(csvfile, delimiter = ',')
        header = next(data)
        for row in data:
        	covid_data_dict[row[10]] = np.array(row[11+covid_skip_start_date:covid_skip_end_date], int)
    xticks_date_string = [date[:-3] for date in header[11+covid_skip_start_date:covid_skip_end_date]]
    county_covid_cases = covid_data_dict[county_key_word]
    xticks_date_index = range(0,len(county_covid_cases), 5)
    xticks_date = [xticks_date_string[i] for i in xticks_date_index]

    county_covid_cases_delta = np.diff(county_covid_cases)
    fig = plt.figure(figsize=(24,8))
    plt.plot(county_covid_cases_delta)
    plt.xticks(xticks_date_index, xticks_date, rotation=20) 
    plt.xlabel("date")
    plt.ylabel("new death cases")
    plt.title("number of new covid cases in " + county_key_word)
    fig.savefig('results/new_confirmed_cases.png')
    if filtering:
        county_covid_cases_delta = butter_lowpass_filter(county_covid_cases_delta, filter_thres, 1)
        fig = plt.figure(figsize=(24,8))
        plt.plot(county_covid_cases_delta)
        plt.xticks(xticks_date_index, xticks_date, rotation=20) 
        plt.xlabel("date")
        plt.ylabel("new confirmed cases")
        plt.title("filtered number of new covid cases in " + county_key_word)
        fig.savefig('results/new_confirmed_cases_filtered.png')

    county_covid_cases_delta = stats.zscore(county_covid_cases_delta)
    return county_covid_cases_delta



def load_and_plot_mobility(mobility_data_type_index, filtering = False, filter_thres = 0.3):
    with open('mobility_king_county.csv','r') as csvfile:
        data = csv.reader(csvfile, delimiter = ',')
        mobility_data = list(data)

    mobility_data = np.array(mobility_data)
    mobility_data = mobility_data[:,5:11]
    mobility_data = np.array(mobility_data, int)

    mobility_data_type_name = ['recreation', 'groecry', 'park', 'transit', 'work', 'residential']
    mobility_type_data = mobility_data[:, mobility_data_type_index]
    fig = plt.figure(figsize=(24,8))
    plt.plot(mobility_type_data)
    # plt.xticks(xticks_date_index, xticks_date, rotation=20) 
    plt.xlabel("date")
    plt.ylabel("mobility rate change")
    plt.title(mobility_data_type_name[mobility_data_type_index] + " rate in King County")
    fig.savefig('results/'+mobility_data_type_name[mobility_data_type_index]+'_mobility.png')

    if filtering:
        mobility_type_data = butter_lowpass_filter(mobility_type_data, filter_thres, 1)
        fig = plt.figure(figsize=(24,8))
        plt.plot(mobility_type_data)
        # plt.xticks(xticks_date_index, xticks_date, rotation=20) 
        plt.xlabel("date")
        plt.ylabel("mobility rate change")
        plt.title(mobility_data_type_name[mobility_data_type_index] + " rate filtered in King County")
        fig.savefig('results/'+mobility_data_type_name[mobility_data_type_index]+'_mobility_filtered.png')

    mobility_type_data = stats.zscore(mobility_type_data)
    return mobility_type_data


def compute_dtw(county_covid_cases_delta, mobility_type_data, mobility_data_type_index):
    print(county_covid_cases_delta.shape)
    print(mobility_type_data.shape)
    mobility_data_type_name = ['recreation', 'groecry', 'park', 'transit', 'work', 'residential']
    distance = dtw_visualize.distance(county_covid_cases_delta, mobility_type_data)
    print(distance)
    with open('results/confirmed_cases_dtw_distance.csv','a') as csvfile:
        csvfile.write(mobility_data_type_name[mobility_data_type_index] + " " + str(distance) + "\n")

    d, paths = dtw_visualize.warping_paths(county_covid_cases_delta, mobility_type_data, window=25, psi=2)
    best_path = dtw_visualize.best_path(paths)
    dtwvis.plot_warpingpaths(county_covid_cases_delta, mobility_type_data, paths, best_path, filename="results/dtw_" + mobility_data_type_name[mobility_data_type_index] +".png")
    
    plt.title("confirmed cases vs " + mobility_data_type_name[mobility_data_type_index] + " DTW")
    # plt.show()

    # alignment = dtw(county_covid_cases_delta, mobility_type_data, keep_internals=True)
    # fig = alignment.plot(type="threeway")
    

def main():
    mobility_data_type_index = 5
    filtered = False
    county_covid_cases_delta = load_and_plot_covid_confirmed_cases_data("King, Washington, US", filtering = filtered)
    mobility_type_data = load_and_plot_mobility(mobility_data_type_index, filtering = filtered)
    compute_dtw(county_covid_cases_delta, mobility_type_data, mobility_data_type_index)
    # compute_dtw(county_covid_cases_delta[40:], mobility_type_data[30:-10], mobility_data_type_index)

if __name__ == "__main__":
    main()





