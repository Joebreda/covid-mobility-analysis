import pandas as pd 
import numpy as np

def get_number_apple_counties():
    df = pd.read_csv('data/location_table.csv')
    print(df.apple.astype(bool).sum(axis=0))

get_number_apple_counties()