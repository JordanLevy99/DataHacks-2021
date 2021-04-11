import os 
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def dict_of_datasets():
    """
    Returns a dictionary of the datasets
    where key is pillar name and value is the [pillar]_train.csv
    """
    d = {}
    directory = 'Datasets/'
    for filename in sorted(os.listdir(directory)):
        if filename.endswith("_train.csv"):
            name = filename.split("_")[0]
            d["{0}".format(name)] = pd.read_csv(os.path.join(directory, filename), index_col=0)
            
    return d

def generate_prosperity():
    prosperity_data = pd.DataFrame()
    fpath = 'Datasets/'
    for file in sorted(os.listdir(fpath)):

        #if file.startswith('.') or 'test' in file: continue
        if 'test' in file: continue
        print(fpath+file)
        data = pd.read_csv(fpath+file)
        pillar = file.split('/')[-1][:4]
        if len(prosperity_data) == 0:
            prosperity_data = data[['country', 'isonum', 'isocode', 'year', pillar]]
            continue
        prosperity_data = prosperity_data.merge(data[['country', 'year', pillar]], on=['country', 'year'], how='left')
    return prosperity_data

def get_prosperity_scores():
    """
    Calculate prosperity score of each country for each year.
    Returns dataframe w/ pillar score & overall prosperity score
    
    ---
    
    Example run: prosperity_data = get_prosperity_scores()
    
    """
    pillars = ['busi', 'econ', 'educ', 'envi','gove', 'heal', 'pers', 'safe', 'soci']  
    prosperity_data = generate_prosperity()
    prosperity_data["prosperity"] = prosperity_data[pillars].mean(axis=1)
    return prosperity_data

def most_growth_5(most=True):
    """
    Gets you the top 5 countries with most growth. 
    
    ---
    
    Example run: top5 = most_growth_5()
    
    ---
    
    ** May need to change end to 2016 (add on to 
    generate_prosperity in that case)
    """
    start = 2007
    end = 2014
    
    # prosperity data with prosperity scores
    prosperity_data = get_prosperity_scores()
    pillars = ['busi', 'econ', 'educ', 'envi','gove', 'heal', 'pers', 'safe', 'soci']  

    # average out all the pillars to get 
    # prosperity score for a country in a certain year
    prosperity_data["prosperity"] = prosperity_data[pillars].mean(axis=1)

    # filter for "end year" and "start year" should it be 2014 or 2016?
    prosperity_data_07= prosperity_data[prosperity_data["year"] == start]
    prosperity_data_14= prosperity_data[prosperity_data["year"] == end]

    # calculate the first part of CAGR, Vfinal/Vbegin ** double check
    prosperity_data_14["temp"] = prosperity_data_14.prosperity.values / prosperity_data_07.prosperity.values


    def CAGR(row):
        c = 1/((end - start)+1)
        if row > 0:
            return (row ** c) - 1
        else:
            temp = abs(row) ** c
            return -1 * temp - 1

    # calculate the second part of CAGR
    prosperity_data_14["CAGR"] = prosperity_data_14["temp"].apply(lambda x: CAGR(x))
    if most:
        return prosperity_data_14.sort_values(by="CAGR", ascending = False)[:5].country.tolist()
    else:
        return prosperity_data_14.sort_values(by="CAGR", ascending = True)[:5].country.tolist()


#### Get dataframe for viz comparison between pillar rate of change for top 5 growing ####
####    countries and top 5 regressing countries ####
def top_bottom_growers(top = True):
    """
    Returns the changes in pillars for top 5 growing countries if top = True
    or for bottom 5 growing (regressing) countries if top = False
    
    """
    
    def pillar_growth_rates(prosperity_data):
        """
        Returns the change in each pillar over time 
        ---
        Example run: pillar_growth_rates(df)
        """
        start = 2007
        end = 2014

        prosperity_data = prosperity_data[['country','year', 'busi', 'econ', 'educ', 'envi', 'gove', 'heal', 'pers', 'safe', 'soci']]
        pillars = ['busi', 'econ', 'educ', 'envi','gove', 'heal', 'pers', 'safe', 'soci']  

        # filter for "end year" and "start year"
        prosperity_data_07= prosperity_data[prosperity_data["year"] == start]
        prosperity_data_14= prosperity_data[prosperity_data["year"] == end]

        for pillar in pillars:    
            # calculate the first part of CAGR, Vfinal/Vbegin 
            prosperity_data_14["temp"] = prosperity_data_14[pillar].values / prosperity_data_07[pillar].values


            def CAGR(row):
                c = 1/((end - start)+1)
                if row > 0:
                    return (row ** c) - 1
                else:
                    temp = abs(row) ** c
                    return -1 * temp - 1

            # calculate the second part of CAGR
            prosperity_data_14["CAGR_{0}".format(pillar)] = prosperity_data_14["temp"].apply(lambda x: CAGR(x))

        return prosperity_data_14.iloc[:,-9:]

    top_5_growth = most_growth_5()
    bottom_5_growth = most_growth_5(False)
    
    prosperity_data = get_prosperity_scores()
    
    top_5_growers = prosperity_data[prosperity_data["country"].isin(top_5_growth)]
    bottom_5_growers = prosperity_data[prosperity_data["country"].isin(bottom_5_growth)]
    
    if top:   
        return pillar_growth_rates(top_5_growers)
    else:
        return pillar_growth_rates(bottom_5_growers)
