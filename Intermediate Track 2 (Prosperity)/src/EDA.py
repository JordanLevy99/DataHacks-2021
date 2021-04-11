def dict_of_datasets():
     """
     Returns a dictionary of the datasets
     where key is pillar name and value is the [pillar]_train.csv
     
     ---
     
     Example run: d = dict_of_datasets(), get busi dataset by d['busi']
     
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
        if file.startswith('.') or 'test' in file: continue
        # print(fpath+file)
        data = pd.read_csv(fpath+file)
        pillar = file.split('/')[-1][:4]
        if len(prosperity_data) == 0:
            prosperity_data = data[['country', 'isonum', 'isocode', 'year', pillar]]
            continue
        prosperity_data = prosperity_data.merge(data[['country', 'year', pillar]], on=['country', 'year'], how='left')
    ### consider writing to csv, os.mkdir('Processed Data') <-- if not exists...
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

def read_in_pillars(remove_star=True):
    train_data = {}
    test_data = {}
    fpath = 'Datasets/'
    for file in sorted(os.listdir(fpath)):
        if file.startswith('.'): continue
        pillar = file.split('/')[-1][:4]
        if 'test' in file:
            test_data[pillar] = pd.read_csv(fpath+file)
        elif 'train' in file:
            train_data[pillar] = pd.read_csv(fpath+file)
    def remove_star_cols(data):
        """
        Return data without columns that only have "***"
        ----
        Used in get_impt_cat()
        """
        return {key: data[key].loc[:, ~(data[key].loc[0].apply(lambda x: '***' in str(x)))] for key in data}
    if remove_star:
        train_data = remove_star_cols(train_data)
        test_data = remove_star_cols(test_data)
    
    return train_data, test_data

 ###### Prompt 2: For finding the top 5 countries with most growth in prosperity ######

 def most_growth_5(most=True):
     """
     Gets you the top 5 countries with most growth if most is True 
     or the bottom 5 countries with regressing growth if most is False.
     
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

