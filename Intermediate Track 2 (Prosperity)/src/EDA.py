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

 ###### For finding the top 5 countries with most growth in prosperity ######

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

