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
    
    
###### For predicting most important pillar for prosperity score for each country of each year ######

def get_most_impact_pillars():
    """
    Returns dataframe with column listing the pillar with most
    impact for each country of each year
    
    ---
    
    Example run: df = get_most_impact_pillars()
    
    """
    prosperity_data = get_prosperity_scores()
    pillars = ['busi', 'econ', 'educ', 'envi','gove', 'heal', 'pers', 'safe', 'soci']  
    prosperity_data["most_impact_pillar"] = prosperity_data[pillars].idxmax(axis=1)
    return prosperity_data



###### For predicting impact of categories for each pillar ######

from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt


def remove_star_cols(pillar):
    """
    Return list of categories without "***"
    
    ----
    
    Used in get_impt_cat()
    """
    d = dict_of_datasets()

    # categories
    categories = d[pillar].columns[6:].tolist()
    categories = [ x for x in categories if "_year" not in x ]
    new_categories = []
    
    # remove all the categories with ***
    for category in categories:
        if "***" not in d[pillar][category].unique():
            new_categories.append(category)

    return new_categories

def get_training_testing_data(pillar):
    """
    Returns [pillar] dataset as training and testing dataset.
    
    ----
    
    Used in get_impt_cat()
    """
    
    d = dict_of_datasets()
    new_categories = remove_star_cols(pillar)
    
    # split into training and testing data
    x = d[pillar][new_categories]
    y = d[pillar][pillar]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

def lasso(X_train, y_train, X_test, y_test):
    """
    Returns lasso model to be inputted into SHAP explainer.
    Prints out score of lasso model as well.
    
    ----
    
    Used in get_impt_cat()
    """
    Lreg = linear_model.Lasso(alpha = 0.5)
    clf = Lreg.fit(X_train, y_train) 
    
    # plot predicted values vs true
    #print(Lreg.predict(X_test))
    print("Score of lasso model", clf.score(X_test, y_test))
    return Lreg

def make_shap(model, X_train, X_test):
    """
    Gets shap values of model
    ----
    
    Used in get_impt_cat()
    """
    # ***use X_test or X_train?
    explainer = shap.LinearExplainer(model, X_train, feature_dependence="independent")
    shap_values = explainer.shap_values(X_test)
    return shap_values


def shap_viz_1(df_shap,df):
    """
    Returns bar plot of each categories impact on pillar score
    ----
    
    Used in get_impt_cat()
    """
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
 
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        #print(shap_v[i],df_v[i])
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'lightgreen','red')
    
    # Plot it
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    colorlist = k2['Sign']
    ax = k2.plot.barh(x='Variable',y='SHAP_abs',color = colorlist, figsize=(5,6),legend=False)
    ax.set_xlabel("SHAP Value (Green = Positive Impact, Red = Negative Impact)")
 
    
def shap_viz_2(shap_values, X_test):
    """
    Returns fancy plot of each categories impact on pillar score
    ----
    
    Used in get_impt_cat()
    """
    shap.summary_plot(shap_values, X_test)


def get_impt_cat(pillar, mod):
    """
    Outputs viz of impact of categories.
    
    ----
    
    Example run: get_impt_cat("busi", lasso)
    """
    
    # split into training and testing dataset
    X_train, X_test, y_train, y_test = get_training_testing_data(pillar)
    
    # model 
    model = mod(X_train, y_train, X_test, y_test)
    
    # ***use X_test or X_train?    
    shap_values = make_shap(model, X_train, X_test)
    
    # viz
    # easier to understand version of shap_viz_2
    shap_viz_1(shap_values, X_test)
    # cooler looking version of shap_viz_1
    #shap_viz_2(shap_values, X_test)
    
    
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
  
