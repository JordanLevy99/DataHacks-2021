###### For predicting impact of categories for each pillar ######

import numpy as np
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, ElasticNet
from src.etl import *

class PillarExplainer:
    def __init__(self, mod='lasso', random_state=42, test_size=0.3, alpha=0.5):
        self.pillar_train, self.pillar_test = read_in_pillars(remove_star=True)
        self.random_state = random_state
        self.test_size = test_size
        self.alpha = alpha
        self.mod_str = mod
        self.models = {}
        self.pillars = list(self.pillar_train.keys())
    
#     def _remove_star_cols(self, data):
#         """
#         Return data without columns that only have "***"

#         ----

#         Used in get_impt_cat()
#         """
#         return {key: data[key].loc[:, ~(data[key].loc[0].apply(lambda x: '***' in str(x)))] for key in data}


    def get_train_val(self, pillar):
        """
        Returns [pillar] dataset as training and testing dataset.

        ----

        Used in get_impt_cat()
        """
    #     random_state = 42
        filt_data = self.pillar_train[pillar].loc[:, ~((self.pillar_train[pillar].columns.str.contains('year')) | (self.pillar_train[pillar].columns == pillar))]
        X = filt_data[filt_data.columns[5:]]
        y = self.pillar_train[pillar][pillar]

        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=self.random_state, test_size=self.test_size)
        return X_train, X_val, y_train, y_val


    def lasso(self, pillar, X_train, y_train, X_val, y_val):
        """
        Returns lasso model to be inputted into SHAP explainer.
        Returns score of lasso model as well.

        ----

        Used in get_impt_cat()
        """
        Lreg = Lasso(alpha = self.alpha)
        Lreg.fit(X_train, y_train)
        self.models[pillar] = Lreg
        # plot predicted values vs true
        #print(Lreg.predict(X_test))
        score = Lreg.score(X_val, y_val)
        print("Score of lasso model", score)
        return score, Lreg

    def elastic(self, pillar, X_train, y_train, X_val, y_val):
        en = ElasticNet()
        en.fit(X_train, y_train)
        self.models[pillar] = en
        score = en.score(X_val, y_val)
        print(f'Elastic net model score is {score} on test data')

        # consider adding y_pred here


        return score, en


    def make_shap(self, model, X_train, X_val):
        """
        Gets shap values of model
        ----

        Used in get_impt_cat()
        """
        # ***use X_test or X_train?
        explainer = shap.LinearExplainer(model, X_train, feature_dependence="independent")
        shap_values = explainer.shap_values(X_val)
        return shap_values
    
    
    
    def shap_viz_1(self, df_shap,df):
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

    
    def shap_viz_2(self, shap_values, X_val):
        """
        Returns fancy plot of each categories impact on pillar score
        ----

        Used in get_impt_cat()
        """
        shap.summary_plot(shap_values, X_val)


        
    def get_model(self, pillar):
        # split into training and testing dataset
        X_train, X_val, y_train, y_val = self.get_train_val(pillar)

        # model and score of model 
        score, model = eval('self.'+self.mod_str)(pillar, X_train, y_train, X_val, y_val)
        #^^utilize score???
        return model
    
    def get_models(self):
        for pillar in self.pillars:
            self.get_model(pillar)
        print('Done, check self.models')
    
    def get_impt_cat(self, pillar):
        """
        Outputs viz of impact of categories.

        ----

        Example run: self.get_impt_cat("busi")
        """

        model = self.get_model(pillar)
        
        # ***use X_test or X_train?    
        shap_values = self.make_shap(model, X_train, X_val)

        # viz
        # easier to understand version of shap_viz_2
        self.shap_viz_1(shap_values, X_val)
        # cooler looking version of shap_viz_1
        #shap_viz_2(shap_values, X_test)

