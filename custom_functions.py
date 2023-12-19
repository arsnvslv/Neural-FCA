import pandas as pd
import numpy as np
import seaborn as sns
import re
import math
import scipy.stats as stats
import scipy
import matplotlib.pyplot as plt
from time import strptime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from astropy.stats import freedman_bin_width


def plot_feature_importance(importance, names, model_type,save=False, name='importance'):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(10, 8))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    if save:
        plt.savefig(name)
    plt.show()
    return fi_df


class MonthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,column_name='Month'):
        self.column_name = column_name
        
    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        X_ = X.copy()
        X_[self.column_name] = X_[self.column_name].apply(lambda x: strptime(x,'%B').tm_mon)
        return X_
    
def reweighting(X, y):
    """
    Функция для ребалансировки классов
    делаю столько же наблюдений сколько у самого частого класса
    """
    target = y.name
    train_data = pd.concat([X,y],axis=1)
    largest_class = y.value_counts()[0]
    new_obs = []
    for name_class, num_class in y.value_counts().iteritems():
        mask = train_data[target] == name_class
        balanced = train_data[mask].sample(largest_class - num_class, replace=True, random_state=42)
        new_obs.append(balanced)
    new_obs = pd.concat(new_obs)
    new_train = pd.concat([train_data, new_obs])
    X_train_balanced, y_train_balanced = new_train.drop([target], axis=1).reset_index(drop=True), new_train[
        target].reset_index(drop=True)
    return X_train_balanced, y_train_balanced

def f(x): # функция которая разибвает историю на год и месяц, а потом переводит все в месяца, пропущенные значения не трогает 
    if not isinstance(x,float):
        history =  re.findall('\d+',x)
        return int(history[0])*12 + int(history[1])

def custom_fill(arr, forward=True):
    """
    функция которая заполняет пропущенное значение в массиве предыдущим значением(если оно есть) на единицу больше,
    остальные пропускает
    """
    # если у нас заполнение вперед, ничего не делаем pl(то что будем добавлять =1)
    # в противном случае разворачиваем массив и будем вычитать (pl=-1)
    arr, pl = (arr, 1) if forward else (arr[::-1], -1)
    last_valid = -1 # Заводим переменную для последнего валидного значения 
    for index, value in enumerate(arr):
        if math.isnan(value) and last_valid != -1: # если невалидное и до этого мы уже знаем историю  
            arr[index] = last_valid
        elif not math.isnan(value): 
            last_valid = value + pl # если валидное то берем его как новое и прибавляем единицу 
    return arr if forward else arr[::-1]

def preprocessing_history(data): 
    data['Credit_History_Age'] = data['Credit_History_Age'].apply(f)
    for customer, values in data.groupby('Customer_ID'):
        updated = custom_fill(values.Credit_History_Age.values) # заполняем вперед 
        updated = custom_fill(updated, forward = False) # заполняем назад 
        data.loc[values.index,'Credit_History_Age'] = updated
    return None

def reomve_outliners_by_group_min_max(df, groupby, column):
    """
    1. Найдем моду для группы(кастомера)
    2. Найдем минимум и максимум из этих значений 
    3. Удалим все что больше и меньше минимума и максимума 
    """
    # Функция для замены пропусков на Nan
    def make_group_NaN_and_fill_mode(df, groupby, column, inplace=True):
        df_dropped = df[df[column].notna()].groupby(groupby)[column].apply(list)
        x, y = df_dropped.apply(lambda x: stats.mode(x)).apply([min, max])
        mini, maxi = x[0][0], y[0][0]

        # Замена
        df[column] = df[column].apply(lambda x: np.NaN if ((x<mini)|(x>maxi)) else x)

    return make_group_NaN_and_fill_mode(df, groupby, column)


def binarizer(X,list_bins=5,columns=None,method='simple'):
    X_ = X.copy()
    
    if not columns: 
        columns = X_.select_dtypes(['int','float']).columns
        
    if isinstance(list_bins,int): 
        list_bins = [list_bins]*len(columns)
    f_binning = pd.cut if method=='simple' else pd.qcut
    
    for bins,column in zip(list_bins,columns):
        X_[column] = pd.cut(X_[column],bins)
        
    ct = ColumnTransformer([('enc', OneHotEncoder(),X_.columns)],
                       remainder='passthrough')
    X_ = ct.fit_transform(X_)
    
    if scipy.sparse.issparse(X_): 
        X_ = pd.DataFrame.sparse.from_spmatrix(X_, columns = ct.get_feature_names_out())
    else: 
        X_ = pd.DataFrame(X_, columns = ct.get_feature_names_out())
        
    patt = re.compile(r'enc__||remainder__')
    names = [patt.sub('', feature) for feature in X_.columns]
    X_.columns = names
        
    X_ = X_.astype(bool)
    X_.index = X_.index.map(str)
    return X_

def find_num_of_bins(X):
    total_bins = []
    for column in X.columns:
        _, bins =  freedman_bin_width(X.loc[:,column],return_bins=True)
        total_bins.append(len(bins))
    # simple heuristic to reduce number of bins 
    total_bins=np.array(total_bins)
    total_bins[total_bins<100] = 5
    total_bins[np.where((total_bins >= 100) & (total_bins <=1000))] = 7
    total_bins[total_bins>1000]=10
    return total_bins