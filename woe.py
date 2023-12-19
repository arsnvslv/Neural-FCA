import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm.notebook import tqdm
tqdm.pandas()
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold

# Функция для подсчета WOE и IV для одной переменной 
def variable_woe_iv(data,variable,target='Converted'):
    lst = [] # тут будет храниься woe для каждой категории 
    categories = data[variable].unique() # наши категории 
    
    for category in categories:
        measures = {
            'category': category,
            'goods': data[(data[variable]==category) & (data[target]==1)].count()[variable], 
            'bads': data[(data[variable]==category) & (data[target]==0)].count()[variable]
        }
        lst.append(measures)
    data_set = pd.DataFrame(lst)
    data_set['goods_dist'] = data_set['goods']/data_set['goods'].sum() # P(x|y=1)
    data_set['bads_dist'] = data_set['bads']/data_set['bads'].sum() # P(x|y=0)
    
    data_set['woe'] = np.log(data_set['goods_dist']/data_set['bads_dist'])
    data_set['woe'] = data_set['woe'].replace({np.inf:0,-np.inf:0})
    
    data_set['iv_prep'] = (data_set['goods_dist'] - data_set['bads_dist'])*data_set['woe']
    iv = data_set['iv_prep'].sum()
    return iv, data_set[['category','woe']]


# Функция для подсчета IV во всех переменных
def iv_calc(data, cols, target='Converted'):
    
    lst = []
    decoding = dict()  
    for col in cols:
        iv, woe_data = variable_woe_iv(data,col)
        decoding[col] = woe_data  
        measure = {
            "variable": col,
            "iv": iv.round(5)
        }
        lst.append(measure)
    return pd.DataFrame(lst).sort_values(by='iv',ascending=False) , decoding 


def plot_categories(data,target='Converted',show='woe',unque=40,height=1000):
    # Первичная подготовка данных
    cols = list(data.nunique()[data.nunique()<40].index) # отбираем переменные где количество уникальных значений меньше 40 
    # (это могут быть и численные переменные)
    categorial_data = data[cols] # выбираем только категориальные переменные
    others = data.drop(cols,axis=1) # оставшиеся переменные 
    categorial_data = categorial_data.fillna('NA') # заполняем пропущенные значения чтоб они отображались на графике  
    grouped = [] # массив со статистиами по каждой переменной 
    cols.remove(target)
    
    # Подсчет WOE
    iv_values, decoding = iv_calc(categorial_data,cols,target = target)
    sorted_cols = iv_values['variable'].values[::-1] # столбцы в порядке уменьшения IV 
    
    # Итерация по переменных, подсчет конверсии 
    for column in sorted_cols:
        count_in_group = categorial_data.loc[:,column].value_counts(normalize=True).round(3) # количество значений каждой категории 
        converted_rate = categorial_data.groupby(column).mean()[target].round(3) # процент положительного таргета 
        statistic = count_in_group.to_frame().join(converted_rate) # соединяем вместе 
        
        statistic['variable'] = column # добавляем столбец с перемменой 
        statistic = statistic.rename(columns = {column:'count'}) # убираем название перменной из название колонки, чтоб конкат работал
        statistic = statistic.merge(decoding[column],left_index=True,right_on='category') # добавляем WOE 
        statistic = statistic.set_index('category') # чтоб нормально отоброжались категории 
        statistic = statistic.sort_values(show) # мы должны отсортировать по той переменной которую хотим показывать
        grouped.append(statistic) 
        
    grouped = pd.concat(grouped) # обьединяем все в один дф 

    
    # Отрисовка графика 
    fig = px.bar(grouped,
                 y='variable',
                 x="count",
                 color=show,
                 barmode='group',
                 hover_name=grouped.index,
                 width=1000,
                 height=height,
                 range_x =[0,1],
                 color_continuous_scale='thermal')
    return fig, grouped , iv_values, categorial_data, others


class KFoldTargetEncoderTrain(BaseEstimator,
                               TransformerMixin):
    def __init__(self,colnames,targetName,
                  n_fold=5,
                  method = 'WOE',
                  discardOriginal_col=False):
        
        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.discardOriginal_col = discardOriginal_col
        self.method = method
        
    def fit(self, X, y=None):
        return self
    
    def transform(self,X):
        X_copy = X.copy()
        for col in tqdm(self.colnames):
            X_copy[col] = X_copy[col].apply(str) # побочная строка, нужно сделать все стрингами для совместимости 
            mean_of_target = X_copy[self.targetName].mean() # на всякий пожарный 
            col_enc_name = col + '_' + f'Kfold_{self.method}_Enc'
            X_copy[col_enc_name] = np.nan
            
            kf = StratifiedKFold(n_splits =self.n_fold,
                                 shuffle = True,
                                 random_state=42)
            
            for tr_ind, val_ind in kf.split(X_copy[col],X_copy[col]):
                
                X_tr, X_val = X_copy.iloc[tr_ind], X_copy.iloc[val_ind]
                
                if self.method == 'WOE':
                    
                    _,decode = variable_woe_iv(X_tr,col,self.targetName) 
                    decode = dict(decode.set_index('category').iloc[:,0]) # чтоб можно было запихнуть в replace
                    X_copy.loc[X_copy.index[val_ind], col_enc_name] = X_val[col].replace(decode)
                    
                if self.method == 'Target':
                    
                    X_copy.loc[X_copy.index[val_ind], col_enc_name] = X_val[col].map(X_tr.groupby(col)[self.targetName].mean())
                    
            X_copy[col_enc_name].fillna(mean_of_target, inplace = True)
                                   
            if self.discardOriginal_col:
                X_copy = X_copy.drop(self.targetName, axis=1)
                                   
        return X_copy
    
    
class KFoldTargetEncoderTest(BaseEstimator, TransformerMixin):
    
    def __init__(self,train,colNames,method = 'WOE'):
        
        self.train = train
        self.colNames = colNames
        self.method = method
        
    def fit(self, X_, y=None):
        return self
    def transform(self,X_):
        X = X_.copy()
        for col in tqdm(self.colNames):
            mean =  self.train[[col, col + '_' + f'Kfold_{self.method}_Enc']].groupby(
                                    col).mean()[col + '_' + f'Kfold_{self.method}_Enc']
            X[col] = X[col].replace(mean)
        return X