#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:17:03 2018

@author: yuli510
"""
import pandas as pd
import numpy as np
import math
import copy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error



def pivotTrainSet(df_train_ratings):
    df_train_ratings_pivot1 = pd.pivot_table(df_train_ratings[['userId','movieId','rating']],columns=['movieId'],index=['userId'],values='rating',fill_value=float("nan"))
    df_train_ratings_pivot2 = df_train_ratings_pivot1.fillna(df_train_ratings_pivot1.mean())
    df_train_ratings_pivot = df_train_ratings_pivot2.sub(df_train_ratings_pivot1.mean())
    return df_train_ratings_pivot, df_train_ratings_pivot1
    
def getIndexMap(df_train_ratings_pivot):
    map_movies_train = dict(enumerate(list(df_train_ratings_pivot.columns)))
    map_users_train = dict(enumerate(list(df_train_ratings_pivot.index)))
    map_movies_train_mid_index = dict(map(lambda t:(t[1],t[0]),map_movies_train.items()))   
    map_users_train_uid_index = dict(map(lambda t:(t[1],t[0]),map_users_train.items()))
    return  map_movies_train_mid_index, map_users_train_uid_index 


def getMostSimUserDict(matrix_user_sim, df_train_ratings_pivot,k,values_train_ratings):
    dict_user_most_sim = dict()
    for i in range(len(values_train_ratings)):
        dict_user_most_sim[i] = sorted(enumerate(list(matrix_user_sim[i])),key = lambda x:x[1], reverse=True)[1:k+1]
    return dict_user_most_sim
    
def getValuesUserRecormmend(values_train_ratings,values_ratings0,dict_user_most_sim,baseline_users_ratingMean_train,baseline_movies_ratingMean_train,mu):
    values_user_recommend = np.zeros((len(values_train_ratings),len(values_train_ratings[0])),dtype=np.float32)
    for i in range(len(values_train_ratings)):
        for j in range(len(values_train_ratings[i])):
            if not math.isnan(values_ratings0[i][j]):
                values_user_recommend[i,j] = values_ratings0[i][j]                 
            if values_train_ratings[i][j] == 0.0 and math.isnan(values_ratings0[i][j]):  
                val = 0
                simSum = 0
                for (sim_uid, sim_value) in dict_user_most_sim[i]:
                    if (values_train_ratings[sim_uid][j] != 0):
                        baseline_xj = mu + baseline_users_ratingMean_train[sim_uid] - mu + baseline_movies_ratingMean_train[j]-mu
                        val += (values_train_ratings[sim_uid][j]+ baseline_movies_ratingMean_train[j] - baseline_xj) * sim_value
                        simSum = simSum + sim_value        
                baseline = mu + baseline_movies_ratingMean_train[j]-mu+ baseline_users_ratingMean_train[i]-mu
                if simSum != 0:
                    values_user_recommend[i,j] = val/simSum + baseline # me add
                else:
                    values_user_recommend[i,j] = baseline    
    return values_user_recommend


def getYtrueYpred(list_test_ratings,map_users_train_uid_index,map_movies_train_mid_index,values_user_recommend,mu):
    true_pred_pair = []   
    y_true = []
    y_pred = []         
    for i in range(len(list_test_ratings)):
        uid_test = int(list_test_ratings[i][0])
        mid_test = int(list_test_ratings[i][1])
        rating_test = list_test_ratings[i][2]
        
        uid_train = map_users_train_uid_index[uid_test]
        if mid_test in map_movies_train_mid_index:
            mid_train = map_movies_train_mid_index[mid_test]
            rating_train = values_user_recommend[uid_train][mid_train]
        else:
            rating_train = mu 
        true_pred_pair.append([float(rating_test),float(rating_train),uid_test,mid_test])
        y_true.append(float(rating_test))
        y_pred.append(float(rating_train))
    return  y_true, y_pred, true_pred_pair

def getTestValueInTest_Df(map_movie_test_mid_index, map_users_test_uid_index,values_user_recommend,df_test_ratings_pivot1_user):
    values_user_recommend_testDF = copy.deepcopy(df_test_ratings_pivot1_user)
    for uid, row in values_user_recommend_testDF.iterrows():
        for mid in values_user_recommend_testDF.columns: 
            if not math.isnan(row[mid]): 
                uid_index = map_users_test_uid_index[uid]
                mid_index = map_movie_test_mid_index[mid]
                rating_pred = values_user_recommend[uid_index][mid_index]
                row[mid] = rating_pred
            
    return values_user_recommend_testDF    
     
    