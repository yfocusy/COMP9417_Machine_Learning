#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 10:58:08 2018

@author: jared
"""


import pandas as pd
import numpy as np
import math
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
import copy


def pivotTrainSet(df_train_ratings):
    df_train_ratings_pivot1 = pd.pivot_table(df_train_ratings[['userId','movieId','rating']],columns=['movieId'],index=['userId'],values='rating',fill_value=float("nan"))
    df_train_ratings_pivot = df_train_ratings_pivot1.fillna(0)

    df_train_ratings_pivot1 = df_train_ratings_pivot1.transpose()
    df_train_ratings_pivot = df_train_ratings_pivot.transpose()
    return df_train_ratings_pivot, df_train_ratings_pivot1


def getIndexMap(df_train_ratings_pivot):
    map_users = dict(enumerate(list(df_train_ratings_pivot.columns)))
    map_movies = dict(enumerate(list(df_train_ratings_pivot.index)))

    rev_map_movies = {}
    for t in map_movies.items():
        k, v = t
        rev_map_movies[v] = k

    rev_map_users = {}
    for t in map_users.items():
        k, v = t
        rev_map_users[v] = k
     # me add   
    rev_map_movies = dict(map(lambda t:(t[1],t[0]),map_movies.items()))   
    rev_map_users = dict(map(lambda t:(t[1],t[0]),map_users.items()))    
        
    return map_users, map_movies, rev_map_movies, rev_map_users 

#def getMostSimUserDict(matrix_movie_sim, df_train_ratings_pivot,k,values_train_ratings):
def getMostSimMovieDict(matrix_movie_sim, df_train_ratings_pivot,k,values_train_ratings):
    ###################### 3. movie相似度计算 = 余弦 ################################
    dict_movie_most_sim = dict()
    for i in range(len(values_train_ratings)):
        dict_movie_most_sim[i] = sorted(enumerate(list(matrix_movie_sim[i])),key = lambda x:x[1], reverse=True)[1:k+1]
    return dict_movie_most_sim
    

#def getValuesMovieRecormmend(df_train_ratings_pivot1,df_test_ratings,rev_map_users,rev_map_movies,dict_movie_most_sim):
def getValuesMovieRecormmend(values_train_ratings,values_train_ratings0,dict_movie_most_sim,baseline_users_ratingMean_train,baseline_movies_ratingMean_train,mu,rev_map_users,rev_map_movies):
    values_movie_recommend = np.zeros((len(values_train_ratings),len(values_train_ratings[0])),dtype=np.float32)
    for i in range(len(values_train_ratings)):
        for j in range(len(values_train_ratings[i])):
            if not math.isnan(values_train_ratings0[i][j]):
                values_movie_recommend[i,j] = values_train_ratings0[i][j] 
            if values_train_ratings[i][j] == 0.0 and math.isnan(values_train_ratings0[i][j]):  
                val = 0
                simSum = 0     
                for (sim_mid, sim_value) in dict_movie_most_sim[i]:
                    #sim_mid = sim_mid + 1
                    #print("rev_map_movies[sim_mid]= ",rev_map_movies[sim_mid])
                    #print("sim_mid = " ,sim_mid)
                    if (values_train_ratings[sim_mid][j] != 0):
                        baseline_xj = mu + baseline_users_ratingMean_train[j] - mu + baseline_movies_ratingMean_train[sim_mid]-mu
                        val += (values_train_ratings[sim_mid][j]  - baseline_xj) * sim_value
                        simSum = simSum + sim_value # me add
                baseline = mu + baseline_movies_ratingMean_train[i]-mu+ baseline_users_ratingMean_train[j]-mu
                if simSum != 0:
                    values_movie_recommend[i,j] = val/simSum + baseline # me add
                else: # 跟它相似的user simSum = 0,被抵消掉了
                    values_movie_recommend[i,j] = baseline                         
                        
    return  values_movie_recommend.transpose(), 0
                 

def getYtrueYpred(df_test_ratings_pivot1,df_test_ratings_esti):

    v1 = np.where(df_test_ratings_pivot1 >= 0)

    y_test = []
    y_pred = []

    for i in range(len(v1[0])):
        
        y_test.append(df_test_ratings_pivot1.iloc[v1[0][i], v1[1][i]])
        y_pred.append(df_test_ratings_esti.iloc[v1[0][i], v1[1][i]])

    return y_test,y_pred


def mappingRating(y_true, y_pred, ratio):
    rating_map = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
    for i in range(len(y_pred)):
        for rate in rating_map:
            if y_pred[i] + ratio >= rate and y_pred[i]<rate:
                y_pred[i] = rate
                break
            elif y_pred[i] + ratio < rate and y_pred[i]>=rate - 0.5: 
                y_pred[i] = rate - 0.5
                break
            elif y_pred[i]<=0:
                y_pred[i] = 0
            elif y_pred[i]>5:
                y_pred[i] = 5
    MSE = metrics.mean_squared_error(y_true,y_pred)
    MAE = metrics.mean_absolute_error(y_true,y_pred)
    return MSE,MAE


def getWholeTrainingMatrixScore(values_train_ratings,values_train_ratings0,df_train_ratings_pivot1,map_users,map_movies,rev_map_users,rev_map_movies,dict_movie_most_sim):
    values_movie_recommend = np.zeros((len(values_train_ratings[0]),len(values_train_ratings)),dtype=np.float32)  

    bx = df_train_ratings_pivot1.mean().values.tolist() # me add
    bi = df_train_ratings_pivot1.transpose().mean().values.tolist()

    s = df_train_ratings_pivot1.sum().sum()
    n = df_train_ratings_pivot1.count().sum()
    mu = s/n

    for i in range(len(values_train_ratings[0])):
        for j in range(len(values_train_ratings)):
            if not math.isnan(values_train_ratings0[j][i]):
                values_movie_recommend[i,j] = values_train_ratings0[j][i]
            else:    
                user = i
                movie = j
                
                real_user = map_users[user]
                real_movie = map_movies[movie]
                
                if real_movie not in rev_map_movies.keys() and real_user in rev_map_users.keys():

                    user_in_train = rev_map_users[real_user]
                    est_rating = bx[user_in_train]

                elif real_movie in rev_map_movies.keys() and real_user not in rev_map_users.keys():

                    movie_in_train = rev_map_movies[real_movie]
                    est_rating = bi[movie_in_train]

                elif real_movie not in rev_map_movies.keys() and real_user not in rev_map_users.keys():

                    est_rating = mu

                else:

                    user_in_train = rev_map_users[real_user]
                    movie_in_train = rev_map_movies[real_movie]
                    baseline = bx[user_in_train] + bi[movie_in_train] - mu
                    
                    sim_movie_list = dict_movie_most_sim[movie_in_train]
                    
                    sum_up = 0
                    sum_down = 0

                    for pairs in sim_movie_list:

                        (sim_movie, sim) = pairs

                        try:
                            x = df_train_ratings_pivot1.iloc[sim_movie, real_user]
                        except:
                            x = -1
                        
                        if x >= 0:
                            
                            try:
                                bxj = bx[user_in_train] + bi[sim_movie] - mu
                            except:
                                bxj = bx[user_in_train]
                            
                            sum_up += sim*(x - bxj)
                            sum_down += sim
                    
                    if sum_down:
                        est_rating = baseline + sum_up/sum_down
                    else:
                        est_rating = baseline

                values_movie_recommend[user, movie] = est_rating
    return values_movie_recommend
