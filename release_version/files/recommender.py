import userCF
import itemCF

import pandas as pd
import numpy as np
import math
import copy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import time


def readDataSet(datasetNum):
    if datasetNum == 1:
        try:
            file_ratingsCSV = './data/ml-latest-small/ratings.csv'
            file_moviesCSV ='./data/ml-latest-small/movies.csv'
            file_tagsCSV ='./data/ml-latest-small/tags.csv'
            df_ratings = pd.read_csv(file_ratingsCSV,index_col=None).drop(columns=['timestamp'])
            df_movies = pd.read_csv(file_moviesCSV,index_col=None).drop(columns=['genres'])
            df_tags = pd.read_csv(file_tagsCSV,index_col=None).drop(columns=['timestamp'])
        except FileNotFoundError:
            print("DataSet NotFound: Please put MovieLens ml-latest-small data set in ./data/ml-latest-small/ folder")
            exit()
    elif datasetNum == 2:
        file_ratingsCSV = './data/ml-20m/ratings.csv'
        file_moviesCSV ='./data/ml-20m/movies.csv'
        file_tagsCSV ='./data/ml-20m/tags.csv'
        df_ratings = pd.read_csv(file_ratingsCSV,index_col=None).drop(columns=['timestamp'])
        df_movies = pd.read_csv(file_moviesCSV,index_col=None).drop(columns=['genres'])
        df_tags = pd.read_csv(file_tagsCSV,index_col=None).drop(columns=['timestamp'])    
    return  df_ratings, df_movies,df_tags

def describeDataSet(df_ratings, df_movies,df_tags):
    describe_ratings = df_ratings.describe()
    describe_movies = df_movies.describe()
    describe_tags = df_tags.describe()
    return  describe_ratings, describe_movies, describe_tags

def describeRatingData(df_data):   
    rating_count_by_movie = df_data.groupby(['movieId','title'],as_index = False) ['rating'].count()
    rating_count_by_movie.columns = ['movieId','title','rating_count']
    rating_count_by_movie.sort_values(by=['rating_count'],ascending= False, inplace=True)
    stddev_rating = df_data.groupby(['movieId','title']).agg({'rating':['mean','std']})
    total_movie_count_participated = len(set(df_ratings['movieId'].values.tolist()))
    total_user_count_participated = len(set(df_ratings['userId'].values.tolist()))
    return rating_count_by_movie,stddev_rating,total_movie_count_participated,total_user_count_participated 

def splitTrainTest(testsize):
    df_train_ratings_raw,df_test_ratings = model_selection.train_test_split(df_ratings, test_size = testsize)
    print()
    print("train_movie_count:" + str(len(set(df_train_ratings_raw['movieId'].values.tolist()))))
    print("train_user_count:" + str(len(set(df_train_ratings_raw['userId'].values.tolist()))))
    print("test_movie_count:" + str(len(set(df_test_ratings['movieId'].values.tolist()))))
    print("test_user_count:" + str(len(set(df_test_ratings['userId'].values.tolist()))))
    return df_train_ratings_raw,df_test_ratings

def removeTestPointFromTrainSet(df_ratings,list_test_ratings):
    df_train_ratings = copy.deepcopy(df_ratings)    
    for i in range(len(list_test_ratings)):
        df_train_ratings.loc[((df_train_ratings['userId']==int(list_test_ratings[i][0])) & (df_train_ratings['movieId']==int(list_test_ratings[i][1]))),['rating']] = float("nan")
    return df_train_ratings
  
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
    MSE = mean_squared_error(y_true,y_pred)
    MAE = mean_absolute_error(y_true,y_pred)
    return MSE,MAE

def afterMappingError(y_true, y_pred, rating_mapping_ratio):
    evaliuators={}
    for ratio in rating_mapping_ratio:
        y_pred2 = y_pred[::1]
        MSE,MAE = mappingRating(y_true, y_pred2,ratio)
        RMSE = math.sqrt(MSE)
        evaliuators[ratio]={"MSE":MSE, "MAE":MAE, "RMSE":RMSE}
    return evaliuators

def getRecommend_top5 (values_final_recommend):
    recommendDict_top5 = {}
    for i in range(len(values_final_recommend)):
        recommendDict_top5[i] = sorted(enumerate(list(values_final_recommend[i])),key= lambda x:x[1],reverse=True)[:5]
    return recommendDict_top5 

def mappingRecommendRating(rating, ratio):
    rating_map = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
    for rate in rating_map:
        if rating + ratio >= rate and rating<rate:
            rating = rate
            break
        elif rating + ratio < rate and rating>=rate - 0.5: 
            rating = rate - 0.5
            break
        elif rating<=0:
            rating= 0
        elif rating>5:
            rating = 5
    return rating


if __name__ == '__main__':

    ###################### 0. globals ################################
    time_start = time.clock()
    datasetNum = 1 # /data/ml-latest-small/ 
    testsize = 0.2


    #user_topK =  [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160]
    #movie_topK = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,120,140,160]
    user_topK =  [15]
    movie_topK = [45]
    current_userK = 15
    current_movieK = 45
    
    
    
    ###################### 1. read MovieLens csv files ################################
    df_ratings, df_movies,df_tags = readDataSet(datasetNum)
    describe_ratings, describe_movies, describe_tags = describeDataSet(df_ratings, df_movies,df_tags)   
    time_readdata_end= time.clock()
    print("\n..........................................................................")
    print("Program Starting. The whole process will take around 230 seconds.")
    print("..........................................................................\n")
    print("\n.................Step1: read csf from MovieLens data file.................")
    print("\nread data time : %f s" % (time_readdata_end - time_start))
    print("running   time : %f s" % (time_readdata_end - time_start))
    
    ###################### 2. merge data set as dataframe + evaluate data ################################
    df_data = pd.merge(df_ratings,df_movies, on="movieId")
    rating_count_by_movie,stddev_rating,total_movie_count_participated,total_user_count_participated = describeRatingData(df_data)

    time_evaluatedata_end= time.clock()
    print("\n.................Step2: merge data set as dataframe & evaluate data.......")
    print("\nevaluate data time : %f s" % (time_evaluatedata_end - time_readdata_end))
    print("running  time : %f s" % (time_evaluatedata_end - time_start))


    #################################
    # Display distribution of rating
    #import seaborn as sns
    #sns.set_style('whitegrid')
    #sns.set(font_scale=1.5)
    #sns.distplot(df_data['rating'].fillna(df_data['rating'].median()))
     
    ##################### 3. split train + test, creat train_pivot ################################
    print("\n.................Step3: train set & test set spliting.....................")
    df_train_ratings_raw,df_test_ratings = splitTrainTest(testsize)
    list_test_ratings = np.array(df_test_ratings).tolist()    
    df_train_ratings = removeTestPointFromTrainSet(df_ratings,list_test_ratings)
    
    # for user
    df_train_ratings_pivot_user, df_train_ratings_pivot1_user = userCF.pivotTrainSet(df_train_ratings)
    map_movies_train_mid_index, map_users_train_uid_index = userCF.getIndexMap(df_train_ratings_pivot_user)

    # for movie 
    df_train_ratings_pivot_movie, df_train_ratings_pivot1_movie = itemCF.pivotTrainSet(df_train_ratings)
    map_users, map_movies, rev_map_movies, rev_map_users = itemCF.getIndexMap(df_train_ratings_pivot_movie)

    time_splitdata_end= time.clock()
    print("\nsplitting train + test set time : %f s" % (time_splitdata_end - time_evaluatedata_end))
    print("running  time : %f s" % (time_splitdata_end - time_start))



    
    ###################### 4 user similarity computing, pearson ################################
    print("\n.................Step4: user similirity computing.........................")
    values_train_ratings_user = df_train_ratings_pivot_user.values.tolist() 
    dict_user_most_sim_diff_k = {}  
    matrix_user_sim = cosine_similarity(df_train_ratings_pivot_user)
    for k in user_topK:
        dict_user_most_sim  = userCF.getMostSimUserDict(matrix_user_sim, df_train_ratings_pivot_user,k,values_train_ratings_user)
        dict_user_most_sim_diff_k[k] = dict_user_most_sim
     
    mu = np.nanmean(np.array(df_train_ratings_pivot1_user)) # mean
    baseline_users_ratingMean_train =  df_train_ratings_pivot1_user.mean(axis=1).values.tolist()
    baseline_movies_ratingMean_train = df_train_ratings_pivot1_user.mean().values.tolist()
    values_train_ratings0_user = df_train_ratings_pivot1_user.values.tolist()
      
    # user
    dict_user_most_sim =  dict_user_most_sim_diff_k[current_userK]
    values_user_recommend = userCF.getValuesUserRecormmend(values_train_ratings_user,values_train_ratings0_user,dict_user_most_sim,baseline_users_ratingMean_train,baseline_movies_ratingMean_train,mu)

    time_userCF_end= time.clock()
    print("\nuserCF time : %f s" % (time_userCF_end - time_splitdata_end))
    print("running  time : %f s" % (time_userCF_end - time_start))


    ###################### 5 item similarity computing, cosine ################################
    print("\n.................Step5: item similirity computing.........................")
    values_train_ratings_movie = df_train_ratings_pivot_movie.values.tolist()  
    dict_movie_most_sim_diff_k = {}  
    matrix_item_sim = cosine_similarity(df_train_ratings_pivot_movie)
    for k in movie_topK:
        dict_movie_most_sim  = itemCF.getMostSimMovieDict(matrix_item_sim, df_train_ratings_pivot_movie,k,values_train_ratings_movie)
        dict_movie_most_sim_diff_k[k] = dict_movie_most_sim
    
    values_train_ratings0_movie = df_train_ratings_pivot1_movie.values.tolist()
    
    # items
    dict_movie_most_sim =  dict_movie_most_sim_diff_k[current_movieK]
    values_movie_recommend,df_test_ratings_pivot1_movie  = itemCF.getValuesMovieRecormmend(values_train_ratings_movie,values_train_ratings0_movie,dict_movie_most_sim,baseline_users_ratingMean_train,baseline_movies_ratingMean_train,mu,rev_map_users,rev_map_movies)
 
    time_itemCF_end= time.clock()
    print("\nitemCF time : %f s" % (time_itemCF_end - time_userCF_end))
    print("running  time : %f s" % (time_itemCF_end - time_start))

      
    ###################### 6. evaluating model ################################
    print("\n.................Step6: evaluating model................................")
    y_userCF_true, y_userCF_pred, userCF_true_pred_pair = userCF.getYtrueYpred(list_test_ratings,map_users_train_uid_index,map_movies_train_mid_index,values_user_recommend,mu)
    y_movieCF_true, y_movieCF_pred, itemCF_true_pred_pair = userCF.getYtrueYpred(list_test_ratings,rev_map_users,rev_map_movies, values_movie_recommend,mu)
    
    values_final_recommend = (values_movie_recommend + values_user_recommend)/2
    y_true, y_pred, y_pair = userCF.getYtrueYpred(list_test_ratings,map_users_train_uid_index,map_movies_train_mid_index,values_final_recommend,mu)

    rating_mapping_ratio = [0,0.3]
    evaliuators_userCF = afterMappingError(y_userCF_true, y_userCF_pred, rating_mapping_ratio)
    print("\ncurrent_userK = ", current_userK)
    print("\n........................................................................")
    print("------------------------------ userCF ----------------------------------")
    print("ratio\t\tMSE\t\t\tRMSE\t\t\tMAE")
    for (k,v) in evaliuators_userCF.items():  
        s = "%s \t\t%.11f\t\t%.11f\t\t%.11f"
        print(s % (k,v["MSE"],v["RMSE"],v["MAE"]))

    evaliuators_movieCF = afterMappingError(y_movieCF_true, y_movieCF_pred, rating_mapping_ratio)
    print("\ncurrent_movieK = ", current_movieK)
    print("\n........................................................................")
    print("--------------------------------- itemCF ---------------------------------")
    print("ratio\t\tMSE\t\t\tRMSE\t\t\tMAE")
    for (k,v) in evaliuators_movieCF.items():  
        s = "%s \t\t%.11f\t\t%.11f\t\t%.11f"
        print(s % (k,v["MSE"],v["RMSE"],v["MAE"]))
        
    evaliuators_movieCF = afterMappingError(y_true, y_pred, rating_mapping_ratio)
    print("\n........................................................................")
    print("----------------------------- Multiple CF ------------------------------\n")
    print("ratio\t\tMSE\t\t\tRMSE\t\t\tMAE")
    for (k,v) in evaliuators_movieCF.items():  
        s = "%s \t\t%.11f\t\t%.11f\t\t%.11f"
        print(s % (k,v["MSE"],v["RMSE"],v["MAE"]))        
        
        
    time_pred_end= time.clock()
    print("\n pred time : %f s" % (time_pred_end - time_itemCF_end))
    print("running  time : %f s" % (time_pred_end - time_start))

        
    ###################### 7. Recommend movies for inpiut uid ################################
    print("\n.................Step7 : evaluating model..............................")
    recommendDict_top5_final = {}
    recommendDict_top5_final_list =[]
    recommendDict_top5 = getRecommend_top5 (values_final_recommend)
    for uid_index in recommendDict_top5.keys():
        tmp = []
        uid = map_users[uid_index]
        for (mid_index, rating) in recommendDict_top5[uid_index]:
            mid = map_movies[mid_index]
            rating = mappingRecommendRating(rating, 0.3)
            tmp.append((mid,rating))
            recommendDict_top5_final_list.append([uid, mid, rating])
        recommendDict_top5_final[uid] = tmp 
           
   
    recommendDF = pd.DataFrame(recommendDict_top5_final_list, columns=["userId","movieId","pred rating"]) 
    recommendDF = pd.merge(recommendDF, df_movies[["movieId","title"]],on="movieId", how="inner") 
    
    uid_current_raw = input("Recommdation System is ready. Please input an integer as uid: ")
    if int(uid_current_raw) not in map_users_train_uid_index.keys():
        print(type(input_uid))
        uid_current = 1
        print("Your input uid is not exist in the system. Recommend uid = 1 for you")        
    else:     
        uid_current = int(uid_current_raw)
    recommendDF_one_uid = recommendDF[recommendDF["userId"] == uid_current] 
    recommend_one_uid = np.array(recommendDF_one_uid)
    print("\n..........................................................................")
    print("\n........... for your input uid, there are 5 movies you may like............")
    print("userId\t\tmovieId\t\tpred rating\t\ttitle")
    for i in range(len(recommend_one_uid)):       
        s = "%s\t\t%s\t\t%s\t\t%s"
        print(s % (recommend_one_uid[i][0],recommend_one_uid[i][1],recommend_one_uid[i][2],recommend_one_uid[i][3]))
    print("\n..........................................................................")
    print("...................................The end...............................\n")      
   
    
    
