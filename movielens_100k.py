import random
import time

from six.moves import xrange
import numpy as np
from numpy.random import RandomState
from numpy.linalg import inv

from dataset import load_movielens_ratings
from dataset import build_user_item_matrix
from ALS_optimize import ALS
from ALS_optimize_origin import ALS_origin
from evaluation import predict
from evaluation import RMSE
from compute_grad import compute_utility_grad
from compute_grad import compute_grad
    
#ratings_file = '/media/wangfuyi15/F/Adversarial machine learning/ml-20m/ratings.csv'
ratings_file = 'F:/Adversarial machine learning/movielens-master/code/ratings-ml.csv'
ratings = load_movielens_ratings(ratings_file)
rand_state = RandomState(0)

max_rating = max(ratings[:, 2])
min_rating = min(ratings[:, 2])
'''
parameters:
lamda_u: the regularization parameter of user
lamda_v: the regularization parameter of item
alpha: the proportion of malicious users
B: the items of malicious users rating
n_iter: number of iteration
converge: the least RMSE between two iterations
train_pct: the proportion of train dataset
'''
lamda_u = 5e-2
lamda_v = 5e-2
alpha = 0.2
B = 25
n_iters = 10
n_feature = 8
seed = None
last_rmse = None
converge = 1e-5
mal_item = B
# split data to training & testing
train_pct = 0.9
rand_state.shuffle(ratings)
train_size = int(train_pct * ratings.shape[0])
train = ratings[:train_size]
validation = ratings[train_size:]

n_user = max(train[:, 0]) + 1
n_item = max(train[:, 1]) + 1 
mal_user = int(alpha * n_user) 

# add malicious users data
mal_ratings = []
for u in xrange(mal_user):
    mal_user_idx = u
    mal_item_idx = random.sample(range(n_item), mal_item)
    for i in xrange(mal_item):
        mal_movie_idx = mal_item_idx[i]
        mal_rating = 2 * (RandomState(seed).rand() > 0.5) - 1
        mal_ratings.append([mal_user_idx, mal_movie_idx, mal_rating])
mal_ratings = np.array(mal_ratings)
#initialize the matrix U U~ and V 
user_features_ = 0.1 * RandomState(seed).rand(n_user, n_feature)
mal_user_features_ = 0.1 * RandomState(seed).rand(mal_user, n_feature)
item_features_ = 0.1 * RandomState(seed).rand(n_item, n_feature)
mean_rating_ = np.mean(train.take(2, axis=1))
mal_mean_rating_ = np.mean(mal_ratings.take(2, axis=1))
user_features_origin_ = 0.1 * RandomState(seed).rand(n_user, n_feature)
item_features_origin_ = 0.1 * RandomState(seed).rand(n_item, n_feature)

#train origin model
for iteration in xrange(n_iters):
    t1 = time.time()
    ALS_origin(n_user, n_item, n_feature, train, mean_rating_, lamda_u, lamda_v, user_features_origin_, item_features_origin_)
    train_preds = predict(train.take([0, 1], axis=1), user_features_origin_, item_features_origin_, mean_rating_)
    train_rmse = RMSE(train_preds, train.take(2, axis=1))
    t2 = time.time()
    print("The %d th iteration \t time: %ds \t RMSE: %f " % (iteration + 1, t2 - t1, train_rmse))
    # stop when converge
    if last_rmse and abs(train_rmse - last_rmse) < converge:
        break
    else:
        last_rmse = train_rmse

#train added attack data model
for iteration in xrange(n_iters):
    t1 = time.time()
    ALS(n_user, n_item, n_feature, mal_user, train, mean_rating_, mal_mean_rating_, mal_ratings, lamda_u, lamda_v, \
    user_features_, mal_user_features_, item_features_)
    train_preds = predict(train.take([0, 1], axis=1), user_features_, item_features_, mean_rating_)
    train_rmse = RMSE(train_preds, train.take(2, axis=1))
    t2 = time.time()
    print("The %d th iteration \t time: %ds \t RMSE: %f " % (iteration + 1, t2 - t1, train_rmse))
    # stop when converge
    if last_rmse and abs(train_rmse - last_rmse) < converge:
        break
    else:
        last_rmse = train_rmse

#using the algorithm of PGA to optimize the utility function
'''
m_iters: number of iteration in PGA
s_t: step size 
Lamda: 
'''
m_iters = 10
s_t = 0.2 * np.ones([m_iters])
converge = 1e-5
Lamda = 1
last_rmse = None
for t in xrange(m_iters):
    t1 = time.time()
    grad_total = compute_grad(n_user, n_item, mal_user, mal_ratings, train, user_features_, mal_user_features_, \
                        item_features_, lamda_v, n_feature, user_features_origin_, item_features_origin_)
    temp = mal_user_features_
    mal_user_features_ +=  grad_total * s_t[t]
    mal_user_features_[mal_user_features_ > Lamda] = Lamda
    mal_user_features_[mal_user_features_ < - Lamda] = - Lamda
    rmse = rmse(mal_user_features_, temp)
    t2 = time.time()
    print("The %d th iteration \t time: %ds \t RMSE: %f " % (t + 1, t2 - t1, rmse))
    if last_rmse and abs(rmse - last_rmse) < converge:
        break
    else:
        last_rmse = rmse
    

