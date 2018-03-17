import numpy as np
from six.moves import xrange
from numpy.linalg import inv
from dataset import build_user_item_matrix

def _update_user_feature(n_user, ratings_csr_, n_feature, lamda_u, mean_rating_, user_features_, item_features_):
    '''
    n_u : number of rating items of user i
    item_features: n_u * n_feature (108 * 8)  
    A_i = v_j' * u_i + lamda_u * I(n_feature)
    V_i = sum(M_ij * v_j)
    '''
    for i in xrange(n_user):
        _, item_idx = ratings_csr_[i, :].nonzero()
        n_u = item_idx.shape[0]
        if n_u == 0:
            continue
        item_features = item_features_.take(item_idx, axis=0) 

        ratings = ratings_csr_[i, :].data - mean_rating_ 
        A_i = (np.dot(item_features.T, item_features) +
                   lamda_u * n_u * np.eye(n_feature))
        V_i = np.dot(item_features.T, ratings)
        user_features_[i, :] = np.dot(inv(A_i), V_i)


def _update_item_feature(n_item, ratings_csc_, n_feature, lamda_v, mean_rating_, user_features_, item_features_):
    '''
    n_i : number of rating items of item j
    '''
    for j in xrange(n_item):
        user_idx, _ = ratings_csc_[:, j].nonzero()
        n_i = user_idx.shape[0]
        if n_i == 0:
            continue
        user_features = user_features_.take(user_idx, axis=0)
        ratings = ratings_csc_[:, j].data - mean_rating_
    
        A_j = (np.dot(user_features.T, user_features)  + lamda_v * n_i * np.eye(n_feature))
        V_j = np.dot(user_features.T, ratings)
        item_features_[j, :] = np.dot(inv(A_j), V_j)

def ALS_origin(n_user, n_item, n_feature, ratings, mean_rating_, lamda_u, lamda_v, user_features_, item_features_):
    ratings_csr_ = build_user_item_matrix(n_user, n_item, ratings)
    ratings_csc_ = ratings_csr_.tocsc()
    _update_user_feature(n_user, ratings_csr_, n_feature, lamda_u, mean_rating_, user_features_, item_features_)
    _update_item_feature(n_item, ratings_csc_, n_feature, lamda_v, mean_rating_, user_features_, item_features_)
