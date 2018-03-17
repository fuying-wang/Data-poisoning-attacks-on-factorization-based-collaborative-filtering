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

def _update_mal_feature(mal_user, mal_ratings_csr_, n_feature, lamda_u, mal_mean_rating_, mal_user_features_, item_features_):
    for m in xrange(mal_user):
        _, item_idx = mal_ratings_csr_[m, :].nonzero()
        n_m = item_idx.shape[0]
        if n_m == 0:
            continue
        item_features = item_features_.take(item_idx, axis=0) 
        
        ratings = mal_ratings_csr_[m, :].data - mal_mean_rating_ 
        A_i = (np.dot(item_features.T, item_features) +
                   lamda_u * n_m * np.eye(n_feature))
        V_i = np.dot(item_features.T, ratings)
        mal_user_features_[m, :] = np.dot(inv(A_i), V_i)

def _update_item_feature(n_item, ratings_csc_, mal_ratings_csc_, n_feature, lamda_v, mean_rating_, \
    mal_mean_rating_, user_features_, mal_user_features_, item_features_):
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
        
        mal_user_idx, _ = mal_ratings_csc_[:, j].nonzero()
        m_i = mal_user_idx.shape[0]
        if m_i == 0:
            continue
        mal_user_features = mal_user_features_.take(mal_user_idx, axis=0)
        mal_ratings = mal_ratings_csc_[:, j].data - mal_mean_rating_

        A_j = (np.dot(user_features.T, user_features) + np.dot(mal_user_features.T, mal_user_features) \
                + lamda_v * (n_i + m_i) * np.eye(n_feature))
        V_j = np.dot(user_features.T, ratings) + np.dot(mal_user_features.T, mal_ratings)
        item_features_[j, :] = np.dot(inv(A_j), V_j)

def ALS(n_user, n_item, n_feature, mal_user, ratings, mean_rating_, mal_mean_rating_, mal_ratings, lamda_u, lamda_v, \
    user_features_, mal_user_features_, item_features_):
    ratings_csr_ = build_user_item_matrix(n_user, n_item, ratings)
    ratings_csc_ = ratings_csr_.tocsc()
    mal_ratings_csr_ = build_user_item_matrix(mal_user, n_item, mal_ratings)
    mal_ratings_csc_ = mal_ratings_csr_.tocsc()

    _update_user_feature(n_user, ratings_csr_, n_feature, lamda_u, mean_rating_, user_features_, item_features_)
    _update_mal_feature(mal_user, mal_ratings_csr_, n_feature, lamda_u, mal_mean_rating_, mal_user_features_, item_features_)
    _update_item_feature(n_item, ratings_csc_, mal_ratings_csc_, n_feature, lamda_v, mean_rating_, \
    mal_mean_rating_, user_features_, mal_user_features_, item_features_)
