import random
import numpy as np
from six.moves import xrange
from dataset import build_user_item_matrix
from numpy.linalg import inv
#compute the gradient of the hyrid utility function
def compute_utility_grad(n_user, n_item, train, user_features_, item_features_,user_features_origin_, item_features_origin_, \
    w_j0 = 0.8, u1 = 0.5, u2 = 0.5):
    ratings_csr_ = build_user_item_matrix(n_user, n_item, train)
    grad_av = 2 * (np.dot(user_features_, item_features_.T) - np.dot(user_features_origin_, item_features_origin_.T))
    for i in xrange(n_user):
        _, item_idx = ratings_csr_[i, :].nonzero()
        grad_av[i, item_idx] = 0
    avg_rating = np.mean(np.dot(user_features_, item_features_.T), axis = 0)
    perfer_index = np.where(avg_rating > 0.03)
    J0 = random.sample(list(perfer_index[0]), 1)
    grad_in = np.zeros([n_user, n_item])
    grad_in[:, J0] = w_j0 
    grad_hy = u1 * grad_av + u2 * grad_in
    return grad_hy

def compute_grad(n_user, n_item, mal_user, mal_ratings, train, user_features_, mal_user_features_, \
    item_features_, lamda_v, n_feature, user_features_origin_, item_features_origin_):
    '''
    A : inv(lamda_v * Ik + sum(u_i* u_i))   (for u_i of item j)  k * k
    u_i : 1 * k
    grad_model: d(u_i * v_j.T)/d(M_ij) = u_i * A * u_i.T
    '''
    grad_R = compute_utility_grad(n_user, n_item, train, user_features_, \
            item_features_, user_features_origin_, item_features_origin_)
    ratings_csr_ = build_user_item_matrix(n_user, n_item, train)
    ratings_csc_ = ratings_csr_.tocsc()
    mal_ratings_csr_ = build_user_item_matrix(mal_user, n_item, mal_ratings)
    mal_ratings_csc_ = mal_ratings_csr_.tocsc()
    grad_total = np.zeros([mal_user, n_item])
    for i in xrange(mal_user):
        for j in xrange(n_item):
            if j % 100 == 0:
                print('Computing the %dth malicious user, the %d item(total users: %d, total items: %d)' % (i, j, n_user, n_item))
            user_idx, _ = ratings_csc_[:, j].nonzero()
            mal_user_idx, _ = mal_ratings_csc_[:, j].nonzero()
            user_features = user_features_.take(user_idx, axis=0)
            mal_user_features = mal_user_features_.take(mal_user_idx, axis=0)
            U = np.vstack((user_features, mal_user_features))  
            u_i = user_features_.take(i, axis = 0)
            A = np.dot(U.T, U) + lamda_v * np.eye(n_feature)  
            A_u = np.dot(A, u_i.T)
            grad_model = np.zeros([n_user, n_item])
            for m in xrange(n_user):
                u_m = user_features_.take(i, axis = 0)
                grad_model[m, j] = np.dot(u_m, np.dot(inv(A), u_i.T))
            grad_total[i, j] = sum(sum(grad_model * grad_R))
    return grad_total
            
    
