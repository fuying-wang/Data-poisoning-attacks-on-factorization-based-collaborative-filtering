import numpy as np
import scipy.sparse as sparse
#from itertools import islice  

def load_movielens_ratings(ratings_file):
    with open(ratings_file) as f:
        ratings = []
        for line in f:
        #for line in islice(f, 1, None):  
            line = line.split(",")[:3]
            line = [int(l) for l in line]
            ratings.append(line)
        ratings = np.array(ratings)
    return ratings

def build_user_item_matrix(n_user, n_item, ratings):
    """Build user-item matrix
    Return
    ------
        sparse matrix with shape (n_user, n_item)
    """
    data = ratings[:, 2]
    row_ind = ratings[:, 0]
    col_ind = ratings[:, 1]
    shape = (n_user, n_item)
    return sparse.csr_matrix((data, (row_ind, col_ind)), shape=shape)
    
