# Data-poisoning-attacks-on-factorization-based-collaborative-filtering
# Reference
* "Data poisoning Attack on Factorization-Based Collaborative Filtering" NIPS 2016 
* Github respository : "recommend"
* Github respository : "movielens"
# Illustration
* ALS_optimize.py: compute optimize solution using malicious data and normal data.
* ALS_optimize_old.py: compute optimize solution only using normal data.
* compute_grad.py: compute two parts of gradient using the formulations in the paper.
* dataset.py: functions of load dataset and build user item matrix.
* evaluation.py: predict data and compure RMSE
* main.py: the implement of PGA
