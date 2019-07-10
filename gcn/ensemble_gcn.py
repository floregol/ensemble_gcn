import random
import time
from utils import *
import os
from scipy import sparse
from train import get_trained_gcn
from copy import copy, deepcopy
import pickle as pk
import multiprocessing as mp
import math
import sys
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from helper import *
from uncertainty_helper import *
from metrics import *
from data_partition import data_partition_random, data_partition_fixed
"""

 Ensemble GCN

"""
dataset = 'cora'
initial_num_labels = 20
# NUM_CROSS_VAL = 1
trials = 1
SEED = 4
np.random.seed(SEED)
get_uncertainty = average_divergence
M = 10  # Number of sampled weights (Ensemble)
# adj, initial_features, _, _, _, _, _, _, labels = load_data(dataset)
# ground_truth = np.argmax(labels, axis=1)
# A = adj.todense()
# full_A_tilde = preprocess_adj(adj, True)
# features_sparse = preprocess_features(initial_features)
# feature_matrix = features_sparse.todense()
# n = feature_matrix.shape[0]
# print(n)
# number_labels = labels.shape[1]
# test_split = StratifiedShuffleSplit(
#     n_splits=NUM_CROSS_VAL, test_size=0.37, random_state=SEED)
# test_split.get_n_splits(labels, labels)
seed_list = np.random.randint(1, 1e6, trials)

rho_list = np.array([5e-10, 5e-8, 5e-12])
validation_list = np.zeros(len(rho_list))

for idx in range(len(rho_list)):

    rho = rho_list[idx]

    # for train_index, test_index in test_split.split(labels, labels):
    #     print(test_index.shape)
    #     y_train, y_val, y_test, train_mask, val_mask, test_mask = get_split(n, train_index, test_index, labels,
    #                                                                         initial_num_labels)

    validation = 0
    for trial in range(trials):
        seed = seed_list[trial]

        np.random.seed(seed)

        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels, order = data_partition_random(
            dataset_name=dataset, label_n_per_class=initial_num_labels)

        N = len(labels)
        K = len(labels[0])
        ground_truth = np.argmax(labels, axis=1)

        test_index = np.where(test_mask == True)[0]

        print('trial: ' + str(trial))
        print('GCN result')

        w_0, w_1, A_tilde, gcn_soft, close_session = get_trained_gcn(
            seed,
            adj,
            features,
            y_train,
            y_val,
            y_test,
            train_mask,
            val_mask,
            test_mask,
            ensemble=False,
            theta=None,
            Gamma=None)
        vec_weights_0 = w_0.ravel()
        vec_weights_1 = w_1.ravel()

        # full_pred_gcn = np.argmax(gcn_soft, axis=1)
        # print("ACC initial pred : " +
        #       str(accuracy_score(ground_truth[test_index],full_pred_gcn[test_index])))

        sigma = np.sqrt(2 / K / initial_num_labels / rho)
        Gamma = 0.5 / (sigma**2) / K / initial_num_labels
        y_pred = np.zeros((N, K, M))

        print('Ensemble results')

        theta_j0_0_all = sigma * np.random.normal(0, 1, (M, len(vec_weights_0)))
        theta_j0_1_all = sigma * np.random.normal(0, 1, (M, len(vec_weights_1)))
        close_session()
        for j in range(M):

            theta_j0_0 = theta_j0_0_all[j]
            theta_j0_1 = theta_j0_1_all[j]
            theta_j0 = [theta_j0_0, theta_j0_1]

            w_0, w_1, A_tilde, gcn_soft, close_session = get_trained_gcn(
                seed,
                adj,
                features,
                y_train,
                y_val,
                y_test,
                train_mask,
                val_mask,
                test_mask,
                ensemble=True,
                theta=theta_j0,
                Gamma=Gamma)

            # Get prediction by the GCN
            # full_pred_gcn = np.argmax(gcn_soft, axis=1)

            # print("ACC ensemble pred : " +
            #       str(accuracy_score(ground_truth[test_index], full_pred_gcn[test_index])))
            # Compute \hat{y}
            y_pred[:, :, j] = gcn_soft
            close_session()
       
        average_softmax = np.average(y_pred, axis=2)
        
        print(average_softmax.shape)
        full_pred_gcn = np.argmax(average_softmax, axis=1)
        print("ACC average ensemble pred : " + str(accuracy_score(ground_truth[test_index], full_pred_gcn[test_index])))
        validation += accuracy_score(ground_truth[test_index], full_pred_gcn[test_index])

        # uncertainty = []
        # for i in range(len(test_index)):
        #
        #     uncertainty.append(get_uncertainty(y_pred[i, :, :]))
        # uncertainty = np.array(uncertainty)
        # ensemble_good = np.where(full_pred_gcn == ground_truth[test_index])[0]
        # ensemble_not = np.where(full_pred_gcn != ground_truth[test_index])[0]
        # classifier_avg_wei_softmax
        # plt.plot(ensemble_good, [0 for _ in ensemble_good], 'o', markersize=1)
        # plt.plot(ensemble_not, [0.05 for _ in ensemble_not], 'o', markersize=1)
        # plt.plot(uncertainty[ensemble_good])
        # plt.plot(uncertainty[ensemble_not])
        #plt.axis([0, 300, -0.1, 0.1])
        # plt.show()

    validation_list[idx] = validation

best_idx = np.argmax(validation_list)
print(validation_list)

print('best rho: ' + str(rho_list[best_idx]))
