
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
"""

 Ensemble GCN

"""
NUM_CROSS_VAL = 1
trials = 1
SEED = 43
get_uncertainty = average_divergence
M = 2  # Number of sampled weights (Ensemble)
initial_num_labels = 5
dataset = 'cora'
adj, initial_features, _, _, _, _, _, _, labels = load_data(dataset)
ground_truth = np.argmax(labels, axis=1)
A = adj.todense()
full_A_tilde = preprocess_adj(adj, True)
features_sparse = preprocess_features(initial_features)
feature_matrix = features_sparse.todense()
n = feature_matrix.shape[0]
number_labels = labels.shape[1]
test_split = StratifiedShuffleSplit(
    n_splits=NUM_CROSS_VAL, test_size=0.37, random_state=SEED)
test_split.get_n_splits(labels, labels)
seed_list = [1]
sigma = 0.1

for train_index, test_index in test_split.split(labels, labels):
    print(test_index.shape)
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_split(n, train_index, test_index, labels,
                                                                        initial_num_labels)

    for trial in range(trials):
        seed = seed_list[trial]
        w_0, w_1, A_tilde, gcn_soft = get_trained_gcn(seed, dataset, y_train, y_val, y_test, train_mask, val_mask,
                                                      test_mask)
        vec_weights_0 = w_0.ravel()
        vec_weights_1 = w_1.ravel()
        Gamma = 1/(np.sqrt(2)*sigma)
        initial_gcn = gcn_soft(sparse_to_tuple(features_sparse))
        full_pred_gcn = np.argmax(initial_gcn, axis=1)

        y_pred = np.zeros((M, len(test_index), number_labels))
        print("ACC initial pred : " +
              str(accuracy_score(ground_truth[test_index], full_pred_gcn[test_index])))
        for j in range(M):
            theta_j0_0 = vec_weights_0 + sigma * \
                np.random.normal(0, 1, len(vec_weights_0))
            theta_j0_1 = vec_weights_1 + sigma * \
                np.random.normal(0, 1, len(vec_weights_1))
            theta_j0 = [theta_j0_0, theta_j0_1]

            w_0, w_1, A_tilde, gcn_soft = get_trained_gcn(seed, dataset, y_train, y_val, y_test, train_mask, val_mask,
                                                          test_mask, ensemble=True, theta=theta_j0, Gamma=Gamma)
            # Get prediction by the GCN
            softmax_output = gcn_soft(sparse_to_tuple(features_sparse))
            full_pred_gcn = np.argmax(softmax_output, axis=1)

            print("ACC ensemble pred : " +
                  str(accuracy_score(ground_truth[test_index], full_pred_gcn[test_index])))
            # Compute \hat{y}
            y_pred[j] = softmax_output[test_index]

        average_softmax = np.average(y_pred, axis=0)

        full_pred_gcn = np.argmax(average_softmax, axis=1)
        print("ACC average ensemble pred : " +
              str(accuracy_score(ground_truth[test_index], full_pred_gcn)))
        uncertainty = []
        for i in range(len(test_index)):

            uncertainty.append(get_uncertainty(y_pred[:, i, :]))

        ensemble_good = np.where(full_pred_gcn == ground_truth[test_index])[0]
        ensemble_not = np.where(full_pred_gcn != ground_truth[test_index])[0]
        # classifier_avg_wei_softmax
        plt.plot(ensemble_good, [0 for _ in ensemble_good], 'o', markersize=1)
        plt.plot(ensemble_not, [0.1 for _ in ensemble_not], 'o', markersize=1)
        plt.plot(uncertainty)
        plt.axis([0, 200, -0.5, 0.2])
        plt.show()
