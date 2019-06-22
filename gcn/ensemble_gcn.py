import random
import time
import tensorflow as tf
from utils import *
from models import GCN, MLP
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
"""

 Moving the nodes around experiment

"""
NUM_CROSS_VAL = 4
trials = 2
CORES = 8
# Train the GCN
SEED = 43
initial_num_labels = 10
THRESHOLD = 0.5
dataset = 'cora'
adj, initial_features, _, _, _, _, _, _, labels = load_data(dataset)
ground_truth = np.argmax(labels, axis=1)
A = adj.todense()
full_A_tilde = preprocess_adj(adj, True)
features_sparse = preprocess_features(initial_features)
feature_matrix = features_sparse.todense()
n = feature_matrix.shape[0]
number_labels = labels.shape[1]

list_new_posititons = random.sample(list(range(n)), 500)
#list_new_posititons = range(n)

test_split = StratifiedShuffleSplit(n_splits=NUM_CROSS_VAL, test_size=0.37, random_state=SEED)
test_split.get_n_splits(labels, labels)
seed_list = [1, 2, 3, 4]

for train_index, test_index in test_split.split(labels, labels):
    print(test_index.shape)
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_split(n, train_index, test_index, labels,
                                                                        initial_num_labels)

    for trial in range(trials):
        seed = seed_list[trial]
        w_0, w_1, A_tilde, gcn_soft = get_trained_gcn(seed, dataset, y_train, y_val, y_test, train_mask, val_mask,
                                                      test_mask)

        # Get prediction by the GCN
        initial_gcn = gcn_soft(sparse_to_tuple(features_sparse))
        print(w_0)

     
