#----------------------------------------------
# Author: Austin Boone
# Date: July 30th, 2023
#
# DecisionTree.py
#
# Implements a decision tree machine learning algorithm. 
# This algorithm can be used for both classification and regression tasks, 
# as specified by the task parameter during the initialization of the DecisionTree class.
#
# Includes an option to perform regression tasks, a feature not present in the previous version. 
# It calculates the mean value for regression tasks when creating leaf nodes.
# ---------------------------------------------

import numpy as np
import pandas as pd
from collections import Counter

def entropy(y):
    """
    Calculate the entropy of a dataset.
    The only parameter of this function is y: a numpy array of target (label) values
    """
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class Node:
    """
    A decision tree node.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """
        Check if current node is a leaf node.
        """
        return self.value is not None

class DecisionTree:
    """
    Decision Tree class, that fits and predicts labels.
    """
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None, task='classification'):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.task = task
        self.root = None

    def fit(self, X, y):
        """
        Fit the data to the tree.
        Parameters are X: a numpy array of features, 
        and y: a numpy array of target (label) values.
        """
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        """
        Predict the labels of the data.
        The parameter is X: a numpy array of features.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
    """
    Recursively grow the decision tree.

    Parameters:
    X (numpy.ndarray): A matrix containing the feature vectors for the dataset.
    y (numpy.ndarray): A vector containing the target labels for the dataset.
    depth (int): The current depth of the tree. Default is 0.

    Returns:
    Node: The root node of the grown tree.
    """
    
    n_samples, n_features = X.shape  # Get the number of samples and features
    n_labels = len(np.unique(y))     # Get the number of unique labels in the target

    # Check stopping criteria: if any of these conditions are met, create a leaf node
    if (depth >= self.max_depth                     # Maximum depth reached
            or n_labels == 1                        # All samples have the same label
            or n_samples < self.min_samples_split): # Number of samples is below the threshold for splitting
        leaf_value = self._most_common_label(y)     # Get the most common label in the target
        if self.task == 'regression':               # If the task is regression, use the mean value instead
            leaf_value = np.mean(y)
        return Node(value=leaf_value)               # Create and return a leaf node with the determined value

    # Select a random subset of features to consider for splitting
    feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

    # Greedily select the best split according to information gain
    best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

    # Split the dataset into left and right subsets based on the best feature and threshold
    left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

    # Recursively grow the left and right children
    left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
    right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

    # Return a node with the best feature, threshold, and left and right children
    return Node(best_feat, best_thresh, left, right)

    def _traverse_tree(self, x, node):
        """
        Traverse a tree to find the node value.
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        """
        Find the most common label in a numpy array.
        """
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _best_criteria(self, X, y, feat_idxs):
        """
        This method finds the best criteria for splitting a node in a decision tree.
    
        Parameters:
        X (numpy.ndarray): A matrix containing the feature vectors for the dataset.
        y (numpy.ndarray): A vector containing the target labels for the dataset.
        feat_idxs (list): A list of feature indices to consider for the split.
    
        Returns:
        split_idx (int): The index of the feature that provides the best split.
        split_thresh (float): The threshold value for the best split.
        """
        
        # Initialize variables to store the best information gain and corresponding split parameters
        best_gain = -1  # Placeholder for the best information gain. Initialized to a value that will always be improved upon.
        split_idx, split_thresh = None, None  # Placeholder for the index of the feature and threshold value for the best split.
    
        # Iterate through the given feature indices
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]         # Extract the column corresponding to the current feature index
            thresholds = np.unique(X_column)  # Obtain unique values in the feature column to consider as potential thresholds
    
            # Iterate through the unique thresholds for the current feature
            for threshold in thresholds:
                # Calculate the information gain using the given threshold for the current feature
                gain = self._information_gain(y, X_column, threshold)
    
                # If the calculated gain is greater than the current best gain, update the best gain and split parameters
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
    
        # Return the index of the feature and threshold value that provides the best split
        return split_idx, split_thresh


    def _information_gain(self, y, X_column, split_thresh):
        """
        Calculate information gain.
        """
        # Parent loss
        parent_entropy = entropy(y)

        # Generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        """
        Split data in a numpy array based on a column and a split threshold.
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
