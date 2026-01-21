"""
Utility functions for mRMR-MS (minimum Redundancy Maximum Relevance - Multi-class SVM).

This module provides utility functions for the mRMR-MS feature selection algorithm
with standardized naming conventions and improved performance.


"""

import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


class OptimizedOneVsRestClassifier(OneVsRestClassifier):
    """
    Optimized OneVsRestClassifier to extract support vectors and coefficients.
    
    This class extends OneVsRestClassifier to provide access to dual coefficients
    and support vectors from the underlying SVM models.
    """
    
    def __init__(self, estimator):
        """
        Initialize the classifier.
        
        Parameters
        ----------
        estimator : sklearn estimator
            Base estimator to use for binary classification problems
        """
        super().__init__(estimator)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OptimizedOneVsRestClassifier':
        """
        Fit the one-vs-rest classifier.
        
        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training labels
            
        Returns
        -------
        OptimizedOneVsRestClassifier
            Fitted classifier
        """
        self.classes_ = np.unique(y)    
        
        # Initialize storage for SVM-specific attributes
        self.dual_coefs_ = []
        self.support_vectors_ = []
        self.estimators_ = []
        
        # Train one classifier per class
        for i, class_label in enumerate(self.classes_):
            # Create binary labels (current class vs all others)
            y_binary = np.where(y == class_label, 1, -1)
            
            # Fit binary classifier
            estimator = self.estimator.fit(X, y_binary)
            
            # Store SVM-specific attributes if available
            if hasattr(estimator, 'dual_coef_'):
                self.dual_coefs_.append(estimator.dual_coef_)
            if hasattr(estimator, 'support_vectors_'):
                self.support_vectors_.append(estimator.support_vectors_)
            
            self.estimators_.append(estimator)
        
        return self


def compute_beta_measures(X: np.ndarray, y: np.ndarray, classes_: List, 
                        split_size: float = 0.5, lambda_param: float = 0.5, 
                        kernel: str = 'linear') -> List[float]:
    """
    Main function to compute beta measures for mRMR-MS feature selection.
    
    Parameters
    ----------
    X : np.ndarray
        Input features
    y : np.ndarray
        Target labels
    classes_ : List
        List of class labels
    split_size : float, default=0.5
        Test split size
    lambda_param : float, default=0.5
        Lambda parameter for multiclass weighting
    kernel : str, default='linear'
        Kernel type for SVM ('linear', 'rbf', 'poly')
        
    Returns
    -------
    List[float]
        Beta measures for each feature
    """
    # Split data for training and calibration
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=split_size, stratify=y)
    
    # Initialize parameters
    n_features = X_train.shape[1]
    lambda_p_param = (1 - lambda_param) / (len(classes_) - 1)

    if len(classes_) == 2:
        # Binary classification case
        
        # Transform labels to {-1, 1} format
        y_train_binary = np.where(y_train == 0, -1, 1)
        y_cal_binary = np.where(y_cal == 0, -1, 1)
        
        # Create and train SVM model
        model = SVC(
            kernel=kernel, 
            degree=3, 
            tol=1e-3, 
            probability=True, 
            cache_size=900, 
            C=1.0, 
            max_iter=900000
        )
        model.fit(X_train, y_train_binary)

        # Get support vectors and dual coefficients
        support_vectors = model.support_vectors_
        dual_coef = model.dual_coef_

        # Approximate weights in original space
        weights = np.dot(dual_coef, support_vectors)
            
        # Compute beta values for each feature
        beta_values = []
        for j in range(n_features):
            beta_j = 0.0
            for i in range(len(X_cal)):
                # Beta contribution from each calibration sample
                beta_contribution = weights[0][j] * y_cal_binary[i] * X_cal[i][j]
                beta_j -= beta_contribution

            beta_values.append(beta_j)

    else:
        # Multiclass classification case
        
        # Create and train multiclass SVM
        model = OptimizedOneVsRestClassifier(
            SVC(
                kernel=kernel, 
                degree=3, 
                tol=1e-3, 
                probability=True, 
                cache_size=900, 
                C=1.0, 
                max_iter=900000
            )
        )
        model.fit(X_train, y_train)

        # Extract weights from all binary classifiers
        support_vectors = model.support_vectors_ 
        dual_coefs = model.dual_coefs_
        
        # Approximate feature weights for each class
        weights = [np.dot(dual_coef, support_vector).flatten() 
                  for dual_coef, support_vector in zip(dual_coefs, support_vectors)]
        weights = np.array(weights)
        
        # Convert arrays to numpy for efficient computation
        X_cal = np.array(X_cal)
        y_cal = np.array(y_cal)
        
        # Compute beta values for multiclass case
        beta_values = []
        for j in range(weights.shape[1]):
            beta_j = 0.0
            for i in range(len(X_cal)):
                # First term: lambda * w[y_i, j] * x[i, j]
                lambda_term = lambda_param * weights[y_cal[i], j] * X_cal[i, j]
                
                # Second term: lambda_p * sum of other class weights * x[i, j]
                other_weights = np.delete(weights, y_cal[i], axis=0)[:, j]
                sum_term = lambda_p_param * np.sum(other_weights) * X_cal[i, j]
                    
                beta_contribution = lambda_term - sum_term
                beta_j -= beta_contribution

            beta_values.append(beta_j)
    
    return beta_values


# Backwards compatibility alias
beta_measures = compute_beta_measures