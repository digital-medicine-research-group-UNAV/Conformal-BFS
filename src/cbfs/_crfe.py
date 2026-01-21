"""


This module implements an efficient version of Conformal Recursive Feature Elimination
(CRFE) with standardized notation and optimized performance.

"""

import numpy as np
from numbers import Integral
import itertools
import sys
from typing import Optional, Union, Tuple, Dict, List
from numba import jit, njit

from sklearn.svm import SVR, LinearSVC
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, clone
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from ._utils_crfe import binary_change, NC_OvsA_SVMl_dev
from ._conformal_module import CP






# Note: numba decorators commented out due to NumPy 2.0 compatibility issues
# @njit
def _compute_beta_binary_optimized(weights: np.ndarray, y_cal: np.ndarray, X_cal: np.ndarray) -> np.ndarray:
    """
    Optimized beta computation for binary classification.
    
    Parameters
    ----------
    weights : np.ndarray
        Model weights/coefficients
    y_cal : np.ndarray
        Calibration labels
    X_cal : np.ndarray
        Calibration features
        
    Returns
    -------
    np.ndarray
        Beta values for each feature
    """
    result = -np.sum(weights[0] * y_cal[:, np.newaxis] * X_cal, axis=0)
    return result.astype(np.float64)

 
# @njit
def _compute_beta_multiclass_optimized(weights: np.ndarray, y_cal: np.ndarray, X_cal: np.ndarray, 
                                     lambda_param: float, lambda_p_param: float) -> np.ndarray:
    """
    Optimized beta computation for multiclass classification using numba.
    
    Parameters
    ----------
    weights : np.ndarray
        Model weights/coefficients for all classes
    y_cal : np.ndarray
        Calibration labels
    X_cal : np.ndarray
        Calibration features
    lambda_param : float
        Lambda parameter for multiclass weighting
    lambda_p_param : float
        Lambda prime parameter for multiclass weighting
        
    Returns
    -------
    np.ndarray
        Beta values for each feature
    """
    n_features = weights.shape[1]
    n_samples = X_cal.shape[0]
    weights_sum = np.sum(weights, axis=0)
    beta = np.zeros(n_features, dtype=np.float64)
    
    for j in range(n_features):
        beta_j = 0.0
        for i in range(n_samples):
            y_i = y_cal[i]
            x_ij = X_cal[i, j]
            lambda_term = lambda_param * weights[y_i, j] * x_ij
            sum_term = lambda_p_param * (weights_sum[j] - weights[y_i, j]) * x_ij
            beta_j -= (lambda_term - sum_term)
        beta[j] = beta_j
    
    return beta


class CRFE(BaseEstimator):
    """
    Optimized Conformal Recursive Feature Elimination.

    This implementation provides an efficient version of CRFE with standardized
    parameter names and improved performance through vectorized operations.

    Parameters
    ----------
    estimator : sklearn estimator, default=None
        Supervised learning estimator with fit method and coef_ attribute.
        
    n_features_to_select : int, default=1
        The target number of features to select.
                         
    lambda_param : float, default=0.5
        Multi-class parameter between 0 and 1.
        Controls the weight of the "one" class in "OneVsRest" strategy.
        
    epsilon : float, default=0.4
        Conformal prediction confidence level parameter.

    Attributes
    ----------
    selected_features_ : np.ndarray
        Indices of the selected features.
    
    feature_betas_ : List[float]
        Beta values associated with each selected feature.

    estimator_ : sklearn estimator
        The fitted estimator used for feature selection.

    classes_ : np.ndarray
        Class labels available when estimator is a classifier.
        
    lambda_p_param_ : float
        Computed lambda prime parameter for multiclass problems.
        
    results_dict_ : Dict
        Dictionary containing intermediate results from the elimination process.
    """

    def __init__(
            self,
            estimator: Optional[BaseEstimator] = None,
            n_features_to_select: int = 1,
            lambda_param: float = 0.5,
            epsilon: float = 0.4
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.lambda_param = lambda_param
        self.epsilon = epsilon
        
        # Internal attributes - initialized during fit
        self.estimator_ = None
        self.classes_ = None
        self.lambda_p_param_ = None
        self.selected_features_ = None
        self.feature_betas_ = None
        self.results_dict_ = {}
        
       


    

    

    def _predict_scores_svm(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_cal: np.ndarray, y_cal: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray) -> Optional[Tuple]:
        """
        Optimized SVM scoring with efficient array operations.
        
        Parameters
        ----------
        X_train, y_train : Training data and labels
        X_cal, y_cal : Calibration data and labels  
        X_test, y_test : Test data and labels
        
        Returns
        -------
        Optional[Tuple] : Conformal prediction scores (currently disabled)
        """
        # Remove header row for processing
        X_train_data = X_train[1:]
        X_cal_data = X_cal[1:]
        X_test_data = X_test[1:]

        # Initialize lambda_p_param_ if not already set
        if self.lambda_p_param_ is None:
            self.lambda_p_param_ = (1 - self.lambda_param) / (len(self.classes_) - 1) if len(self.classes_) > 2 else 0

        if len(self.classes_) == 2:
            # Binary classification
            estimator = self.estimator.fit(X_train_data, y_train)
            weights = estimator.coef_
            bias = estimator.intercept_

            multiclass = False
            ncm_cal = NC_OvsA_SVMl_dev(X_cal_data, y_cal, weights, bias, multiclass)

            # Vectorized computation for test NCM
            ncm_test = [NC_OvsA_SVMl_dev(
                        np.tile(sample, (len(self.classes_), 1)),
                        self.classes_,
                        weights, bias,
                        self.lambda_param,
                        self.lambda_p_param_,
                        multiclass) for sample in X_test_data]
        else:
            # Multiclass classification
            if isinstance(self.estimator_, OneVsRestClassifier):
                estimator = self.estimator_.fit(X_train_data, y_train)
                weights = estimator.coef_
                bias = estimator.intercept_
            else:
                estimator = OneVsRestClassifier(self.estimator_, n_jobs=-1)
                estimator.fit(X_train_data, y_train)
                
                # Vectorized coefficient extraction
                weights = np.array([est.coef_[0] for est in estimator.estimators_])
                bias = np.array([est.intercept_[0] for est in estimator.estimators_])

            multiclass = True
            ncm_cal = NC_OvsA_SVMl_dev(X_cal_data, y_cal, weights, bias, 
                                     self.lambda_param, self.lambda_p_param_, multiclass)

            ncm_test = [NC_OvsA_SVMl_dev(
                        np.tile(sample, (len(self.classes_), 1)),
                        self.classes_,
                        weights, bias,
                        self.lambda_param,
                        self.lambda_p_param_,
                        multiclass) for sample in X_test_data]

        # Conformal prediction scores (currently disabled)
        scores = CP(0.10).Conformal_prediction_scores(y_test, ncm_test, ncm_cal, self.classes_)
        print("Scores: ", scores[0], scores[1], scores[2])
        sys.stdout.flush()
        
        return None  # scores

    def _recursive_elimination(self, X_train: np.ndarray, y_train: np.ndarray, 
                              X_cal: np.ndarray, y_cal: np.ndarray, 
                              X_test: np.ndarray, y_test: np.ndarray) -> List[np.ndarray]:
        """
        Optimized recursive elimination with reduced memory allocations.
        
        Parameters
        ----------
        X_train, y_train : Training data and labels
        X_cal, y_cal : Calibration data and labels
        X_test, y_test : Test data and labels
        
        Returns
        -------
        List[np.ndarray] : List of feature indices at each elimination step
        """
        n_features = X_train.shape[1]
        feature_indices = np.arange(n_features)
        
        # Initialize lambda_p parameter for multiclass
        self.lambda_p_param_ = (1 - self.lambda_param) / (len(self.classes_) - 1)
        
        # Pre-allocate output dictionary
        results_keys = ["Index", "coverage", "inefficiency", 
                       "certainty", "uncertainty", "mistrust",
                       "S_score", "F_score", "Creditibily"]
        elimination_results = {name: [] for name in results_keys}

        # Add header row efficiently with feature indices
        X_train_work = np.vstack([feature_indices, X_train])
        X_cal_work = np.vstack([feature_indices, X_cal])
        X_test_work = np.vstack([feature_indices, X_test])

        n_current_features = n_features

        # Main elimination loop
        #print(f"Starting recursive feature elimination from {n_current_features} features to {self.n_features_to_select} features.")
        while n_current_features != self.n_features_to_select:
            current_indices = X_train_work[0].astype(int)
            
            # Get data views without header
            X_train_data = X_train_work[1:]
            X_cal_data = X_cal_work[1:]

            # Clone estimator for clean fit
            model = clone(self.estimator)
            
            # Fit classifier and compute beta values
            if len(self.classes_) == 2:
                # Binary classification
                model.fit(X_train_data, y_train)
                weights = model.coef_
                
                # Use optimized beta computation
                beta = _compute_beta_binary_optimized(weights, y_train, X_train_data)
            else:
                # Multiclass classification
                if isinstance(model, OneVsRestClassifier):
                    model.fit(X_train_data, y_train)
                else:
                    model = OneVsRestClassifier(model, n_jobs=-1)
                    model.fit(X_train_data, y_train)

                # Vectorized coefficient extraction
                weights = np.array([est.coef_[0] for est in model.estimators_])
                
                # Use optimized beta computation
                beta = _compute_beta_multiclass_optimized(weights, y_train, X_train_data, 
                                                        self.lambda_param, self.lambda_p_param_)


            feat_for_rem = int(n_current_features - self.n_features_to_select)

            features_to_eliminate = np.argsort(beta)[-feat_for_rem:]
            #features_to_eliminate  = np.sort(features_to_eliminate ) # this does not seem necessary

            # Find feature to eliminate (feature with maximum beta)
            #feature_to_eliminate = np.argmax(beta)
            
            # Efficient column deletion using boolean indexing
            keep_mask = np.ones(n_current_features, dtype=bool)
            keep_mask[features_to_eliminate] = False
            #print(f"Eliminating features: {current_indices[features_to_eliminate].astype(int)}")

            # Update working arrays
            X_train_work = X_train_work[:, keep_mask]
            X_cal_work = X_cal_work[:, keep_mask]
            X_test_work = X_test_work[:, keep_mask]

            # Update feature indices
            X_train_work[0] = current_indices[keep_mask]
            X_cal_work[0] = current_indices[keep_mask]
            X_test_work[0] = current_indices[keep_mask]

            n_current_features = X_train_work.shape[1]
            
            #print(f"Remaining features: {n_current_features}")
            sys.stdout.flush()

            # Store current feature indices
            elimination_results["Index"].append(current_indices[keep_mask].astype(int))

            # Calculate conformal prediction scores (currently disabled)
            #self._predict_scores_svm(X_train_work, y_train, X_cal_work, y_cal, X_test_work, y_test)
            # for i, name in enumerate(results_keys[1:]):
            #     elimination_results[name].append(scores[i])
        
        # Store final results
        self.selected_features_ = X_train_work[0].astype(int)
        
        # Store beta values for selected features (excluding the last eliminated feature)
        if "beta" in locals():
            final_beta = np.delete(beta, features_to_eliminate)
            self.feature_betas_ = final_beta.tolist()
        else:
            self.feature_betas_ = []
            
        return elimination_results["Index"]

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_cal: np.ndarray, y_cal: np.ndarray, 
            X_test: np.ndarray, y_test: np.ndarray) -> 'CRFE':
        """
        Fit the CRFE feature selector.
        
        This method performs the complete feature selection process using the
        provided training, calibration, and test datasets.
        
        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples_train, n_features)
            Training input samples.
        y_train : np.ndarray of shape (n_samples_train,)
            Training target values.
        X_cal : np.ndarray of shape (n_samples_cal, n_features)
            Calibration input samples.
        y_cal : np.ndarray of shape (n_samples_cal,)
            Calibration target values.
        X_test : np.ndarray of shape (n_samples_test, n_features)
            Test input samples.
        y_test : np.ndarray of shape (n_samples_test,)
            Test target values.
            
        Returns
        -------
        self : CRFE
            The fitted feature selector.
            
        Raises
        ------
        ValueError
            If the datasets have inconsistent numbers of features or classes.
        """
        # Parameter validation
        if self.estimator is None:
            raise ValueError("An estimator must be provided")

        # Efficient array conversion
        array_names = ['X_train', 'y_train', 'X_cal', 'y_cal', 'X_test', 'y_test']
        arrays = [X_train, y_train, X_cal, y_cal, X_test, y_test]
        
        converted_arrays = {}
        for name, arr in zip(array_names, arrays):
            if not isinstance(arr, np.ndarray):
                converted_arrays[name] = np.asarray(arr)
            else:
                converted_arrays[name] = arr

        X_train = converted_arrays['X_train']
        y_train = converted_arrays['y_train']
        X_cal = converted_arrays['X_cal']
        y_cal = converted_arrays['y_cal']
        X_test = converted_arrays['X_test']
        y_test = converted_arrays['y_test']

        # Validation checks
        if X_train.shape[1] != X_cal.shape[1]:
            raise ValueError("Training and calibration datasets must have the same number of features")
        
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError("Training and test datasets must have the same number of features")

        # Validate classes consistency
        train_classes = set(y_train)
        cal_classes = set(y_cal)
        if not train_classes.issubset(cal_classes) or not cal_classes.issubset(train_classes):
            raise ValueError("All classes in training must be present in calibration")

        # Store and encode classes
        self.classes_ = np.unique(y_train)
        
        # Create consistent class encoding
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_train_encoded = np.array([class_to_idx[y] for y in y_train])
        y_cal_encoded = np.array([class_to_idx[y] for y in y_cal])
        
        # Handle binary classification special case
        if len(self.classes_) == 2:
            y_train_encoded, y_cal_encoded, self.classes_ = binary_change(y_train_encoded, y_cal_encoded)
        
        # Perform recursive feature elimination
        self.results_dict_ = self._recursive_elimination(
            X_train, y_train_encoded, X_cal, y_cal_encoded, X_test, y_test
        )

        # Set up final estimator
        self.estimator_ = clone(self.estimator)
        
        return self

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """
        Get a mask or integer index of the selected features.
        
        Parameters
        ----------
        indices : bool, default=False
            If True, return the feature indices. If False, return a boolean mask.
            
        Returns
        -------
        support : np.ndarray or List[int]
            Feature selection mask or indices.
        """
        check_is_fitted(self, 'selected_features_')
        
        if indices:
            return self.selected_features_.tolist()
        else:
            # Assuming we know the original number of features
            # This could be stored during fit for more robustness
            max_feature_idx = np.max(self.selected_features_) if len(self.selected_features_) > 0 else 0
            mask = np.zeros(max_feature_idx + 1, dtype=bool)
            mask[self.selected_features_] = True
            return mask
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X to the selected features.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_selected_features)
            Transformed input with only selected features.
        """
        check_is_fitted(self, 'selected_features_')
        X = np.asarray(X)
        return X[:, self.selected_features_]
    
    def fit_transform(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_cal: np.ndarray, y_cal: np.ndarray, 
                     X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """
        Fit the selector and transform the training data.
        
        Parameters
        ----------
        X_train, y_train : Training data and labels
        X_cal, y_cal : Calibration data and labels
        X_test, y_test : Test data and labels
        
        Returns
        -------
        X_new : np.ndarray
            Transformed training data with selected features.
        """
        return self.fit(X_train, y_train, X_cal, y_cal, X_test, y_test).transform(X_train)
    
    @property
    def idx_features_(self) -> np.ndarray:
        """Backwards compatibility property for selected features."""
        return getattr(self, 'selected_features_', np.array([]))
    
    @property  
    def idx_betas_(self) -> List[float]:
        """Backwards compatibility property for feature betas."""
        return getattr(self, 'feature_betas_', [])
    
    def __repr__(self) -> str:
        """String representation of the CRFE object."""
        return (f"CRFE(estimator={self.estimator}, "
                f"n_features_to_select={self.n_features_to_select}, "
                f"lambda_param={self.lambda_param}, "
                f"epsilon={self.epsilon})")
