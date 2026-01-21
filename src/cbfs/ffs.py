
"""
Floating Feature Selection (FFS) - Main execution module.

This module provides the main entry point for running feature selection experiments
using CRFE (Conformal Recursive Feature Elimination) and mRMR methods with
standardized naming conventions and improved performance.

"""

import numpy as np
import sys
import os
import json
import argparse
import pickle
from functools import lru_cache
from typing import Tuple, Dict, Any, Optional

from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV

from ._utils_crfe import DataReader, h5adDataReader
from ._crfe import CRFE
from ._MrmrMS import FeatureSelector
from ._utils import predict_scores_svm


# Cache random number generator for efficiency
@lru_cache(maxsize=128)
def _get_random_state(seed: int) -> np.random.Generator:
    """Cached random state generator to avoid recreation."""
    return np.random.default_rng(seed)

def calculate_decay_factor(k, U0_initial):
    x = (k/U0_initial)**(1/k)
    return x


def generate_random_integer(seed: int) -> int:
    """
    Generate random integer with caching for efficiency.
    
    Parameters
    ----------
    seed : int
        Random seed
        
    Returns
    -------
    int
        Random integer between 0 and 99999
    """
    rng = _get_random_state(seed)
    return rng.integers(low=0, high=100000)


def create_linear_svc_estimator() -> LinearSVC:
    """
    Factory function for creating optimized LinearSVC estimator.
    
    Returns
    -------
    LinearSVC
        Configured LinearSVC estimator
    """
    return LinearSVC(
        tol=1e-4, 
        loss='squared_hinge',
        max_iter=14000,
        dual="auto"
    )



def split_dataset(X: np.ndarray, y: np.ndarray, run_id: int, 
                 test_size: float = 0.15, cal_size: float = 0.5) -> Tuple[np.ndarray, ...]:
    """
    Split data into training, calibration, and test sets.
    
    Parameters
    ----------
    X : np.ndarray
        Input features
    y : np.ndarray
        Target labels
    run_id : int
        Run identifier for reproducible splits
    test_size : float, default=0.15
        Proportion of data to use for testing
    cal_size : float, default=0.5
        Proportion of remaining data to use for calibration
        
    Returns
    -------
    Tuple[np.ndarray, ...]
        X_train, X_cal, X_test, y_train, y_cal, y_test
        
    Raises
    ------
    ValueError
        If data shapes are inconsistent or sizes are invalid
    """
    if X.shape[0] != len(y):
        raise ValueError("X and y must have the same number of samples")
    
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1")
    
    if not (0 < cal_size < 1):
        raise ValueError("cal_size must be between 0 and 1")

    # Generate deterministic seed based on run_id
    seed = generate_random_integer(42 + run_id)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        shuffle=True, 
        stratify=y, 
        random_state=seed
    )
    
    # Second split: separate training and calibration
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, 
        test_size=cal_size, 
        shuffle=True, 
        stratify=y_temp, 
        random_state=seed
    )
    
    return X_train, X_cal, X_test, y_train, y_cal, y_test


def run_crfe_experiment(estimator, X_train: np.ndarray, y_train: np.ndarray, 
                       X_cal: np.ndarray, y_cal: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray,
                       perc_features_to_remove = 1) -> Dict[str, Any]:
    
    """
    Run CRFE experiment with the given data splits.
    
    Parameters
    ----------
    estimator : sklearn estimator
        Base estimator for feature selection
    X_train, y_train : Training data and labels
    X_cal, y_cal : Calibration data and labels
    X_test, y_test : Test data and labels
    perc_features_to_remove : float, default=0 Percentage of features to remove. If 0, removes 1 feature.
    
    Returns
    -------
    Dict[str, Any]
        Results dictionary or error information
    """

    
    
    
    n_features = X_train.shape[1]
    feature_indices = np.arange(n_features)

    if isinstance(perc_features_to_remove, float):
        n_features_to_remove = int(n_features * perc_features_to_remove)
    else:
        n_features_to_remove = perc_features_to_remove
    
    if n_features <= 1:
        raise ValueError("Need at least 2 features for feature selection")
    
    # Create CRFE with standardized parameters
    crfe = CRFE(
        estimator=estimator, 
        n_features_to_select=n_features - n_features_to_remove, 
        lambda_param=0.5,  # Default value
        epsilon=0.4        # Default value
    )
    
    # Fit the CRFE model
    crfe.fit(X_train, y_train, X_cal, y_cal, X_test, y_test)

    list_of_selected_features = crfe.results_dict_

    
    if len(list_of_selected_features) > 1:
        list_of_selected_features = np.array(list_of_selected_features[-1])
    

    removed_features = np.setdiff1d(feature_indices, list_of_selected_features)

    return removed_features



def run_mrmr_experiment_(estimator, X_train: np.ndarray, y_train: np.ndarray, 
                       X_cal: np.ndarray, y_cal: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray,
                       selected_features, per_feat_add: int = 1) -> Dict[str, Any]:
    """
    Run mRMR-MS experiment with the given data splits.
    
    Parameters
    ----------
    estimator : sklearn estimator
        Base estimator (unused in mRMR but kept for consistency)
    X_train, y_train : Training data and labels
    X_cal, y_cal : Calibration data and labels (unused)
    X_test, y_test : Test data and labels (unused)
    
    Returns
    -------
    Dict[str, Any]
        Results dictionary or error information
    """
    try: 
        selected_features = selected_features.tolist()
    except:
        pass

    try:
        n_classes = len(np.unique(y_train))
        
        if n_classes < 2:
            raise ValueError("Need at least 2 classes for classification")
        
        # Create mRMR feature selector
        # Disable parallel processing on Windows to avoid multiprocessing issues
        import platform
        use_parallel = True if platform.system() != "Windows" else False
        
        FFS = FeatureSelector(
            classes_=[i for i in range(n_classes)],
            max_features=len(selected_features)+per_feat_add, 
            parallel=use_parallel,
            verbose=True
        )
        
        # mRMR-MS parameters
        kernel = "linear"     # Options: "linear", "rbf", "poly"
        split_size = 0.5      # Split size for mRMR-MS if applied
        
        # Run mRMR-MS feature selection
        #mrmr.mRMR_MS(X_train, y_train, kernel, split_size,  selected_features)
        FFS.JMI(X_train, y_train,  selected_features)

        return FFS.all_selected_features[-1] if hasattr(FFS, 'all_selected_features') else {}
    
    except Exception as e:
        print(f"Error in mRMR experiment: {str(e)}")
        return {'error': str(e)} 





class FloatingFeatureSelector:
    """
    Floating Feature Selection (FFS) experiment manager.
    
    This class provides a structured approach to running feature selection experiments
    using both CRFE (Conformal Recursive Feature Elimination) and mRMR methods.
    
    Attributes
    ----------
    run_id : int
        Run identifier for reproducibility
    data_path : str
        Path to data or "synthetic" for artificial data
    test_size : float
        Proportion of data to use for testing
    cal_size : float
        Proportion of remaining data to use for calibration
    estimator : sklearn estimator
        Base estimator for feature selection
    data_reader : DataReader
        Data loading utility
    experiment_results : Dict[str, Any]
        Storage for experiment results
    """
    
    def __init__(self, run_id: int = 1, data_path: str = "synthetic", 
                 test_size: float = 0.15, cal_size: float = 0.5,
                 estimator=None, max_patience: int = 3, 
                 target_column: Optional[str] = None):
        """
        Initialize the Floating Feature Selector.
        
        Parameters
        ----------
        run_id : int, default=1
            Run identifier for reproducibility
        data_path : str, default="synthetic"
            Path to data or "synthetic" for artificial data
        test_size : float, default=0.15
            Proportion of data to use for testing
        cal_size : float, default=0.5
            Proportion of remaining data to use for calibration
        estimator : sklearn estimator, optional
            Base estimator for feature selection. If None, creates LinearSVC
        max_patience : int, default=5
            Maximum number of consecutive iterations without improvement before stopping
            to avoid local minima
        """
        self.run_id = run_id
        self.data_path = data_path
        self.test_size = test_size
        self.cal_size = cal_size
        self.max_patience = max_patience
        self.estimator = estimator or create_linear_svc_estimator()
        self.data_reader = DataReader()
        self.target_column = target_column
        
        # Initialize storage for experiment components
        self.X = None
        self.y = None
        self.class_names = None
        self.classes_ = None  # For sklearn compatibility
        self.lambda_param = 0.5  # Default lambda parameter
        self.lambda_p_param = None  # Will be calculated based on number of classes
        self.X_train = None
        self.X_cal = None
        self.X_test = None
        self.y_train = None
        self.y_cal = None
        self.y_test = None
        
        # Floating Feature Selection specific attributes
        self.S = None  # Selected features set
        self.U_start = None  
        self.U = None  # Available features set
        self.best_metric = None
        self.best_subset = None
        self.new_scores = None
        self.new_subsets = None  # Store corresponding subsets for new_scores
        self.counter = 0
        
        # Patience mechanism for avoiding local minima
        self.patience_counter = 0  # Track consecutive iterations without improvement
        self.global_best_metric = None  # Track absolute best metric seen
        self.global_best_subset = None  # Track absolute best subset seen
        self.improvement_tol = 1e-8  # tolerancia para empates numéricos
        self.max_restarts = 1        # reinicios permitidos al agotar paciencia
        self.restart_counter = 0
        
        # Conformal prediction metrics
        self.Empirical_coverage_ = None
        self.Uncertainty_ = None
        self.Certainty_ = None
        
        self.experiment_results = {}
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:


        if "h5ad" in self.data_path:
            data_reader = h5adDataReader(test_size=0.20,cal_size=0.30, random_state=self.run_id, var_keep_top=3500)
            splits, class_names = data_reader.load_data(self.data_path, self.target_column)

            self.X_train, self.X_cal, self.X_test, self.y_train, self.y_cal, self.y_test = splits

            print(f"Train shape: {self.X_train.shape}, Cal shape: {self.X_cal.shape}, Test shape: {self.X_test.shape}")
            # Set classes and calculate lambda_p_param
            self.classes_ = np.unique(self.y_train)
            if len(self.classes_) > 2:
                self.lambda_p_param = (1 - self.lambda_param) / (len(self.classes_) - 1)
            else:
                self.lambda_p_param = 0.0

        else:

            self.X, self.y, self.class_names = self.data_reader.load_data(self.data_path)
        
            print(f"Dataset loaded: {self.X.shape} samples, {len(self.class_names)} classes: {self.class_names}")
            sys.stdout.flush()

            self.split_data()
        
        return None
    

    def split_data(self) -> Tuple[np.ndarray, ...]:
        """
        Split loaded data into training, calibration, and test sets.
        
        Returns
        -------
        Tuple[np.ndarray, ...]
            X_train, X_cal, X_test, y_train, y_cal, y_test
            
        Raises
        ------
        ValueError
            If data has not been loaded yet
        """
        if self.X is None or self.y is None:
            raise ValueError("Data must be loaded before splitting. Call load_data() first.")
            
        splits = split_dataset(self.X, self.y, self.run_id, self.test_size, self.cal_size)
        self.X_train, self.X_cal, self.X_test, self.y_train, self.y_cal, self.y_test = splits
        
        # Set classes and calculate lambda_p_param
        self.classes_ = np.unique(self.y_train)
        if len(self.classes_) > 2:
            self.lambda_p_param = (1 - self.lambda_param) / (len(self.classes_) - 1)
        else:
            self.lambda_p_param = 0.0
        
        print(f"Data split - Train: {self.X_train.shape}, Cal: {self.X_cal.shape}, Test: {self.X_test.shape}")
        sys.stdout.flush()
        
        return splits
    
    def init_S_U(self):

        """Initialize the selected (S) and unselected (U) feature sets."""
        if self.S is None:
            
            S = set() #set(np.random.choice(self.X_train.shape[1], size=int(self.X_train.shape[1] * 0.15), replace=False))
            U_start = set()


            all_features = set(range(self.X_train.shape[1])) #idx of all features
            U = all_features - S  # Complementary set

            self.S = np.array(sorted(S), dtype=int)                # selected features
            self.U_start = np.array(sorted(U_start), dtype=int)    # removed features
            self.U = np.array(sorted(U), dtype=int)                # unselected features      


        print(f"Initial S (selected): {len(self.S)} ")
        print(f"Initial U (unselected): {len(self.U)} ")
        print(f"Initial U_start (removed): {len(self.U_start)} \n")


        return None
    


    def update(self, update_flag=True):
        if not update_flag:
            self.best_metric = np.inf
            self.best_subset = None
            return None

        # Handle initialization
        if self.best_metric is None:
            self.best_metric = self.Uncertainty_
            self.best_subset = self.S.copy()
            return None

        # Logic to find the BEST improvement in the current batch
        if hasattr(self, 'new_scores') and self.new_scores is not None:
            # Find the index of the absolute best score in this batch
            best_in_batch_idx = np.argmin(self.new_scores)
            best_in_batch_val = self.new_scores[best_in_batch_idx]

            # Compare the best of the batch against the global best
            is_improvement = best_in_batch_val < (self.best_metric - self.improvement_tol)
            is_tie_but_new = (
                abs(best_in_batch_val - self.best_metric) <= self.improvement_tol
                and hasattr(self, 'new_subsets') and self.new_subsets is not None
                and self.best_subset is not None
                and not np.array_equal(self.new_subsets[best_in_batch_idx], self.best_subset)
            )

            if is_improvement or is_tie_but_new:
                self.best_metric = best_in_batch_val

                # Ensure subset is updated in sync
                if hasattr(self, 'new_subsets') and self.new_subsets is not None:
                    self.best_subset = self.new_subsets[best_in_batch_idx].copy()

                return best_in_batch_idx

    



    def init_method(self):

        """Initialize the floating feature selection method."""
        
        self.init_S_U()  # random init if not initialized

        # Calculate initial scores using the utility function
        if self.S is None or len(self.S) == 0:

            self.Empirical_coverage_ = 0.0
            self.Uncertainty_ = np.inf
            self.Certainty_ = 0.0

        else:

            X_train_init = self.X_train[:,  self.S]
            X_cal_init = self.X_cal[:,  self.S]  
            X_test_init = self.X_test[:, self.S]

            scores = predict_scores_svm(
                self.estimator, self.classes_, self.lambda_param, self.lambda_p_param,
                X_train_init, self.y_train, X_cal_init, self.y_cal, X_test_init, self.y_test
            )

            self.Empirical_coverage_ = scores[0]
            self.Uncertainty_ = scores[1]
            self.Certainty_ = scores[2]

        print(f"Initial scores with features {self.S}:")
        print(f"  Empirical Coverage: {self.Empirical_coverage_:.4f}")
        print(f"  Uncertainty: {self.Uncertainty_:.4f}")
        print(f"  Certainty: {self.Certainty_:.4f}")

        self.update()

    
        return None
    


    def run_crfe_experiment_U(self, per_feat_remove=0.1) -> Dict[str, Any]:
        """
        Run CRFE (Conformal Recursive Feature Elimination) experiment.
        
        """

        if self.X_train is None:
            raise ValueError("Data must be split before running experiments. Call split_data() first.")
            
        print("Running CRFE experiment...")
        sys.stdout.flush()


        X_train_U = self.X_train[:, self.U].copy()
        X_cal_U = self.X_cal[:, self.U].copy()
        X_test_U = self.X_test[:, self.U].copy()


        removed_feature = run_crfe_experiment(
            self.estimator, X_train_U, self.y_train, 
            X_cal_U, self.y_cal, X_test_U, self.y_test,
            perc_features_to_remove=per_feat_remove
        )
        

        f_removed_feature = self.U[removed_feature]
        print("Removed feature: ", f_removed_feature)
        
        return f_removed_feature
    
    def run_crfe_experiment(self, per_feat_remove = 1) -> Dict[str, Any]:
        """
        Run CRFE (Conformal Recursive Feature Elimination) experiment.
        
        Returns
        -------
        Dict[str, Any]
            CRFE experiment results
            
        Raises
        ------
        ValueError
            If data has not been split yet
        """
        if self.X_train is None:
            raise ValueError("Data must be split before running experiments. Call split_data() first.")
            
        print("Running CRFE experiment...")
        sys.stdout.flush()


        X_train_S = self.X_train[:, self.S].copy()
        X_cal_S = self.X_cal[:, self.S].copy()
        X_test_S = self.X_test[:, self.S].copy()


        removed_feature = run_crfe_experiment(
            self.estimator, X_train_S, self.y_train, 
            X_cal_S, self.y_cal, X_test_S, self.y_test,
            perc_features_to_remove=per_feat_remove
        )
        

        f_removed_feature = self.S[removed_feature]
        print("Removed feature: ", f_removed_feature)
        
        return f_removed_feature
    
    
    
    def run_mrmr_experiment(self, per_feat_add =  1 ) -> Dict[str, Any]:
        """
        Run CMI (minimum Redundancy Maximum Relevance - Multi-class SVM) experiment.
        
        """
        if self.X_train is None:
            raise ValueError("Data must be split before running experiments. Call split_data() first.")
            
        print("Running JMI experiment...")
        sys.stdout.flush()

        #X_train_U = self.X_train.copy()
        #X_cal_U = self.X_cal.copy()
        #X_test_U = self.X_test.copy()
        S_U = np.sort(self.U.tolist() + self.S.tolist())
        encoder = {col: idx for idx, col in enumerate(S_U)}
        encoded_S = [encoder[col] for col in self.S]

        X_train_U = self.X_train[:, S_U].copy()
        X_cal_U = self.X_cal[:, S_U].copy()
        X_test_U = self.X_test[:, S_U].copy()

        S = run_mrmr_experiment_(
            self.estimator, X_train_U, self.y_train, 
            X_cal_U, self.y_cal, X_test_U, self.y_test, encoded_S, per_feat_add=per_feat_add)
        
        #f_added_feature = S[-per_feat_add]
        #print(f"Added feature: {f_added_feature}")

        S = list(S)
        new_features = S[-per_feat_add:] if len(S) >= per_feat_add else S
        f_added_features = np.array([S_U[f] for f in new_features])
        print(f"\nAdded features: {f_added_features}")
        
        return f_added_features
    
    def _evaluate_moves(self, f_removed, f_added):
        """
        Evaluate three potential moves: removal, addition, and swap.
        
        Parameters
        ----------
        f_removed : int
            Feature index to be removed from current subset S
        f_added : int  
            Feature index to be added to current subset S
            
        Returns
        -------
        dict
            Dictionary containing metrics for each move type
        """
        current_S = self.S.copy()
        
        
        # 1. Removal move: S_minus = S \ {f_removed}       
        S_minus = np.setdiff1d(current_S, f_removed)

    
        # Extract training, calibration, and test data for removal move
        X_train_minus = self.X_train[:, S_minus]
        X_cal_minus = self.X_cal[:, S_minus]  
        X_test_minus = self.X_test[:, S_minus]
    
        # Compute scores for removal move
        scores_minus = predict_scores_svm(
            self.estimator, self.classes_, self.lambda_param, self.lambda_p_param,
            X_train_minus, self.y_train, X_cal_minus, self.y_cal, X_test_minus, self.y_test
        )

        metric_minus = scores_minus[1]  # Using Uncertainty as the metric
    
        # 2. Addition move: S_plus = S ∪ {f_added}  
        S_plus = np.append(current_S, f_added)
    
        # Extract training, calibration, and test data for addition move
        X_train_plus = self.X_train[:, S_plus]
        X_cal_plus = self.X_cal[:, S_plus]
        X_test_plus = self.X_test[:, S_plus]
    
        # Compute scores for addition move
        scores_plus = predict_scores_svm(
            self.estimator, self.classes_, self.lambda_param, self.lambda_p_param,
            X_train_plus, self.y_train, X_cal_plus, self.y_cal, X_test_plus, self.y_test
        )
        metric_plus = scores_plus[1]  # Using Uncertainty as the metric
    
        # 3. Swap move: S_swap = (S \ {f_removed}) ∪ {f_added}
        S_swap = np.setdiff1d(current_S, f_removed)
        S_swap = np.append(S_swap, f_added)
    
        # Extract training, calibration, and test data for swap move  
        X_train_swap = self.X_train[:, S_swap]
        X_cal_swap = self.X_cal[:, S_swap]
        X_test_swap = self.X_test[:, S_swap]
    
        # Compute scores for swap move
        scores_swap = predict_scores_svm(
            self.estimator, self.classes_, self.lambda_param, self.lambda_p_param,
            X_train_swap, self.y_train, X_cal_swap, self.y_cal, X_test_swap, self.y_test
        )
        metric_swap = scores_swap[1]  # Using Uncertainty as the metric
    
        # Store results
        move_results = {
            'removal': {
                'subset': S_minus,
                'metric': metric_minus,
                'scores': scores_minus
            },
            'addition': {
                'subset': S_plus, 
                'metric': metric_plus,
                'scores': scores_plus
            },
            'swap': {
                'subset': S_swap,
                'metric': metric_swap, 
                'scores': scores_swap
            }
        }


    
    
        print(f"Move evaluation results:")
        print(f"  Current best metric: {self.best_metric:.4f}")
        print(f"  Removal move metric: {metric_minus:.4f}")
        print(f"  Addition move metric: {metric_plus:.4f}")  
        print(f"  Swap move metric: {metric_swap:.4f}")

        # Store the new scores and corresponding subsets for update method
        self.new_scores = [metric_minus, metric_plus, metric_swap]
        self.new_subsets = [S_minus, S_plus, S_swap]
        
        # Try to update with new scores
        improvement_index = self.update()
        
        if improvement_index is not None:
            # An improvement was found
            move_names = ['removal', 'addition', 'swap']
            best_move_name = move_names[improvement_index]
            best_move_subset = self.new_subsets[improvement_index]
            best_move_scores = [scores_minus, scores_plus, scores_swap][improvement_index]
            
            print(f"  Improvement found! Best move: {best_move_name}")
            print(f"  New best metric: {self.best_metric:.4f}")
            
            # Reset patience counter on improvement
            self.patience_counter = 0
            
            # Update global best if this is the best we've seen
            if self.global_best_metric is None or self.best_metric <= self.global_best_metric:
                self.global_best_metric = self.best_metric
                self.global_best_subset = best_move_subset.copy()
                print(f"  New global best metric: {self.global_best_metric:.4f}")
            
            # Update current feature sets
            self.S = best_move_subset.copy()
            all_features = set(range(self.X_train.shape[1]))
            self.U = np.array(sorted(all_features - set(self.S)), dtype=int)
            
            # Update current conformal prediction metrics
            self.Empirical_coverage_ = best_move_scores[0]
            self.Uncertainty_ = best_move_scores[1] 
            self.Certainty_ = best_move_scores[2]
            
            return move_results
        
        else:
            # No immediate improvement found
            self.patience_counter += 1
            print(f"  No improvement found. Patience: {self.patience_counter}/{self.max_patience}")
            
            if self.patience_counter >= self.max_patience:

                print(f"  Patience exhausted ({self.max_patience} iterations without improvement).")
                if self.global_best_subset is not None:
                    print(f"  → Reverting to global best subset with metric: {self.global_best_metric:.4f}")
                    # Revert to global best
                    self.S = self.global_best_subset.copy()
                    self.best_metric = self.global_best_metric
                    all_features = set(range(self.X_train.shape[1]))
                    self.U = np.array(sorted(all_features - set(self.S)), dtype=int)

                if self.global_best_subset is not None and self.restart_counter < self.max_restarts:
                    self.restart_counter += 1
                    self.patience_counter = 0
                    print(f"  → Restarting from global best (restart {self.restart_counter}/{self.max_restarts}).")
                    move_results['improvement'] = False
                    move_results['best_move'] = None
                    move_results['patience_exhausted'] = False
                    return move_results  # continue loop

                print(f"  → Stopping FFS due to exhausted patience.\n")
                move_results['improvement'] = False
                move_results['best_move'] = None
                move_results['patience_exhausted'] = True
                return None  # Signal to stop the FFS loop
            else:
                print(f"  → Continuing search (patience remaining: {self.max_patience - self.patience_counter}).\n")
                move_results['improvement'] = False
                move_results['best_move'] = None
                move_results['patience_exhausted'] = False
                return move_results  # Continue searching despite no improvement
            
    
    def _run_ffs(self,n_feat = 0.1):

        if n_feat is None:
            print("n_feat not specified, defaulting to 10% of total unselected features.")
            n_feat = int(0.1 * len(self.U))
        if isinstance(n_feat, float):
            n_feat = int(n_feat * len(self.U))
        else:
            n_feat = n_feat
        
        decay = calculate_decay_factor(n_feat , len(self.U))  # compute decay factor for CRFE
        print(f"Target number of features to select: {n_feat}")
        print(f"Decay factor for CRFE: {decay:.4f}\n")
        n_feat_to_rem = len(self.U)
        
        while len(self.S) < n_feat:

            f_added = self.run_mrmr_experiment()  

            self.S = np.append(self.S.copy(), f_added).copy()
            self.U = np.setdiff1d(self.U.copy(), f_added).copy()

            n_feat_to_rem = int(n_feat_to_rem * decay )
            f_removed = self.run_crfe_experiment_U(per_feat_remove=len(self.U) - n_feat_to_rem) 
             

            self.U_start = np.append(self.U_start.copy(), f_removed).copy()
            self.U = np.setdiff1d(self.U.copy(), f_removed).copy()

            print("\n#######\n")
            print("Current features selected: ",len(self.S))
            print("Current features unselected: ",len(self.U))
            print("Current features removed: ",len(self.U_start), "\n")
            print("\n#######\n")

        print("\nStarting Floating Feature Selection iterations...\n")

        merged_array = np.concatenate((self.U, self.U_start))
        self.U = np.unique(merged_array).copy()
        
        # Initialize global best metrics at the start of FFS iterations
        if self.global_best_metric is None:
            self.global_best_metric = self.best_metric
            self.global_best_subset = self.S.copy()
            print(f"Initialized global best metric: {self.global_best_metric:.4f}")

        # Run experiments with patience mechanism
        max_iterations = 20  # Increased to allow for patience exploration
        for i in range(max_iterations):
            
            if self.patience_counter > 0:
                max_move = min(len(self.S), len(self.U))
                if max_move == 0:
                    print("No moves possible (S or U empty). Stopping.")
                    break
                
                if self.patience_counter >= self.max_patience:
                    print("Maximum patience reached, stopping FFS.")
                    break
                #n_feat_to_eval = self.patience_counter + 1
                n_feat_to_eval = min(self.patience_counter + 1, max_move)
            else:
                n_feat_to_eval = 1

            f_removed = self.run_crfe_experiment(per_feat_remove=n_feat_to_eval)  #self.f_removed
            f_added = self.run_mrmr_experiment(per_feat_add=n_feat_to_eval)  #self.f_added

            ret = self._evaluate_moves(f_removed, f_added)
            print(f"Iteration {i+1} completed.")
            print("Current features selected: ", self.S, "\n")

            # Check stopping conditions
            if ret is None:
                print("Stopping due to termination condition.")
                break
            elif isinstance(ret, dict) and ret.get('patience_exhausted', False):
                print("Stopping due to exhausted patience.")
                break
            
        print(f"FFS completed after {i+1} iterations.")
        if self.global_best_subset is not None:
            print(f"Final global best metric: {self.global_best_metric:.4f}")
            print(f"Final feature subset: {self.global_best_subset}")
        
        return self.global_best_subset
        
    

    def run_ffs(self, n_feat = None) -> Dict[str, Any]:
        """
        
        Returns
        -------
        Dict[str, Any]
            Combined results from all experiments
        
        n_feat : int or float, optional
            Number or proportion of features to select. If float, treated as proportion.
        """
        # Load and prepare data
        self.load_data()
        self.init_method()

        # Run your floating feature selection algorithm
        ffs_results = self._run_ffs(n_feat=n_feat)
        
        
        return ffs_results
    



if __name__ == "__main__":
    """Main execution block with standardized parameters."""
  
    run_id = 1              # Fixed seed for reproducibility
    data_path = "synthetic"  # Use synthetic dataset


        
    ffs = FloatingFeatureSelector(run_id=run_id, data_path=data_path)
    results = ffs.run_ffs()


    
    print("Experiment completed successfully!")
        

