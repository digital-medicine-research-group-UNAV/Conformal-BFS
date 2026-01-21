"""
This module provides utility functions for the Conformal Recursive Feature Elimination
algorithm with standardized naming conventions and performance improvements.
"""

import numpy as np
import pandas as pd
import os
from typing import Tuple, Union, List, Optional
from numba import jit, njit

from scipy.stats import zscore
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
import warnings

warnings.filterwarnings("ignore")

_NEUROCOMBAT_ONEHOT_PATCHED = False

def _patch_neurocombat_onehot():
    global _NEUROCOMBAT_ONEHOT_PATCHED
    if _NEUROCOMBAT_ONEHOT_PATCHED:
        return
    try:
        import neurocombat_sklearn.neurocombat_sklearn as ncs
        from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder
        import inspect
    except Exception:
        return

    params = inspect.signature(SkOneHotEncoder).parameters
    if "sparse" in params or "sparse_output" not in params:
        _NEUROCOMBAT_ONEHOT_PATCHED = True
        return

    def _OneHotEncoderCompat(*args, **kwargs):
        if "sparse" in kwargs:
            kwargs["sparse_output"] = kwargs.pop("sparse")
        return SkOneHotEncoder(*args, **kwargs)

    ncs.OneHotEncoder = _OneHotEncoderCompat
    _NEUROCOMBAT_ONEHOT_PATCHED = True

@njit
def _find_argmax_fast(beta_values: np.ndarray) -> int:
    """Optimized argmax finding using numba JIT compilation."""
    return np.argmax(beta_values)


def to_list(data: Union[List, np.ndarray]) -> List:
    """
    Convert array-like data to list format.
    
    Parameters
    ----------
    data : array-like
        Input data to convert
        
    Returns
    -------
    List
        Data in list format
        
    Raises
    ------
    ValueError
        If input is not a valid list or array
    """
    if isinstance(data, list):
        return data
    elif isinstance(data, np.ndarray):  
        return data.tolist()
    else:
        raise ValueError("Error: input must be a valid list or array")


def binary_change(y_train: np.ndarray, y_cal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Binary label transformation for two-class problems.
    
    Transforms binary labels to ensure consistent encoding with values {-1, 1}.
    
    Parameters
    ----------
    y_train : np.ndarray
        Training labels
    y_cal : np.ndarray
        Calibration labels
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Transformed training labels, calibration labels, and unique class names
    """
    # Efficient vectorized operations: map 0 -> -1, keep other values
    y_train_transformed = np.where(y_train == 0, -1, y_train)
    y_cal_transformed = np.where(y_cal == 0, -1, y_cal)
    
    # Get unique classes and re-encode labels consistently
    unique_classes, y_train_encoded = np.unique(y_train_transformed, return_inverse=True)
    print(f"Binary classes: {unique_classes}")

    return y_train_encoded, y_cal_transformed, unique_classes


def find_argmax(beta_values: Union[List, np.ndarray]) -> int:
    """
    Find index of maximum value in beta array.
    
    Parameters
    ----------
    beta_values : array-like
        Beta values to find maximum from
        
    Returns
    -------
    int
        Index of maximum value
    """
    return _find_argmax_fast(np.array(beta_values))


def create_artificial_dataset(n_samples: int = 350, n_informative_features: int = 10, 
                            n_classes: int = 4, n_random_features: int = 25, 
                            random_seed: int = 12345, 
                            normalization: str = "zscore") -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Generate artificial dataset with optimized operations.
    
    Parameters
    ----------
    n_samples : int, default=350
        Number of samples to generate
    n_informative_features : int, default=10  
        Number of informative features
    n_classes : int, default=4
        Number of classes
    n_random_features : int, default=25
        Number of random noise features to add
    random_seed : int, default=12345
        Random seed for reproducibility
    normalization : str, default="zscore"
        Normalization method: "zscore" or "minmax"
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List]
        Features, labels, and class names
    """
    
    # Generate base classification dataset
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_informative_features, 
        n_redundant=0, 
        n_classes=n_classes,
        n_informative=n_informative_features, 
        n_clusters_per_class=1, 
        class_sep=1.5, 
        flip_y=0.05, 
        scale=None, 
        random_state=random_seed, 
        weights=[0.25] * n_classes, 
        shuffle=True
    )
    
    # Add random noise features efficiently
    rng = np.random.RandomState(random_seed)
    random_features = rng.randint(10, size=(n_samples, n_random_features))
    X = np.hstack([X, random_features])
    
    # Apply normalization
    if normalization == "zscore":
        X = zscore(X, axis=1)
    else:
        X = MinMaxScaler().fit_transform(X)

    # Create consistent class names and labels
    unique_classes = sorted(np.unique(y))
    
    # Vectorized label conversion
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    y_encoded = np.array([class_to_idx[cls] for cls in y])

    print(f"Classes: {unique_classes}")

    return X, y_encoded, unique_classes


class DataReader:
    """
    Optimized data reader with better performance and standardized interface.
    
    This class provides methods to load data from various sources including
    synthetic datasets and CSV files.
    """
    
    def __init__(self):
        """Initialize the DataReader."""
        self.X = None
        self.y = None
        self.n_classes = None
        self.n_samples = None
        self.n_features = None

    def load_data(self, data_path: str, target_column: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data with optimized I/O operations.
        
        Parameters
        ----------
        data_path : str
            Path to data or "synthetic" for artificial data generation
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Features, labels, and class names
            
        Raises
        ------
        ValueError
            If path is invalid
        FileNotFoundError
            If data files are not found
        """
        
        if not isinstance(data_path, str):
            raise ValueError("Error: data path must be a valid string")

        if data_path == "synthetic":
            # Generate synthetic dataset
            X, y, class_names = create_artificial_dataset(
                n_samples=350, 
                n_informative_features=10, 
                n_classes=4, 
                n_random_features=25, 
                random_seed=12345, 
                normalization="zscore"
            )
            
            self.n_classes = len(class_names)
            self.n_samples = X.shape[0]
            self.n_features = X.shape[1]
 
        else:

            # Load from file
            full_data_path =  data_path

            print(f"Loading data from: {full_data_path}")
             
            # Efficient CSV reading with appropriate dtypes
            try:
                if target_column == "recist":
                    df = pd.read_csv(full_data_path, header=0, index_col = 0)
                else:
                    df = pd.read_csv(full_data_path, header=0)
                
                df = df.dropna(subset=[target_column])

                print("We consider the first column as labels and the rest as features!!!!")
                
                y = df[target_column].values
                X = df.drop(columns=[target_column]).values
                
                #X = df.iloc[:, 2:].values
                #y = df.iloc[:, 1].values 

                # Create a mapping from unique response labels to integers
                unique_labels = pd.Series(y).dropna().unique()
                label_map = {label: idx for idx, label in enumerate(unique_labels)}
                print(f"Label mapping: {label_map}")
                # Map the response labels in y to integers using label_map
                y = pd.Series(y).map(label_map).values
                
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Data files not found in {full_data_path}: {e}")

            print(f"Data shape: X={X.shape}, y={y.shape}")
            
            # Get unique classes and encode consistently
            class_names, y_encoded = np.unique(y, return_inverse=True)
            y = y_encoded

            self.n_classes = len(class_names)
            self.n_samples = X.shape[0]
            self.n_features = X.shape[1]

        # Store data for potential reuse
        self.X = X
        self.y = y

        return X, y, class_names
        
    def get_n_classes(self) -> Optional[int]:
        """Get number of classes."""
        return self.n_classes

    def get_n_samples(self) -> Optional[int]:
        """Get number of samples."""
        return self.n_samples

    def get_n_features(self) -> Optional[int]:
        """Get number of features."""
        return self.n_features


@njit
def _compute_ncm_optimized(X: np.ndarray, y: np.ndarray, weights: np.ndarray, 
                          bias: np.ndarray, lambda_param: float, lambda_p_param: float, 
                          is_multiclass: bool) -> np.ndarray:
    """
    Optimized Non-Conformity Measure computation using numba.
    
    Parameters
    ----------
    X : np.ndarray
        Input features
    y : np.ndarray
        Labels
    weights : np.ndarray
        Model weights
    bias : np.ndarray
        Model bias terms
    lambda_param : float
        Lambda parameter for multiclass
    lambda_p_param : float
        Lambda prime parameter for multiclass
    is_multiclass : bool
        Whether this is a multiclass problem
        
    Returns
    -------
    np.ndarray
        Non-conformity measures
    """
    if not is_multiclass:
        # Binary case - vectorized computation
        scores = np.sum(weights * X, axis=1) + bias
        result = -(y * scores)
        return result.astype(np.float64)
    else:
        # Multiclass case
        n_samples = X.shape[0]
        ncm_values = np.zeros(n_samples, dtype=np.float64)
        
        for i in range(n_samples):
            y_label = int(y[i])
            
            # Compute first term: -lambda * (w_y^T x + b_y)
            term1 = -lambda_param * (np.dot(weights[y_label], X[i]) + bias[y_label])
            
            # Sum over all classes except y_label
            term2_sum = 0.0
            for k in range(weights.shape[0]):
                if k != y_label:
                    term2_sum += np.dot(weights[k], X[i]) + bias[k]
            
            # Second term: lambda_p * sum_{k != y}(w_k^T x + b_k)
            term2 = lambda_p_param * term2_sum
            ncm_values[i] = term1 + term2
            
        return ncm_values


def compute_nonconformity_measures(X: np.ndarray, y: np.ndarray, 
                                 weights: np.ndarray, bias: np.ndarray, 
                                 lambda_param: Optional[float] = None, 
                                 lambda_p_param: Optional[float] = None, 
                                 is_multiclass: bool = False) -> List[float]:
    """
    Compute Non-Conformity Measures with optimized implementation.
    
    This function provides backwards compatibility with the original API while using
    optimized implementations under the hood.
    
    Parameters
    ----------
    X : np.ndarray
        Input features
    y : np.ndarray
        Labels
    weights : np.ndarray
        Model weights/coefficients
    bias : np.ndarray
        Model bias terms
    lambda_param : float, optional
        Lambda parameter for multiclass problems
    lambda_p_param : float, optional  
        Lambda prime parameter for multiclass problems
    is_multiclass : bool, default=False
        Whether this is a multiclass problem
        
    Returns
    -------
    List[float]
        Non-conformity measure values
        
    Raises
    ------
    ValueError
        If lambda parameters are missing for multiclass problems
    """
    # Ensure consistent data types
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    weights = np.asarray(weights, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    
    # Handle different input formats for y
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.ravel()
    
    if not is_multiclass:
        # Binary classification case
        y = y.astype(np.float64)
        scores = np.sum(weights * X, axis=1) + bias
        return (-(y * scores)).tolist()
    else:
        # Multiclass case - validate parameters
        y = y.astype(np.int32)
        if lambda_param is None or lambda_p_param is None:
            raise ValueError("Lambda parameters must be provided for multiclass problems")
        
        # Use the optimized numba function
        result = _compute_ncm_optimized(X, y, weights, bias, lambda_param, lambda_p_param, True)
        return result.tolist()


# Backwards compatibility aliases
NC_OvsA_SVMl_dev = compute_nonconformity_measures
READER = DataReader  # Backwards compatibility alias



import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit


def _to_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)

def _sum_rows(X):
    if sparse.issparse(X):
        return np.asarray(X.sum(axis=1)).ravel()
    return X.sum(axis=1)

def logcpm(X_counts):
    """
    Library-size normalize to CPM then log1p.
    Returns dense float32 (n_samples x n_genes).
    """
    lib = _sum_rows(X_counts).astype(np.float64)
    lib[lib == 0] = np.nan

    if sparse.issparse(X_counts):
        Xcpm = X_counts.multiply(1e6 / lib[:, None])
        Xlog = Xcpm.log1p()
        out = Xlog.toarray().astype(np.float32)
    else:
        Xcpm = (X_counts / lib[:, None]) * 1e6
        out = np.log1p(Xcpm).astype(np.float32)

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out

import numpy as np
import pandas as pd
from scipy import sparse

# assumes you already have these utilities in your codebase:
#   _sum_rows(X)  -> row sums for sparse/dense
#   logcpm(X)     -> library-size normalize to CPM then log1p, returns dense float32


class DCB6Preprocessor:
    """
    Train-only, leakage-safe preprocessing:

      1) Low-expression filter on TRAIN counts:
         keep genes with CPM >= cpm_threshold in >= min_frac_samples of TRAIN samples
      2) Library-size normalization + transform:
         logCPM (log1p(CPM)) applied after train-derived gene mask
      3) Low-variance filter on TRAIN logCPM:
         keep top var_keep_top genes
      4) Batch correction (ComBat) fit on TRAIN, applied to CAL/TEST:
         uses neurocombat-sklearn CombatModel (scikit-learn style, inductive).

    Notes:
      - Requires adata.obs[batch_col] to exist.
      - covariates are optional (e.g., ["AGE","SEX","TMB"] if present).
      - If neurocombat-sklearn is not installed, raises ImportError (no silent leakage fallback).
    """

    def __init__(
        self,
        batch_col="dataset",
        covariates=None,
        cpm_threshold=1.0,
        min_frac_samples=0.05,
        var_keep_top=5000,
        combat_ref_batch=None,   # kept for API compatibility; CombatModel does not use it directly
        dtype=np.float32,
    ):
        self.batch_col = batch_col
        self.covariates = covariates or []
        self.cpm_threshold = float(cpm_threshold)
        self.min_frac_samples = float(min_frac_samples)
        self.var_keep_top = int(var_keep_top) if var_keep_top is not None else None
        self.combat_ref_batch = combat_ref_batch
        self.dtype = dtype

        # learned
        self.var_names_ = None
        self.expr_mask_ = None
        self.var_mask_ = None
        self.feature_names_ = None

        # ComBat model (train-fitted)
        self._combat_model = None

    def _low_expr_mask(self, X_train_counts):
        lib = _sum_rows(X_train_counts).astype(np.float64)
        lib[lib == 0] = np.nan

        if sparse.issparse(X_train_counts):
            Xcpm = X_train_counts.multiply(1e6 / lib[:, None])
            frac = (Xcpm >= self.cpm_threshold).mean(axis=0)
            frac = np.asarray(frac).ravel()
        else:
            Xcpm = (X_train_counts / lib[:, None]) * 1e6
            frac = (Xcpm >= self.cpm_threshold).mean(axis=0)

        return frac >= self.min_frac_samples

    def _build_batch_and_cov(self, adata):
        if self.batch_col not in adata.obs.columns:
            raise KeyError(f"batch_col='{self.batch_col}' not found in adata.obs")

        batch_df = pd.DataFrame(
            {"batch": adata.obs[self.batch_col].astype(str).values},
            index=adata.obs_names,
        )

        cov_dfs = []
        for c in self.covariates:
            if c not in adata.obs.columns:
                raise KeyError(f"covariate '{c}' not found in adata.obs")
            cov_dfs.append(pd.DataFrame({c: adata.obs[c].values}, index=adata.obs_names))

        return batch_df, cov_dfs

    def fit(self, adata_train):
        # --- basic checks ---
        if adata_train.n_obs < 2:
            raise ValueError("Need at least 2 samples to fit DCB6Preprocessor.")
        if adata_train.n_vars < 2:
            raise ValueError("Need at least 2 genes to fit DCB6Preprocessor.")

        Xtr_counts = adata_train.X
        self.var_names_ = adata_train.var_names.astype(str)

        # 1) expression mask (TRAIN only)
        self.expr_mask_ = self._low_expr_mask(Xtr_counts)

        if self.expr_mask_.sum() < 10:
            raise ValueError(f"Too few genes pass low-expression filter: {self.expr_mask_.sum()}")

        # 2) logCPM (TRAIN only, after expr filter)
        Xtr_log = logcpm(Xtr_counts[:, self.expr_mask_])  # dense

        # 3) variance mask (TRAIN only)
        v = Xtr_log.var(axis=0, ddof=1)
        if self.var_keep_top is not None and self.var_keep_top < Xtr_log.shape[1]:
            top = np.argsort(v)[::-1][: self.var_keep_top]
            var_mask = np.zeros(Xtr_log.shape[1], dtype=bool)
            var_mask[top] = True
        else:
            var_mask = np.ones(Xtr_log.shape[1], dtype=bool)

        self.var_mask_ = var_mask

        genes_expr = self.var_names_[self.expr_mask_]
        self.feature_names_ = genes_expr[self.var_mask_]

        # 4) Fit ComBat on TRAIN only
        Xtr_sel = Xtr_log[:, self.var_mask_]  # (n_train, n_selected_genes)

        try:
            from neurocombat_sklearn import CombatModel
        except ImportError as e:
            raise ImportError(
                "neurocombat-sklearn is required for inductive (train-only) ComBat.\n"
                "Install with: pip install neurocombat-sklearn"
            ) from e

        _patch_neurocombat_onehot()

        #batch_tr, cov_tr_list = self._build_batch_and_cov(adata_train)

        batch_cat = pd.Categorical(adata_train.obs[self.batch_col].astype(str))
        self._batch_categories_ = list(batch_cat.categories)
        sites_tr = batch_cat.codes.reshape(-1, 1).astype(np.float64)

        # covariates -> numeric arrays (minimal handling)
        cov_tr_list = []
        for c in self.covariates:
            v = pd.to_numeric(adata_train.obs[c], errors="coerce")
            if v.isna().any():
                # if non-numeric, you need to encode it; smallest solution = factor codes
                v = pd.Categorical(adata_train.obs[c].astype(str)).codes.astype(np.float64)
            else:
                v = v.astype(np.float64).values
            cov_tr_list.append(v.reshape(-1, 1))

        self._combat_model = CombatModel()
        _ = self._combat_model.fit_transform(Xtr_sel, sites_tr, *cov_tr_list)
            #self._combat_model = CombatModel()
            #_ = self._combat_model.fit_transform(Xtr_sel, batch_tr, *cov_tr_list)

        return self

    def transform(self, adata):
        if self.expr_mask_ is None or self.var_mask_ is None or self._combat_model is None:
            raise RuntimeError("Call fit() first.")

        # apply train-learned gene masks, then logCPM
        X_counts = adata.X
        X_log = logcpm(X_counts[:, self.expr_mask_])
        X_sel = X_log[:, self.var_mask_]

        batch_cat = pd.Categorical(
            adata.obs[self.batch_col].astype(str),
            categories=self._batch_categories_
        )
        sites = batch_cat.codes.reshape(-1, 1).astype(np.float64)

        # If any unseen batch label appears, fail fast (simplest safe behavior)
        if (batch_cat.codes == -1).any():
            unseen = pd.unique(adata.obs[self.batch_col].astype(str)[batch_cat.codes == -1])
            raise ValueError(f"Unseen batch labels at transform(): {unseen}. "
                            f"Ensure every dataset appears in TRAIN or disable ComBat for LOSO.")

        # Covariates: must be numeric arrays too (minimal handling; matches your fit() approach)
        cov_list = []
        for c in self.covariates:
            v = pd.to_numeric(adata.obs[c], errors="coerce")
            if v.isna().any():
                v = pd.Categorical(adata.obs[c].astype(str)).codes.astype(np.float64)
            else:
                v = v.astype(np.float64).values
            cov_list.append(v.reshape(-1, 1))

        X_h = self._combat_model.transform(X_sel, sites, *cov_list)
        return np.asarray(X_h, dtype=self.dtype)

    def get_feature_names(self):
        return self.feature_names_



class h5adDataReader:
    """
    Loads an .h5ad dataset and returns (X_train, X_cal, X_test, y_train, y_cal, y_test), class_names

    Assumptions:
      - target_column is either already present in adata.obs (e.g., "DCB6"),
        OR you pass a special target_column value to create it:
          target_column="__DCB6__"  (will create from time/event columns you provide)

    Supports:
      - train-only preprocessing: gene filtering + logCPM + ComBat (fit on train only)
      - stratified splits: train / calibration / test

    If you already saved an integrated h5ad with obs["DCB6"], set target_column="DCB6"
    and you DON'T need to provide time/event columns.
    """

    def __init__(
        self,
        # splitting
        test_size=0.20,
        cal_size=0.40,
        random_state=0,

        # preprocessing
        use_raw=False,                 # if True and adata.raw exists, will use adata.raw.X and adata.raw.var_names
        batch_col="dataset",
        covariates=None,               # optional obs columns for ComBat model
        cpm_threshold=1.0,
        min_frac_samples=0.05,
        var_keep_top=6000,
        combat_ref_batch=None):


        self.test_size = float(test_size)
        self.cal_size = float(cal_size)
        self.random_state = int(random_state)

        self.use_raw = bool(use_raw)
        self.batch_col = batch_col
        self.covariates = covariates or []
        self.cpm_threshold = cpm_threshold
        self.min_frac_samples = min_frac_samples
        self.var_keep_top = var_keep_top
        self.combat_ref_batch = combat_ref_batch


        # will be set after load
        self.feature_names_ = None
        self.preprocessor_ = None

    def _ensure_target(self, adata, target_column):
        # If target exists, use it
        if target_column in adata.obs.columns:
            y = adata.obs[target_column]
            return y
        raise KeyError(f"target_column='{target_column}' not found in adata.obs, and no rule to create it.")

    def _get_matrix_view(self, adata):
        if self.use_raw and (adata.raw is not None):
            X = adata.raw.X
            var_names = adata.raw.var_names.astype(str)
        else:
            X = adata.X
            var_names = adata.var_names.astype(str)
        return X, var_names

    def load_data(self, data_path, target_column):
        # read
        adata = sc.read_h5ad(data_path)

        # ensure target column
        y = self._ensure_target(adata, target_column)

        # drop missing labels
        y_num = pd.to_numeric(pd.Series(y), errors="coerce")
        keep = y_num.notna()
        adata = adata[keep.values].copy()
        y_num = y_num.loc[keep].astype(int).values

        print(f"Data shape after dropping missing labels: {adata.shape}")
        print("RANDOM STATE:", self.random_state    )
        # Ensure batch_col exists if you plan ComBat
        if self.batch_col not in adata.obs.columns:
            warnings.warn(f"Batch column '{self.batch_col}' not found in adata.obs. Creating single-batch label to avoid errors.")
            # If missing, create a single-batch label so preprocessing still works
            adata.obs[self.batch_col] = "batch0"

        # If using raw, mirror var/ X into .X so downstream slicing is consistent
        X, var_names = self._get_matrix_view(adata)
        if (self.use_raw and (adata.raw is not None)):
            # rebuild a lightweight AnnData with raw X/var but same obs
            import anndata as ad
            adata = ad.AnnData(X=X, obs=adata.obs.copy(), var=pd.DataFrame(index=var_names))
        else:
            # ensure var_names match what we use
            adata.var_names = var_names

        # class names
        class_names = sorted(list(pd.unique(y_num)))

        # split: train vs temp, then temp -> cal/test
        temp_size = self.test_size + self.cal_size
        if temp_size >= 1.0:
            raise ValueError("test_size + cal_size must be < 1.0")

        splitter1 = StratifiedShuffleSplit(
            n_splits=1, test_size=temp_size, random_state=self.random_state
        )
        train_idx, temp_idx = next(splitter1.split(np.zeros(len(y_num)), y_num))

        y_temp = y_num[temp_idx]
        # relative size of test within temp
        test_frac_within_temp = self.test_size / temp_size

        splitter2 = StratifiedShuffleSplit(
            n_splits=1, test_size=test_frac_within_temp, random_state=self.random_state + 1
        )
        cal_rel_idx, test_rel_idx = next(splitter2.split(np.zeros(len(y_temp)), y_temp))
        cal_idx = temp_idx[cal_rel_idx]
        test_idx = temp_idx[test_rel_idx]

        print("train: ", train_idx[:15])    
        print("cal: ", cal_idx[:15])
        print("test: ", test_idx[:15])    

        # train-only preprocessing (filter + logCPM + ComBat)
        self.preprocessor_ = DCB6Preprocessor(
            batch_col=self.batch_col,
            covariates=self.covariates,
            cpm_threshold=self.cpm_threshold,
            min_frac_samples=self.min_frac_samples,
            var_keep_top=self.var_keep_top,
            combat_ref_batch=self.combat_ref_batch,
        )

        ad_train = adata[train_idx].copy()
        ad_cal   = adata[cal_idx].copy()
        ad_test  = adata[test_idx].copy()

        self.preprocessor_.fit(ad_train)

        X_train = self.preprocessor_.transform(ad_train)
        X_cal   = self.preprocessor_.transform(ad_cal)
        X_test  = self.preprocessor_.transform(ad_test)

        self.feature_names_ = self.preprocessor_.get_feature_names()

        y_train = y_num[train_idx]
        y_cal   = y_num[cal_idx]
        y_test  = y_num[test_idx]

        splits = (X_train, X_cal, X_test, y_train, y_cal, y_test)
        return splits, class_names
