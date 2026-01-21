



import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit
import os 
import pickle
import json
import warnings

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



def conformal_prediction_svm_ova(X_train, y_train, X_cal, y_cal, X_test, y_test, class_names, lambda_param=0.1, lambda_p_param=None):
    
    from sklearn.svm import LinearSVC
    from sklearn.multiclass import OneVsRestClassifier

    # Train One-vs-Rest SVM
    estimator_= LinearSVC(
        tol=1e-4, 
        loss='squared_hinge',
        max_iter=14000,
        dual="auto",
        C=1.0
    )

    if len(class_names) > 2:
        # For binary classification, use SVC directly
        
        estimator_ = ovr_classifier = OneVsRestClassifier(estimator_)
    
    

    # Predict scores using the optimized function
    from pathlib import Path
    import sys

    pkg_path = Path(__file__).resolve().parents[3] / "package"  # .../conformal-ffs/package
    if str(pkg_path) not in sys.path:
        sys.path.insert(0, str(pkg_path))

    from _utils import predict_scores_svm
    


    results = predict_scores_svm(
        estimator_=estimator_,
        classes_=class_names,
        lambda_param=lambda_param,
        lambda_p_param=lambda_p_param,
        X_tr=X_train,
        Y_tr=y_train,
        X_cal=X_cal,
        Y_cal=y_cal,
        X_test=X_test,
        Y_test=y_test
    )

    return results

  


data_path = data_path = "data/imvigor/integrated_DCB6.h5ad"
target_column =  "DCB6" #"target" #"recist"  # Specify the target column for real datasets

n_experiments = 20
save_results = True  # Set to True to save results to files
results_dir = "results"  # Directory to save

all_results_crfe = {}
all_results_rfe = {}
all_results_lasso = {}
all_results_enet = {}
for run_id in range(n_experiments):

    print(f"Running experiment {run_id+1}/{n_experiments} with run_id={run_id}")
    # Run the experiment    

    data_reader = h5adDataReader(test_size=0.20,cal_size=0.30, random_state=run_id, var_keep_top=3500)
    splits, class_names = data_reader.load_data(data_path, target_column)

    X_train, X_cal, X_test, y_train, y_cal, y_test = splits


    from pathlib import Path
    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    #PROJECT_ROOT = "Floating/conformal-ffs/../../CRFE/CRFE-lib" # Path.cwd()
    
    from crfe_lib.crfe import CRFE  # noqa: E402
    from crfe_lib.rfe import SVMRFE
    from LassoElasticnet import LassoSelector, ElasticNetSelector
    
    crfe = CRFE(C=1.0, lr=0.01, epochs=1000, tol=1e-4)
    selected_crfe = crfe.select(X_train, y_train, n_final_features= 10, step_deletion = .1)

    rfe = SVMRFE(C=1.0, lr=0.01, epochs=1000, tol=1e-4, step=0.1)
    selected_rfe = rfe.select(X_train, y_train, n_final_features= 10, step_deletion = .1)

    lasso = LassoSelector(C=1.0, max_iter=5000, tol=1e-4)
    selected_lasso = lasso.select(X_train, y_train, n_final_features=10, step_deletion=.1)

    enet = ElasticNetSelector(C=1.0, l1_ratio=0.5, max_iter=5000, tol=1e-4)
    selected_enet = enet.select(X_train, y_train, n_final_features=10, step_deletion=.1)

    

    results_svm_ova_crfe = conformal_prediction_svm_ova(
        X_train[:, crfe.selected_features.tolist()],
        y_train,
        X_cal[:, crfe.selected_features.tolist()],
        y_cal,
        X_test[:, crfe.selected_features.tolist()],
        y_test,
        class_names,
        lambda_param=0.1,
        lambda_p_param=None
    ) 

    results_svm_ova_rfe = conformal_prediction_svm_ova(
        X_train[:, selected_rfe.tolist()],
        y_train,
        X_cal[:, selected_rfe.tolist()],
        y_cal,
        X_test[:, selected_rfe.tolist()],
        y_test,
        class_names,
        lambda_param=0.1,
        lambda_p_param=None
    )


    results_svm_ova_lasso = conformal_prediction_svm_ova(
        X_train[:, selected_lasso.tolist()],
        y_train,
        X_cal[:, selected_lasso.tolist()],
        y_cal,
        X_test[:, selected_lasso.tolist()],
        y_test,
        class_names,
        lambda_param=0.1,
        lambda_p_param=None
    )

    results_svm_ova_enet = conformal_prediction_svm_ova(
        X_train[:, selected_enet.tolist()],
        y_train,
        X_cal[:, selected_enet.tolist()],
        y_cal,
        X_test[:, selected_enet.tolist()],
        y_test,
        class_names,
        lambda_param=0.1,
        lambda_p_param=None
    )

    coverage_crfe = results_svm_ova_crfe[0]
    uncertainty_crfe = results_svm_ova_crfe[1]
    certainty_crfe = results_svm_ova_crfe[2]
    print(f"CRFE Empirical coverage: {coverage_crfe}")
    print(f"CRFE Uncertainty: {uncertainty_crfe}")
    print(f"CRFE Certainty: {certainty_crfe}")

    coverage_rfe = results_svm_ova_rfe[0]
    uncertainty_rfe = results_svm_ova_rfe[1]
    certainty_rfe = results_svm_ova_rfe[2]
    print(f"RFE Empirical coverage: {coverage_rfe}")
    print(f"RFE Uncertainty: {uncertainty_rfe}")
    print(f"RFE Certainty: {certainty_rfe}")

    coverage_lasso = results_svm_ova_lasso[0]
    uncertainty_lasso = results_svm_ova_lasso[1]
    certainty_lasso = results_svm_ova_lasso[2]
    print(f"Lasso Empirical coverage: {coverage_lasso}")
    print(f"Lasso Uncertainty: {uncertainty_lasso}")
    print(f"Lasso Certainty: {certainty_lasso}")

    coverage_enet = results_svm_ova_enet[0]
    uncertainty_enet = results_svm_ova_enet[1]
    certainty_enet = results_svm_ova_enet[2]
    print(f"Elastic Net Empirical coverage: {coverage_enet}")
    print(f"Elastic Net Uncertainty: {uncertainty_enet}")
    print(f"Elastic Net Certainty: {certainty_enet}")


    all_results_crfe[run_id + 1] = {"selected_features": crfe.selected_features.tolist(), "run_id": run_id + 1, "empirical_coverage": coverage_crfe,
                    "uncertainty": uncertainty_crfe, "certainty": certainty_crfe}
    all_results_rfe[run_id + 1] = {"selected_features": selected_rfe.tolist(), "run_id": run_id + 1, "empirical_coverage": coverage_rfe,
                    "uncertainty": uncertainty_rfe, "certainty": certainty_rfe}
    all_results_lasso[run_id + 1] = {"selected_features": selected_lasso.tolist(), "run_id": run_id + 1, "empirical_coverage": coverage_lasso,
                    "uncertainty": uncertainty_lasso, "certainty": certainty_lasso}
    all_results_enet[run_id + 1] = {"selected_features": selected_enet.tolist(), "run_id": run_id + 1, "empirical_coverage": coverage_enet,
                    "uncertainty": uncertainty_enet, "certainty": certainty_enet}

    print(all_results_crfe)
    print(all_results_rfe)
    print(all_results_lasso)
    print(all_results_enet)

if save_results:
    print("Saving results to files...")
    results_dir_CRFE = os.path.join(results_dir, "Imvigor_CRFE_Results")
    os.makedirs(results_dir_CRFE, exist_ok=True)
    results_dir_RFE =  os.path.join(results_dir, "Imvigor_RFE_Results")
    os.makedirs(results_dir_RFE, exist_ok=True)
    results_dir_Lasso =  os.path.join(results_dir, "Imvigor_Lasso_Results")
    os.makedirs(results_dir_Lasso, exist_ok=True)
    results_dir_Enet =  os.path.join(results_dir, "Imvigor_Enet_Results")
    os.makedirs(results_dir_Enet, exist_ok=True)

    # Save as JSON (human-readable)
    json_file_crfe = os.path.join(results_dir_CRFE, f"crfe_results_imvigor_{n_experiments}.json")
    try:
        with open(json_file_crfe, 'w') as f:
            json.dump(all_results_crfe, f, indent=2, default=str)
        print(f"Results saved to JSON: {json_file_crfe}")
    except Exception as e:
        print(f"Error saving JSON: {e}")

    # Save as pickle (preserves Python objects)
    pickle_file = os.path.join(results_dir_CRFE, f"crfe_results_imvigor_{n_experiments}.pkl")
    try:
        with open(pickle_file, 'wb') as f:
            pickle.dump(all_results_crfe, f)
        print(f"Results saved to pickle: {pickle_file}")
    except Exception as e:
        print(f"Error saving pickle: {e}")



    # Save as JSON (human-readable)
    json_file_rfe = os.path.join(results_dir_RFE, f"rfe_results_imvigor_{n_experiments}.json")
    try:
        with open(json_file_rfe, 'w') as f:
            json.dump(all_results_rfe, f, indent=2, default=str)
        print(f"Results saved to JSON: {json_file_rfe}")
    except Exception as e:
        print(f"Error saving JSON: {e}")

    # Save as pickle (preserves Python objects)
    pickle_file = os.path.join(results_dir_RFE, f"rfe_results_imvigor_{n_experiments}.pkl")
    try:
        with open(pickle_file, 'wb') as f:
            pickle.dump(all_results_rfe, f)
        print(f"Results saved to pickle: {pickle_file}")
    except Exception as e:
        print(f"Error saving pickle: {e}")

    # Save as JSON (human-readable)
    json_file_lasso = os.path.join(results_dir_Lasso, f"lasso_results_imvigor_{n_experiments}.json")
    try:
        with open(json_file_lasso, 'w') as f:
            json.dump(all_results_lasso, f, indent=2, default=str)
        print(f"Results saved to JSON: {json_file_lasso}")
    except Exception as e:
        print(f"Error saving JSON: {e}")

    # Save as pickle (preserves Python objects)
    pickle_file = os.path.join(results_dir_Lasso, f"lasso_results_imvigor_{n_experiments}.pkl")
    try:
        with open(pickle_file, 'wb') as f:
            pickle.dump(all_results_lasso, f)
        print(f"Results saved to pickle: {pickle_file}")
    except Exception as e:
        print(f"Error saving pickle: {e}")

    
    # Save as JSON (human-readable)
    json_file_enet = os.path.join(results_dir_Enet, f"enet_results_imvigor_{n_experiments}.json")
    try:
        with open(json_file_enet, 'w') as f:
            json.dump(all_results_enet, f, indent=2, default=str)
        print(f"Results saved to JSON: {json_file_enet}")
    except Exception as e:
        print(f"Error saving JSON: {e}")    
    
    # Save as pickle (preserves Python objects)
    pickle_file = os.path.join(results_dir_Enet, f"enet_results_imv igor_{n_experiments}.pkl")
    try:
        with open(pickle_file, 'wb') as f:
            pickle.dump(all_results_enet, f)
        print(f"Results saved to pickle: {pickle_file}")
    except Exception as e:
        print(f"Error saving pickle: {e}")


    print("Results successfully saved!")

    




def analyze_ffs_results(json_file):
    from collections import Counter

    # Fix path issues for json_file
    json_file = os.path.normpath(json_file)
    print(f"Using JSON file: {json_file}")

    with open(json_file, "r") as f:
        ffs_results = json.load(f)
    # Extract all selected features from all runs
    all_features = []

    for run_data in ffs_results.values():
        # Parse the string representation of the array
        feature_str = run_data['selected_features']
        
        # Remove brackets and split by whitespace
        #features = feature_str.strip([]).split()
        # Convert to integers
        features = [int(f) for f in feature_str]
        
        all_features.extend(features)

    # Count frequency of each feature
    feature_counts = Counter(all_features)

    # Get top k most frequent features
    k = 10  # You can change this value
    top_k_features = [feature for feature, count in feature_counts.most_common(k)]

    #print(f"Top {k} most frequent features:")
    #print(top_k_features)

    return top_k_features


top_k_features_crfe = analyze_ffs_results(json_file_crfe)
top_k_features_rfe = analyze_ffs_results(json_file_rfe)
top_k_features_lasso = analyze_ffs_results(json_file_lasso)
top_k_features_enet = analyze_ffs_results(json_file_enet)

print("Top features from CRFE:")
print(top_k_features_crfe)
print("Top features from RFE:")
print(top_k_features_rfe)
print("Top features from Lasso:")
print(top_k_features_lasso)
print("Top features from Elastic Net:")
print(top_k_features_enet)


results_svm_ova_crfe = conformal_prediction_svm_ova(
    X_train[:, top_k_features_crfe],
    y_train,
    X_cal[:, top_k_features_crfe],
    y_cal,
    X_test[:, top_k_features_crfe],
    y_test,
    class_names,
    lambda_param=0.1,
    lambda_p_param=None
)   

results_svm_ova_rfe = conformal_prediction_svm_ova(
    X_train[:, top_k_features_rfe],
    y_train,
    X_cal[:, top_k_features_rfe],
    y_cal,
    X_test[:, top_k_features_rfe],
    y_test,
    class_names,
    lambda_param=0.1,
    lambda_p_param=None
) 


results_svm_ova_lasso = conformal_prediction_svm_ova(
    X_train[:, top_k_features_lasso],
    y_train,
    X_cal[:, top_k_features_lasso],
    y_cal,
    X_test[:, top_k_features_lasso],
    y_test,
    class_names,
    lambda_param=0.1,
    lambda_p_param=None
)

results_svm_ova_enet = conformal_prediction_svm_ova(
    X_train[:, top_k_features_enet],
    y_train,        
    X_cal[:, top_k_features_enet],
    y_cal,
    X_test[:, top_k_features_enet],
    y_test,
    class_names,
    lambda_param=0.1,
    lambda_p_param=None
)

# Save results for CRFE and RFE conformal prediction
if save_results:
    print("Saving conformal prediction results...")
    
    # Save CRFE conformal prediction results (use already created folder)
    # Save as JSON
    json_file_cp_crfe = os.path.join(results_dir_CRFE, f"crfe_cp_results_imvigor_{n_experiments}.json")
    try:
        with open(json_file_cp_crfe, 'w') as f:
            json.dump(results_svm_ova_crfe, f, indent=2, default=str)
        print(f"CRFE CP results saved to JSON: {json_file_cp_crfe}")
    except Exception as e:
        print(f"Error saving CRFE CP JSON: {e}")
    
    # Save as pickle
    pickle_file_cp_crfe = os.path.join(results_dir_CRFE, f"crfe_cp_results_imvigor_{n_experiments}.pkl")
    try:
        with open(pickle_file_cp_crfe, 'wb') as f:
            pickle.dump(results_svm_ova_crfe, f)
        print(f"CRFE CP results saved to pickle: {pickle_file_cp_crfe}")
    except Exception as e:
        print(f"Error saving CRFE CP pickle: {e}")
    
    # Save RFE conformal prediction results (use already created folder)
    # Save as JSON
    json_file_cp_rfe = os.path.join(results_dir_RFE, f"rfe_cp_results_imvigor_{n_experiments}.json")
    try:
        with open(json_file_cp_rfe, 'w') as f:
            json.dump(results_svm_ova_rfe, f, indent=2, default=str)
        print(f"RFE CP results saved to JSON: {json_file_cp_rfe}")
    except Exception as e:
        print(f"Error saving RFE CP JSON: {e}")
    
    # Save as pickle
    pickle_file_cp_rfe = os.path.join(results_dir_RFE, f"rfe_cp_results_imvigor_{n_experiments}.pkl")
    try:
        with open(pickle_file_cp_rfe, 'wb') as f:
            pickle.dump(results_svm_ova_rfe, f)
        print(f"RFE CP results saved to pickle: {pickle_file_cp_rfe}")
    except Exception as e:
        print(f"Error saving RFE CP pickle: {e}")

    
    json_file_cp_lasso = os.path.join(results_dir_Lasso, f"lasso_cp_results_imvigor_{n_experiments}.json")
    try:
        with open(json_file_cp_lasso, 'w') as f:
            json.dump(results_svm_ova_lasso, f, indent=2, default=str)
        print(f"Lasso CP results saved to JSON: {json_file_cp_lasso}")
    except Exception as e:
        print(f"Error saving Lasso CP JSON: {e}")

    pickle_file_cp_lasso = os.path.join(results_dir_Lasso, f"lasso_cp_results_imvigor_{n_experiments}.pkl")
    try:
        with open(pickle_file_cp_lasso, 'wb') as f:
            pickle.dump(results_svm_ova_lasso, f)
        print(f"Lasso CP results saved to pickle: {pickle_file_cp_lasso}")
    except Exception as e:
        print(f"Error saving Lasso CP pickle: {e}")
    

    json_file_cp_enet = os.path.join(results_dir_Enet, f"enet_cp_results_imvigor_{n_experiments}.json")
    try:
        with open(json_file_cp_enet, 'w') as f:
            json.dump(results_svm_ova_enet, f, indent=2, default=str)
        print(f"Elastic Net CP results saved to JSON: {json_file_cp_enet}")
    except Exception as e:
        print(f"Error saving Elastic Net CP JSON: {e}")
    
    pickle_file_cp_enet = os.path.join(results_dir_Enet, f"enet_cp_results_imvigor_{n_experiments}.pkl")
    try:
        with open(pickle_file_cp_enet, 'wb') as f:
            pickle.dump(results_svm_ova_enet, f)
        print(f"Elastic Net CP results saved to pickle: {pickle_file_cp_enet}")
    except Exception as e:
        print(f"Error saving Elastic Net CP pickle: {e}")
    
    
    # Save top features in base results directory
    top_features_dict = {
        "crfe_top_features": top_k_features_crfe,
        "rfe_top_features": top_k_features_rfe,
        "lasso_top_features": top_k_features_lasso,
        "enet_top_features": top_k_features_enet
    }
    
    top_features_file = os.path.join(results_dir, f"top_features_imvigor_{n_experiments}.json")
    try:
        with open(top_features_file, 'w') as f:
            json.dump(top_features_dict, f, indent=2)
        print(f"Top features saved to: {top_features_file}")
    except Exception as e:
        print(f"Error saving top features: {e}")
    
    print("All conformal prediction results successfully saved!")
