

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from crfe_lib.linear_svm import FeatureMap, LinearSVM


class SVMRFE(LinearSVM):
    """
    Classical recursive feature elimination (RFE) using a LinearSVM
    as the base estimator.

    - Binary case: feature importance is w_j^2.
    - Multi-class case: one-vs-all (OVA) scheme where the importance
      of feature j is sum_k w_{k, j}^2 over classes k.

    Notes
    -----
    - The indices in `selected_features_` refer to the original columns
      of X passed to `select`.
    - The `step` parameter controls how many features are removed at
      each iteration: removing more than one per step can speed things up.
    """

    def __init__(
        self,
        C: float = 1.0,
        lr: float = 0.01,
        epochs: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
        feature_map: Optional[FeatureMap] = None,
        step: Union[int, float] = 1,
    ) -> None:
        
        """
        Args
        ----
        C, lr, epochs, tol, fit_intercept, feature_map:
            Same as in LinearSVM.

        step:
            If int: number of features to remove at each iteration.
                    Must be >= 1.
            If float: fraction of remaining features to remove at each
                      iteration. Must be in (0, 1).
        """
        super().__init__(C, lr, epochs, tol, fit_intercept, feature_map)
        self.step: Union[int, float] = step

        # RFE bookkeeping
        self.selected_features_: Optional[np.ndarray] = None  # (n_selected,)
        self.ranking_: Optional[np.ndarray] = None            # (n_total_features,)
        self.scores_: Optional[np.ndarray] = None             # (n_total_features,)

        # For multi-class (OVA) diagnostics
        self.W_: Optional[np.ndarray] = None      # (n_classes, n_features_current)
        self.b_vec_: Optional[np.ndarray] = None  # (n_classes,)

    # ------------------------------------------------------------------
    # Internal helpers
    def _compute_step_size(self, n_current, step_deletion) -> int:
        
        if isinstance(step_deletion, int):
            n_remove = step_deletion
            return int(max(1, n_remove))
        
        elif isinstance(step_deletion, float):
            n_remove = int(np.ceil(step_deletion * n_current)) 
            return int(max(1, n_remove))
        

        else:
            raise ValueError("step must be int or float.")
        
        

    def _binary_rfe_scores(self) -> np.ndarray:
    
        if self.w_ is None:
            raise RuntimeError("Model must be fitted before computing scores.")
        return self.w_ ** 2

    def _ova_rfe_scores(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """
        Train one-vs-all LinearSVMs and compute RFE scores as the sum
        of squared weights across classes.

        Parameters
        ----------
        X : (n_samples, n_features_current)
        y : (n_samples,)
        classes : (n_classes,)

        Returns
        -------
        scores : (n_features_current,)
        """
        n_classes = classes.size
        n_features = X.shape[1]

        W = np.zeros((n_classes, n_features), dtype=float)
        b_vec = np.zeros(n_classes, dtype=float)

        for k, cls in enumerate(classes):
            # One-vs-all targets in {-1, +1}
            y_bin = np.where(y == cls, 1.0, -1.0)
            super().fit(X, y_bin)
            W[k, :] = self.w_
            b_vec[k] = self.b_

        self.W_ = W
        self.b_vec_ = b_vec

        # Classical multi-class extension: sum of squared weights per feature
        return np.sum(W ** 2, axis=0)

   
    def select(self, X: np.ndarray, y: np.ndarray, n_final_features: int, step_deletion = 1) -> np.ndarray:
        
        """
        Run recursive feature elimination and return the indices of the
        selected features (w.r.t. the original input columns).

        Returns
        -------
        selected_features_ : np.ndarray of shape (n_features,)
            Indices of the selected features.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        n_samples, n_total_features = X.shape

        if n_final_features <= 0:
            raise ValueError("n_features must be strictly positive.")

        if n_final_features >= n_total_features:
            # Nothing to eliminate
            self.selected_features_ = np.arange(n_total_features)
            self.ranking_ = np.ones(n_total_features, dtype=int)
            self.scores_ = np.zeros(n_total_features, dtype=float)
            # For completeness
            self.classes_ = np.unique(y)
            return self.selected_features_

        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("At least two classes are required for RFE.")

        # Track which original columns are still in play
        selected = np.arange(n_total_features)
        ranking = np.ones(n_total_features, dtype=int)

        while selected.size > n_final_features:
            X_sub = X[:, selected]
            n_current = X_sub.shape[1]

            if classes.size == 2:
                # Binary case: fit once and use w_j^2
                super().fit(X_sub, y)
                scores = self._binary_rfe_scores()
            else:
                # Multi-class: OVA with sum_k w_{k,j}^2
                scores = self._ova_rfe_scores(X_sub, y, classes)

            #print(scores)

            # Least important features have the smallest scores
            n_remove = self._compute_step_size(n_current, step_deletion)
            #print(f"Removing {n_remove} features...")
            remove_local_idx = np.argsort(scores)[:n_remove]

            
            remove_global_idx = selected[remove_local_idx]

            # Update ranking of removed features
            next_rank = ranking.max() + 1
            ranking[remove_global_idx] = next_rank

            # Keep the remaining features
            mask_keep = np.ones(n_current, dtype=bool)
            mask_keep[remove_local_idx] = False
            selected = selected[mask_keep]

            print(selected)
            

        
        X_final = X[:, selected]
        if classes.size == 2:
            super().fit(X_final, y)
            scores_final = self._binary_rfe_scores()
        else:
            scores_final = self._ova_rfe_scores(X_final, y, classes)

        self.selected_features_ = selected
        self.ranking_ = ranking

        self.scores_ = np.zeros(n_total_features, dtype=float)
        self.scores_[selected] = scores_final

        self.classes_ = classes

        return self.selected_features_

