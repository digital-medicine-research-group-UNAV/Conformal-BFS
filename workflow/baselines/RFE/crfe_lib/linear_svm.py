"""Naive but efficient linear SVM, structured for future extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


FeatureMap = Callable[[np.ndarray], np.ndarray]


@dataclass
class TrainingHistory:
    epochs_run: int
    converged: bool


class LinearSVM:
    """
    L2-regularized linear SVM trained with batch sub-gradient descent.

    The class is intentionally small and explicit to ease future extensions
    (e.g. one-vs-all wrappers or explicit additive kernel feature maps).
    """

    def __init__(
        self,
        C: float = 1.0,
        lr: float = 0.01,
        epochs: int = 1000,
        tol: float = 1e-4,
        fit_intercept: bool = True,
        feature_map: Optional[FeatureMap] = None,
    ) -> None:
        
        """
        Args:
            C: Soft-margin penalty term.
            lr: Learning rate for gradient steps.
            epochs: Maximum number of passes over the data.
            tol: Stop early when the weight update norm falls below this value.
            fit_intercept: Whether to learn an intercept term.
            feature_map: Optional explicit feature map; useful for future
                additive-kernel style extensions. Defaults to identity.
        """
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.feature_map = feature_map or (lambda x: x)

        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.classes_: Optional[np.ndarray] = None
        self.label_map_: Optional[dict] = None
        self.inverse_label_map_: Optional[dict] = None
        self.history_: Optional[TrainingHistory] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVM":
        X = self._prepare_features(X)
        y_signed = self._prepare_labels(y)

        n_samples, n_features = X.shape
        self.w_ = np.zeros(n_features)
        self.b_ = 0.0

        converged = False
        for epoch in range(self.epochs):
            margins = y_signed * (X @ self.w_ + (self.b_ if self.fit_intercept else 0.0))
            misclassified = margins < 1

            grad_w = self.w_.copy()
            if np.any(misclassified):
                grad_w -= (self.C / n_samples) * (X[misclassified].T @ y_signed[misclassified])

            if self.fit_intercept:
                grad_b = -(self.C / n_samples) * y_signed[misclassified].sum() if np.any(misclassified) else 0.0
            else:
                grad_b = 0.0

            prev_w = self.w_.copy()
            self.w_ -= self.lr * grad_w
            self.b_ -= self.lr * grad_b

            if np.linalg.norm(self.w_ - prev_w) < self.tol:
                converged = True
                break

        self.history_ = TrainingHistory(epochs_run=epoch + 1, converged=converged)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X_t = self._prepare_features(X)
        scores = X_t @ self.w_ + (self.b_ if self.fit_intercept else 0.0)
        return scores

    def predict(self, X: np.ndarray, *, return_signed: bool = False) -> np.ndarray:
        signed = np.sign(self.decision_function(X))
        signed[signed == 0] = 1  # treat exact margin hits as positive
        if return_signed:
            return signed
        return np.vectorize(self.inverse_label_map_.get)(signed)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_true = np.asarray(y).ravel()
        y_pred = self.predict(X)
        return float(np.mean(y_true == y_pred))

    # Internal helpers -----------------------------------------------------
    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")
        return self.feature_map(X_arr)

    def _prepare_labels(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y).ravel()
        classes = np.unique(y_arr)
        if classes.size != 2:
            raise ValueError(f"LinearSVM supports exactly 2 classes, received {classes.size}.")

        ordered = np.sort(classes)
        self.classes_ = ordered
        self.label_map_ = {ordered[0]: -1, ordered[1]: 1}
        self.inverse_label_map_ = {-1: ordered[0], 1: ordered[1]}

        y_signed = np.vectorize(self.label_map_.get)(y_arr)
        try:
            return y_signed.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Unexpected label encountered when normalizing classes.") from exc

    def _check_is_fitted(self) -> None:
        if self.w_ is None or self.label_map_ is None:
            raise RuntimeError("Model has not been fitted yet.")
