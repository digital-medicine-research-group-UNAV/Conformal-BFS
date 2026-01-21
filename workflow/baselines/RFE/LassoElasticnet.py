

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


class LassoSelector:
    """
    L1-penalized logistic regression feature selector (LASSO).
    Usage:
        lasso = LassoSelector(C=1.0, max_iter=5000, class_weight="balanced")
        selected = lasso.select(X_train, y_train, n_final_features=10, step_deletion=0.1)
    Notes:
      - step_deletion is accepted for API compatibility but not used (L1 selects via sparsity).
      - If fewer than n_final_features are non-zero, it will backfill with largest |coef|.
    """
    def __init__(self, C=1.0, max_iter=5000, class_weight="balanced", random_state=0, tol=1e-4):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.random_state = int(random_state)
        self.tol = float(tol)
        self.model_ = None
        self.coef_ = None

    def select(self, X_train, y_train, n_final_features=10, step_deletion=0.1):
        X = np.asarray(X_train)
        y = np.asarray(y_train).astype(int)

        clf = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=self.C,
            max_iter=self.max_iter,
            tol=self.tol,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )
        clf.fit(X, y)
        w = clf.coef_.ravel()
        self.model_ = clf
        self.coef_ = w

        # Prefer non-zero coefficients
        nz = np.flatnonzero(np.abs(w) > 1e-12)
        if nz.size >= n_final_features:
            chosen = nz[np.argsort(np.abs(w[nz]))[::-1][:n_final_features]]
        else:
            # backfill by absolute magnitude
            chosen = np.argsort(np.abs(w))[::-1][:n_final_features]

        return np.array(sorted(chosen))


class ElasticNetSelector:
    """
    Elastic-net logistic regression feature selector.
    Usage:
        enet = ElasticNetSelector(C=1.0, l1_ratio=0.5, max_iter=5000)
        selected = enet.select(X_train, y_train, n_final_features=10, step_deletion=0.1)

    Notes:
      - step_deletion is accepted for API compatibility but not used.
      - If fewer than n_final_features are non-zero, it backfills with largest |coef|.
    """
    def __init__(self, C=1.0, l1_ratio=0.5, max_iter=5000, class_weight="balanced", random_state=0, tol=1e-4):
        self.C = float(C)
        self.l1_ratio = float(l1_ratio)
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.random_state = int(random_state)
        self.tol = float(tol)
        self.model_ = None
        self.coef_ = None

    def select(self, X_train, y_train, n_final_features=10, step_deletion=0.1):
        X = np.asarray(X_train)
        y = np.asarray(y_train).astype(int)

        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=self.l1_ratio,
            C=self.C,
            max_iter=self.max_iter,
            tol=self.tol,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )
        clf.fit(X, y)
        w = clf.coef_.ravel()
        self.model_ = clf
        self.coef_ = w

        nz = np.flatnonzero(np.abs(w) > 1e-12)
        if nz.size >= n_final_features:
            chosen = nz[np.argsort(np.abs(w[nz]))[::-1][:n_final_features]]
        else:
            chosen = np.argsort(np.abs(w))[::-1][:n_final_features]

        return np.array(sorted(chosen))


# -------------------------
# Optional: CV-tuned variants (drop-in)
# -------------------------
class ElasticNetSelectorCV:
    """
    Elastic-net selector with inner CV over C (and optionally l1_ratio).
    Keeps the same select() API.
    """
    def __init__(
        self,
        Cs=(0.01, 0.1, 1.0, 10.0),
        l1_ratios=(0.2, 0.5, 0.8),
        cv_splits=5,
        max_iter=5000,
        class_weight="balanced",
        random_state=0,
        tol=1e-4,
        scoring="balanced_accuracy",
    ):
        self.Cs = list(Cs)
        self.l1_ratios = list(l1_ratios)
        self.cv_splits = int(cv_splits)
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.random_state = int(random_state)
        self.tol = float(tol)
        self.scoring = scoring

        self.best_params_ = None
        self.model_ = None
        self.coef_ = None

    def _score(self, y_true, y_pred_proba):
        # Minimal: balanced accuracy via 0.5 threshold
        y_pred = (y_pred_proba >= 0.5).astype(int)
        # balanced accuracy = (TPR + TNR)/2
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tpr = tp / (tp + fn + 1e-12)
        tnr = tn / (tn + fp + 1e-12)
        return 0.5 * (tpr + tnr)

    def select(self, X_train, y_train, n_final_features=10, step_deletion=0.1):
        X = np.asarray(X_train)
        y = np.asarray(y_train).astype(int)

        skf = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)

        best = (-np.inf, None)
        for C in self.Cs:
            for l1r in self.l1_ratios:
                scores = []
                for tr_idx, va_idx in skf.split(X, y):
                    clf = LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=float(l1r),
                        C=float(C),
                        max_iter=self.max_iter,
                        tol=self.tol,
                        class_weight=self.class_weight,
                        random_state=self.random_state,
                        n_jobs=-1,
                    )
                    clf.fit(X[tr_idx], y[tr_idx])
                    proba = clf.predict_proba(X[va_idx])[:, 1]
                    scores.append(self._score(y[va_idx], proba))
                mean_score = float(np.mean(scores))
                if mean_score > best[0]:
                    best = (mean_score, (float(C), float(l1r)))

        _, (bestC, bestL1) = best
        self.best_params_ = {"C": bestC, "l1_ratio": bestL1}

        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=bestL1,
            C=bestC,
            max_iter=self.max_iter,
            tol=self.tol,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
        )
        clf.fit(X, y)
        w = clf.coef_.ravel()
        self.model_ = clf
        self.coef_ = w

        nz = np.flatnonzero(np.abs(w) > 1e-12)
        if nz.size >= n_final_features:
            chosen = nz[np.argsort(np.abs(w[nz]))[::-1][:n_final_features]]
        else:
            chosen = np.argsort(np.abs(w))[::-1][:n_final_features]

        return np.array(sorted(chosen))


# -------------------------
# Example usage (same style as your pipeline)
# -------------------------
# from crfe_lib.crfe import CRFE
# from crfe_lib.rfe import SVMRFE
#
# crfe = CRFE(C=1.0, lr=0.01, epochs=1000, tol=1e-4)
# selected_crfe = crfe.select(X_train, y_train, n_final_features=10, step_deletion=.1)
#
# rfe = SVMRFE(C=1.0, lr=0.01, epochs=1000, tol=1e-4, step=0.1)
# selected_rfe = rfe.select(X_train, y_train, n_final_features=10, step_deletion=.1)



# Optional CV-tuned elastic net:
# enet_cv = ElasticNetSelectorCV(Cs=(0.01,0.1,1,10), l1_ratios=(0.2,0.5,0.8))
# selected_enet_cv = enet_cv.select(X_train, y_train, n_final_features=10, step_deletion=.1)
