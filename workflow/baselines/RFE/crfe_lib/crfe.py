



from typing import Optional
import numpy as np

from crfe_lib.linear_svm import FeatureMap, LinearSVM

class CRFE(LinearSVM):
    """
    CRFE model extending LinearSVM with custom behavior.
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
        super().__init__(C, lr, epochs, tol, fit_intercept, feature_map)

        self.selected_features: Optional[np.ndarray] = None  # shape (n_selected_features,)
        self.classes_: Optional[np.ndarray] = None   # shape (n_classes,)
        self.W_: Optional[np.ndarray] = None         # shape (n_classes, n_features)
        self.b_vec_: Optional[np.ndarray] = None     # shape (n_classes,)
        self.beta_scores_: Optional[np.ndarray] = None  # shape (n_features,)


    def _compute_step_size(self, n_current, step_deletion) -> int:
        
        if isinstance(step_deletion, int):
            n_remove = step_deletion
            return int(max(1, n_remove))
        
        elif isinstance(step_deletion, float):
            n_remove = int(np.ceil(step_deletion * n_current)) 
            return int(max(1, n_remove))
        

        else:
            raise ValueError("step must be int or float.")

    def compute_binary_nc_scores(self, X: np.ndarray, y:np.ndarray) -> np.ndarray:

        beta_scores = -self.w_ * (y @ X)
        print(self.w_)
        print(beta_scores)
        return beta_scores



    def compute_OVA_nc_scores(self, X: np.ndarray, y:np.ndarray) -> np.ndarray:
        
        lambda_param = 1.0 / (2.0 * self.C)
        lambda_p_param = 1.0 / (2.0 * self.C * (len(self.classes_) - 1))

        weights_sum = np.sum(self.W_, axis=0)

        beta_scores = np.zeros(X.shape[1], dtype=np.float64)
        
        for j in range(X.shape[1]):
            beta_j = 0.0

            for i in range(y.shape[0]):

                y_i = y[i]
                x_ij = X[i, j]
                
                lambda_term = lambda_param * self.W_[y_i, j] * x_ij
                sum_term = lambda_p_param * (weights_sum[j] -  self.W_[y_i, j]) * x_ij

                beta_j -= (lambda_term - sum_term)

            beta_scores[j] = beta_j
        
        return beta_scores




    def select(self, X: np.ndarray, y: np.ndarray, n_final_features: int, step_deletion = 1) -> np.ndarray:

        n_cols = X.shape[1]
        if n_final_features > n_cols:
            self.selected_features = np.arange(n_cols)
            return self.selected_features


        classes = np.unique(y)
        self.classes_ = classes
        n_classes = classes.size

        
        y_encoded = np.vectorize({cls: idx for idx, cls in enumerate(classes)}.get)(y)
        y = y_encoded

        self.selected_features = np.arange(n_cols) # tracker
        data_X = X.copy()

        if n_classes <= 2:
            if n_classes == 2:

                pos_class = classes[1]  # arbitrary choice: last as positive
                y_bin = np.where(y == pos_class, 1.0, -1.0)
                y = y_bin

                while data_X.shape[1] > n_final_features:

                    super().fit(data_X, y)
                    beta_scores = self.compute_binary_nc_scores(data_X, y)

                    n_remove = self._compute_step_size(data_X.shape[1], step_deletion)
                    print(f"Removing {n_remove} features...")
                    #idx_to_remove = np.argmax(beta_scores) # deprecated
                    idx_to_remove = np.argsort(beta_scores)[-n_remove:]
                    idx_to_remove = np.sort(idx_to_remove)

                    data_X = np.delete(data_X, idx_to_remove, axis=1)
                    self.selected_features = np.delete(self.selected_features, idx_to_remove) # update tracker with the remaining features
                    
                    print(f"Features remaining: {data_X.shape[1]}")
                    #print(f"Beta scores: {beta_scores}")
                    print(self.selected_features)

                self.beta_scores_ = beta_scores
                return None

            else:
                raise ValueError("At least two classes are required for feature selection.")

        else:

            while data_X.shape[1] > n_final_features:

                self.W_ = np.zeros((n_classes, data_X.shape[1]), dtype=float)
                self.b_vec_ = np.zeros(n_classes, dtype=float)

                for k, cls in enumerate(classes):

                    y_bin = np.where(y == cls, 1.0, -1.0) # one-vs-all labels

                    # train a binary LinearSVM 
                    super().fit(data_X, y_bin)

                    # store the learned weights
                    self.W_[k, :] = self.w_
                    self.b_vec_[k] = self.b_
        
                beta_scores = self.compute_OVA_nc_scores(data_X, y)

                #idx_to_remove = np.argmax(beta_scores) # deprecated
                n_remove = self._compute_step_size(data_X.shape[1], step_deletion)
                idx_to_remove = np.argsort(beta_scores)[-n_remove:]
                idx_to_remove = np.sort(idx_to_remove)

                data_X = np.delete(data_X, idx_to_remove, axis=1)
                self.selected_features = np.delete(self.selected_features, idx_to_remove) # update tracker with the remaining features
                print(f"Features remaining: {data_X.shape[1]}")
                #print(f"Beta scores: {beta_scores}")
                print(self.selected_features)

            self.beta_scores_ = beta_scores
            return None

        # Call the parent class's fit method
        



