import sys
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from ._utils_crfe import binary_change, NC_OvsA_SVMl_dev
from ._conformal_module import CP

def predict_scores_svm(estimator_, classes_, lambda_param, lambda_p_param,
                      X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test):
    """
    Optimized SVM scoring with efficient array operations.
    
    Parameters
    ----------
    estimator_ : sklearn estimator
        Fitted estimator instance
    classes_ : array-like
        Class labels
    lambda_param : float
        Lambda parameter for multiclass
    lambda_p_param : float
        Lambda prime parameter for multiclass
    X_tr, Y_tr : Training data and labels
    X_cal, Y_cal : Calibration data and labels
    X_test, Y_test : Test data and labels
    
    Returns
    -------
    List[float]
        [Empirical_coverage, Uncertainty, Certainty]
    """
    #X_tr_view = X_tr[1:]
    #X_cal_view = X_cal[1:]
    #X_test_view = X_test[1:]

    # Initialize lambda_p_param if not provided
    print(classes_)
    print(len(classes_))
    print(lambda_param)
    if lambda_p_param is None:
        lambda_p_param = (1 - lambda_param) / (len(classes_) - 1) if len(classes_) > 2 else 0

    if len(classes_) == 2:
        estimator = estimator_.fit(X_tr, Y_tr)
        w = estimator.coef_
        bias = estimator.intercept_

        multiclass = False
        NCM_cal = NC_OvsA_SVMl_dev(X_cal, Y_cal, w, bias, multiclass)

        # Vectorized computation for test NCM
        NCM_test = [NC_OvsA_SVMl_dev(
                    np.tile(sample, (len(classes_), 1)),
                    classes_,
                    w, bias,
                    lambda_param,
                    lambda_p_param,
                    multiclass) for sample in X_test]
    else:
        if isinstance(estimator_, OneVsRestClassifier):
            estimator = estimator_.fit(X_tr, Y_tr)
            w = estimator.coef_
            bias = estimator.intercept_
        else:
            estimator = OneVsRestClassifier(estimator_, n_jobs=-1)
            estimator.fit(X_tr, Y_tr)
            
            # Vectorized coefficient extraction
            w = np.array([est.coef_[0] for est in estimator.estimators_])
            bias = np.array([est.intercept_[0] for est in estimator.estimators_])

        multiclass = True
        NCM_cal = NC_OvsA_SVMl_dev(X_cal, Y_cal, w, bias, lambda_param, lambda_p_param, multiclass)

        NCM_test = [NC_OvsA_SVMl_dev(
                    np.tile(sample, (len(classes_), 1)),
                    classes_,
                    w, bias,
                    lambda_param,
                    lambda_p_param,
                    multiclass) for sample in X_test]

    scores = CP(0.10).Conformal_prediction_scores(Y_test, NCM_test, NCM_cal, classes_)
    sys.stdout.flush()

    empirical_coverage = scores[0]
    uncertainty = scores[1]
    certainty = scores[2]
    print(f"Empirical coverage: {empirical_coverage}, Uncertainty: {uncertainty}, Certainty: {certainty}")

    return [empirical_coverage, uncertainty, certainty]