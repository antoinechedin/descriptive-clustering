# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.cluster.hierarchy import linkage, fcluster

import kneedetector


class DesC(BaseEstimator):

    knee_detection_switch = {
        "kneedle": kneedetector.kneedle_scan,
        "l-method": kneedetector.l_method_scan,
        }

    def __init__(self, knee_method="kneedle"):

        if knee_method not in DesC.knee_detection_switch:
            valid_methods = ", ".join(DesC.knee_detection_switch.keys())
            raise ValueError(
                "Unknown method: {}. Valid knee methods are [{}].".format(
                    knee_method, valid_methods
                    )
                )
        self.knee_method = knee_method

    def fit(self, X, y=None):
        self.X_ = check_array(X)

        # Build linkage
        self._z = linkage(X, method="single")
        D = np.vstack((
            np.arange(1, len(self.X_)),
            self._z[:, 2][::-1]
        )).T

        # Add last point with a 0 cost
        self.eval_graph_ = np.vstack((D, np.array([len(D) + 1., 0.])))
        self.knee_ = DesC.knee_detection_switch[self.knee_method]()
        self.K_ = int(self.knee_[0])
        self.labels_ = fcluster(self._z, self.K_, criterion="maxclust")

        # Return the classifier
        return self

    def predict(self, X):
        raise NotImplementedError()
        # Check is fit had been called
        check_is_fitted(self, ["X_", "labels_", ])

        # Input validation
        X = check_array(X)
