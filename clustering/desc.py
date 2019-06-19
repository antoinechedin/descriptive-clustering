# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from scipy.cluster.hierarchy import linkage, fcluster

from . import kneedetector


class DesC(BaseEstimator):
    """Automatic descriptive clustering model

    Parameters
    ----------
    knee_method : string
        Knee detection method to use. Valid method are: kneedle, l-method and
        amplitude.

    Attributes
    ----------
    eval_graph_ : ndarray, shape: (n_samples, 2)
        Evaluation graph of the fitted data set.
    knee_ : ndarray, shape: (n_features)
        Evaluation graph knee point.
    K_ : int
        Number of cluster.
    labels_ : ndarray, shape: (n_samples)
        Labels of each point.
    plots_ : tuple of ndarray
        Tuple of plots for knee detection graphical explanations

    Raise
    -----
    ValueError
        If knee_method is unknown

    Examples
    --------
    >>> from clustering import DesC
    >>> import numpy as np
    >>> X = np.array([[1, 3], [2, 3], [1, 0], [5, 2], [5, 3]])
    >>> desc = DesC().fit(X)
    >>> desc.K_
    3
    >>> desc.eval_graph_
    array([[1., 3.],
           [2., 3.],
           [3., 1.],
           [4., 1.],
           [5., 0.]])
    >>> desc.labels_
    array([2, 2, 3, 1, 1], dtype=int32)


    """

    knee_detection_switch = {
        "kneedle": kneedetector.kneedle_scan,
        "l-method": kneedetector.l_method_scan,
        "amplitude": kneedetector.max_amplitude_scan,
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
        """Compute DesC clustering

        Parameters
        ----------
        X : array-like, shape: (n_samples, n_features)
            Data set to cluster

        y : Ignored
            Not used, present here for API consistency

        """
        self.X_ = check_array(X)

        # Build linkage
        z = linkage(X, method="single")
        D = np.vstack((
            np.arange(1, len(self.X_)),
            z[:, 2][::-1]
        )).T

        # Add last point with a 0 cost
        self.eval_graph_ = np.vstack((D, np.array([len(D) + 1., 0.])))
        self.knee_, self.plots_ = DesC.knee_detection_switch[self.knee_method](
            self.eval_graph_,
            True
            )
        self.K_ = int(self.knee_[0])
        self.labels_ = fcluster(z, self.K_, criterion="maxclust")

        # Return the classifier
        return self
