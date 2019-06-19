# -*- coding: utf-8 -*-
import numpy as np
from sklearn.utils.validation import check_array


def kneedle_scan(X, return_plots=False):
    """Knee detection using Kneedle method

    Parameters
    ----------
    X : array-like, shape: (n_samples, n_features)
        Data graph in which you want to find the knee.
    return_plots : boolean, optional
        If True, also return a data list to plot for graphical explanations.

    Returns
    -------
    knee : ndarray, shape: (n_features)
        The detected knee point
    plots : tuple of ndarray, optional
        Tuple of plots for graphical explanations

    Raise
    -----
    ValueError
        If paramater X has less than 2 samples

    Examples
    --------
    >>> from clustering import kneedetector
    >>> import numpy as np
    >>> X = np.array([[0, 8], [1, 4], [2, 2], [4, 1], [8, 0]])
    >>> kneedetector.kneedle_scan(X)
    array([2, 2])

    """

    X = check_array(X)
    if len(X) <= 2:
        raise ValueError("The dataset size is 2 or less."
                         + " There's no knee in such graph.")

    a = (X[-1, 1] - X[0, 1]) / (X[-1, 0] - X[0, 0])
    b = X[0, 1] - a * X[0, 0]
    line = np.vstack((X[:, 0], [a * x + b for x in X[:, 0]])).T
    dist_to_line = line[:, 1] - X[:, 1]
    dist_to_line = dist_to_line[1: -1]  # Remove edges
    index = np.argmax(dist_to_line) + 1
    knee = X[index]

    if return_plots:
        plots = (
            np.vstack((line[0], line[-1])),
            np.vstack((knee, line[index]))
            )
        return knee, plots
    return knee


def l_method_scan(X, return_plots=False):
    """Knee detection using L-method

    Parameters
    ----------
    X : array-like, shape: (n_samples, n_features)
        Data graph in which you want to find the knee.
    return_plots : boolean, optional
        If True, also return a data list to plot for graphical explanations.

    Returns
    -------
    knee : ndarray, shape: (n_features)
        The detected knee point
    plots : tuple of ndarray, optional
        Tuple of plots for graphical explanations

    Raise
    -----
    ValueError
        If paramater X has less than 2 samples

    Examples
    --------
    >>> from clustering import kneedetector
    >>> import numpy as np
    >>> X = np.array([[0, 8], [1, 4], [2, 2], [4, 1], [8, 0]])
    >>> kneedetector.l_method_scan(X)
    array([2, 2])

    """

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error as mse

    X = check_array(X)
    if len(X) <= 2:
        raise ValueError("The dataset size is 2 or less."
                         + " There's no knee in such graph.")

    best_rmse = None
    for i in range(1, len(X) - 1):
        left = X[: i + 1, :]
        right = X[i:, :]

        lr = LinearRegression()

        lr.fit(left[:, 0, None], left[:, 1])
        left_pred = lr.predict(left[:, 0, None])

        lr.fit(right[:, 0, None], right[:, 1])
        right_pred = lr.predict(right[:, 0, None])

        rmse = (
            len(left) / len(X)
            * np.sqrt(mse(left[:, 1], left_pred))
            + len(right) / len(X)
            * np.sqrt(mse(right[:, 1], right_pred))
            )

        if best_rmse is None or best_rmse >= rmse:
            best_rmse = rmse
            knee = X[i]
            if return_plots:
                left_line = np.vstack((left[:, 0], left_pred)).T
                right_line = np.vstack((right[:, 0], right_pred)).T

    if return_plots:
        plots = (left_line, right_line)
        return knee, plots
    return knee


def max_amplitude_scan(X, return_plots=False):
    """Knee detection using max amplitude method

    Parameters
    ----------
    X : array-like, shape: (n_samples, n_features)
        Data graph in which you want to find the knee.
    return_plots : boolean, optional
        If True, also return a data list to plot for graphical explanations.

    Returns
    -------
    knee : ndarray, shape: (n_features)
        The detected knee point
    plots : tuple of ndarray, optional
        Tuple of plots for graphical explanations

    Raise
    -----
    ValueError
        If paramater X has less than 2 samples

    Examples
    --------
    >>> from clustering import kneedetector
    >>> import numpy as np
    >>> X = np.array([[0, 8], [1, 4], [2, 2], [4, 1], [8, 0]])
    >>> kneedetector.max_amplitude_scan(X)
    array([1, 4])

    """

    X = check_array(X)
    if len(X) <= 2:
        raise ValueError("The dataset size is 2 or less."
                         + " There's no knee in such graph.")

    max_amp = 0
    knee = None
    index = None
    for i in range(1, len(X)):
        amp = abs(X[i - 1, 1] - X[i, 1]) / abs(X[i - 1][0] - X[i][0])
        if amp > max_amp:
            index = i
            knee = X[i]
            max_amp = amp

    if return_plots:
        plots = (np.vstack((knee, X[index - 1])))
        return knee, plots
    return knee
