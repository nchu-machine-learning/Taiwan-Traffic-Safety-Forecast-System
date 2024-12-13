from copy import deepcopy
import numpy as np
from tqdm import tqdm

def autoregressive_predicting(predictor, final_segment, test_len):
    """
    Perform autoregressive predictions for a specified number of time steps.

    Parameters
    ----------
    predictor : object
        A trained model with a `predict` method that accepts a 2D array of inputs.
    final_segment : np.array
        The last segment of the training data, used as the initial input for predictions.
    test_len : int
        The number of time steps to predict.

    Returns
    -------
    res : list
        A list of predicted values for the specified number of time steps.

    Notes
    -----
    - The function iteratively predicts one step ahead and updates the input by appending the prediction
      and removing the oldest value.
    - This approach is suitable for autoregressive models or time series forecasting tasks.
    - The predictions are rounded to the nearest integer before being stored in the result.

    Example
    -------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> predictor = RandomForestRegressor().fit(train_x, train_y)
    >>> predictions = autoregressive_predicting(predictor, final_segment, test_len=10)
    """
    x = np.array(deepcopy(final_segment))
    out = predictor.predict([x]).round()
    res = [out]
    for _ in tqdm(range(test_len - 1)):
        x = np.concatenate([x[1:], out])
        out = predictor.predict([x]).round()
        res.append(out)
    return res

def select_model(
        model: object,
        config: dict,
        train_x: np.array,
        train_y: np.array
):
    """
    Initialize, configure, and train a regression model.

    Parameters
    ----------
    model : object
        The model class to be used (e.g., `RandomForestRegressor` or `GradientBoostingRegressor`).
    config : dict
        Configuration parameters for initializing the model. These are passed as keyword arguments
        to the model's constructor.
    train_x : np.array
        Training features, typically a 2D array where each row corresponds to a data sample.
    train_y : np.array
        Training labels, typically a 1D array where each element corresponds to the target value
        for a data sample.

    Returns
    -------
    regressor : object
        The trained regression model.

    Notes
    -----
    - This function abstracts the model selection and training process, making it easier to switch
      between different regression models.
    - The input model should be compatible with the scikit-learn API (i.e., have `fit` and `predict` methods).

    Example
    -------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> config = {"n_estimators": 100, "max_depth": 5}
    >>> model = select_model(RandomForestRegressor, config, train_x, train_y)
    """
    regressor = model(*config)
    regressor.fit(train_x, train_y)
    return regressor

