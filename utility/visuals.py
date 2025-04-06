
import os
import sys
parent_path = os.path.join(os.getcwd(), '..')
if parent_path not in sys.path:
    sys.path.append(parent_path)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dtaidistance import dtw
from matplotlib.dates import DateFormatter, MonthLocator
from utility.utils import select_model, autoregressive_predicting
import re
def plot_time_series(
        train: pd.DataFrame,
        test: pd.DataFrame,
        figsize: tuple = (15, 3),
        xlabel: str = "Dates",
        ylabel: str = "DTW Scores",
        xticks: np.array = np.array([]),
        rot: int = 60,
        title: str = 'forecasting plot',
        pred: np.array = np.array([]),
        metric: dict = {'dtw': dtw.distance},
        saving=False,
        train_limit: int = None  # New parameter to limit number of training points
    ) -> dict:
    """
    Plot a time series for training, testing, and optionally predictions.

    Parameters
    ----------
    train : pd.DataFrame
        The training time series data.
    test : pd.DataFrame
        The testing time series data.
    figsize : tuple, optional
        Size of the figure, by default (15, 3).
    xlabel : str, optional
        Label for the x-axis, by default "Dates".
    ylabel : str, optional
        Label for the y-axis, by default "DTW Scores".
    xticks : np.array, optional
        The tick labels for the x-axis, typically corresponding to datetime values.
    rot : int, optional
        The rotation angle for x-axis labels, by default 60.
    title : str, optional
        Title of the plot, by default 'forecasting plot'.
    pred : np.array, optional
        Predicted values for the test data. If provided, they are plotted alongside test data.
    metric : dict, optional
        A dictionary of metric functions to compute scores between predictions and actual test data.
    saving : bool, optional
        If True, saves the plot to file.
    train_limit : int, optional
        If set, limits the number of training data points plotted. Default is None (no limit).

    Returns
    -------
    loss_map : dict
        Dictionary containing computed metric scores (if predictions are provided).
    """
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    
    if train_limit is not None:
        train = train[:train_limit]
        xticks = xticks[:train_limit + len(test)]

    plt.figure(figsize=figsize)
    plt.plot(xticks[:len(train)], train, label='Training')
    plt.plot(xticks[len(train): len(train) + len(test)], test, label='Testing')
    
    is_pred = len(pred) == len(test)
    if is_pred:
        plt.plot(xticks[len(train): len(train) + len(test)], pred, label='Prediction', c=(100/255, 205/255, 100/255))

    plt.xticks(rotation=rot)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))

    plt.legend(loc='upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    loss_map = {}
    if is_pred:
        loss_map = {k: v(pred, test) for k, v in metric.items()}

    plt.title(
        title + f" score: {round(loss_map['dtw'], 2)}"
        if is_pred else title
    )

    if saving:
        if '\n' in title:
            title = title.split('\n')[0]
        title = re.sub(r'/', '_', title)
        plt.savefig("../figure/" + title + ".png", bbox_inches="tight", dpi=300)

    plt.show()
    return loss_map

 
def yield_visuals(
        model: object,
        final_segment: np.array,
        config,
        train_x,
        train_y,
        train,
        test,
        date_num,
        title,
        saving=False,
        selecting=True,
        train_limit: int = None,  # New parameter to limit number of training points
        figsize: tuple = (15, 3),
) -> dict:
    """
    Train a model, make predictions, and plot the results.

    Parameters
    ----------
    model : object
        The initial model object to be trained.
    final_segment : np.array
        The last segment of the training data, used for generating predictions.
    config : object
        Configuration object or parameters for the model.
    train_x : np.array
        Features for training the model.
    train_y : np.array
        Labels for training the model.
    train : pd.DataFrame
        The training time series data.
    test : pd.DataFrame
        The testing time series data.
    date_num : np.array
        An array of datetime values for the x-axis of the plot.
    title : str
        The title of the plot.
    saving : bool, default is False.
        To save the figure or not.
    selecting : bool, default is True .
        Call the select_model function or not
    train_limit : int, optional
        If set, limits the number of training data points plotted. Default is None (no limit).
    figsize : tuple, optional
        Size of the figure, by default (15, 3).
    Returns
    -------
    loss_map : dict

    Notes
    -----
    - This function selects the appropriate model, trains it, generates predictions,
      and plots the results with the `plot_time_series` function.
    - The plot includes training, testing, and predicted values if available.
    """
    if selecting:
        model = select_model(model, config, train_x, train_y)
    title += "/" + model.__str__().split('(')[0] + "\n"
    pred = autoregressive_predicting(model, final_segment, len(test))
    return plot_time_series(train, test, xticks=date_num, title=title, pred=pred, saving=saving, train_limit=train_limit, figsize=figsize)    