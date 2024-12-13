
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
        xlabel: str = "x",
        ylabel: str = "y",
        xticks: np.array = np.array([]),
        rot: int = 60,
        title: str = 'forecasting plot',
        pred: np.array = np.array([]),
        metric: dict = {'dtw': dtw.distance},
        saving=False
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
        Size of the figure, by default (15, 2).
    xlabel : str, optional
        Label for the x-axis, by default "x".
    ylabel : str, optional
        Label for the y-axis, by default "y".
    xticks : np.array, optional
        The tick labels for the x-axis, typically corresponding to datetime values.
        Default is an empty array.
    rot : int, optional
        The rotation angle for x-axis labels, by default 80.
    title : str, optional
        Title of the plot, by default 'forecasting plot'.
    pred : np.array, optional
        Predicted values for the test data. If provided, they are plotted alongside test data.
        Default is an empty array.
    metric : dict, optional
        A dictionary of metric functions to compute scores between predictions and actual test data.
        Default includes Dynamic Time Warping (DTW).

    Returns
    -------
    loss_map : dict

    Notes
    -----
    This function supports the visualization of time series data with an optional prediction overlay
    and computes metric scores if predictions are provided.
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.figure(figsize=figsize)
    plt.plot(xticks[:len(train)], train, label='Training')
    plt.plot(xticks[len(train): len(train) + len(test)], test, label='Testing')
    is_pred = pred.__len__() == len(test)
    if is_pred:
        plt.plot(xticks[len(train): len(train) + len(test)], pred, label='Prediction', c=(100/255, 205/255, 100/255))

    plt.xticks(rotation=rot)
    ax = plt.gca()  # Get current axis
    ax.xaxis.set_major_locator(MonthLocator())  # Show ticks for each month
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))  # Display year and month only
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if is_pred: loss_map = {k: v(pred, test) for k, v in metric.items()}
    else: loss_map = {}
    plt.title(title + " score: " +\
               str(loss_map) \
                if is_pred else title)
    if saving:
        if '\n' in title: title = title.split('\n')[0]
        title = re.sub(r'/', '_', title)
        plt.savefig( "../figure/"+ title + ".png", bbox_inches = "tight")
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
        selecting=True
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
    return plot_time_series(train, test, xticks=date_num, title=title, pred=pred, saving=saving)    