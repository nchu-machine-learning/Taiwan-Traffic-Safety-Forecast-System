
import pandas as pd
import numpy as np


def data_imputer(
    df: pd.DataFrame
    ):
    """
    Impute missing data for each group in a DataFrame based on a complete date range.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing columns `datetime`, `address1`, `address2`,
        and `受傷` (a column to aggregate and impute).

    Returns
    -------
    df_list : list of pd.DataFrame
        A list of DataFrames where missing values in the `受傷` column are imputed with zeros.
        Each DataFrame corresponds to a unique `address1` group.
    date_range : pd.DatetimeIndex
        The full date range spanning the minimum and maximum `datetime` in the input data.

    Notes
    -----
    - Rows with missing values for the `datetime` column are filled with zeros for each address.
    - The `is_holiday` and `address2` columns are dropped before grouping and imputing.
    """
    date_range = pd.date_range(df.datetime.min(), df.datetime.max(), freq='D')

        
    grouped = df.drop(['is_holiday', 'address2'], axis=1) \
        .groupby(['datetime', 'address1']).sum().reset_index().groupby('address1')

    full_df = pd.DataFrame({'datetime': date_range})

    df_list = []
    for g in grouped:
        target_df = g[1][['datetime', 'address1', '受傷']].copy()
        imputed_data = full_df.merge(target_df, on='datetime', how='left').fillna(0)
        imputed_data['address1'] = g[0]
        df_list.append(imputed_data)
    return df_list, date_range

def get_dict_from_pd(
        df_list: list[pd.DataFrame],
        key_col: str,
        val_col: str
    ):
    """
    Convert a list of DataFrames into a dictionary.

    Parameters
    ----------
    df_list : list of pd.DataFrame
        A list of DataFrames, typically with a common structure.
    key_col : str
        The column name to use as the key for the dictionary.
    val_col : str
        The column name to use as the values for the dictionary.

    Returns
    -------
    dict
        A dictionary where each key is the unique value in `key_col` from a DataFrame,
        and the corresponding value is an array of values from `val_col`.

    Example
    -------
    >>> df_list = [pd.DataFrame({'key': ['A'], 'val': [1]})]
    >>> get_dict_from_pd(df_list, 'key', 'val')
    {'A': array([1])}
    """
    return {ele[key_col].iloc[0]: ele[val_col].values for ele in df_list}

def train_test_split(
        data,
        length: int,
        ratio=.8,
    ):
    """
    Split a dataset into training and testing sets based on a specified ratio.

    Parameters
    ----------
    data : array-like
        The dataset to split.
    length : int
        The total length of the dataset.
    ratio : float, optional
        The ratio of the dataset to allocate to the training set. Default is 0.8 (80%).

    Returns
    -------
    tuple
        A tuple containing the training data and testing data.

    Example
    -------
    >>> data = np.arange(10)
    >>> train, test = train_test_split(data, length=10, ratio=0.8)
    >>> train
    array([0, 1, 2, 3, 4, 5, 6, 7])
    >>> test
    array([8, 9])
    """
    training_length = int(length * ratio)
    return data[:training_length], data[training_length:]
def data_slicing(
        data,
        lookback
    ):
    """
    Slice a dataset into input-output pairs for supervised learning.

    Parameters
    ----------
    data : array-like
        The dataset to be sliced.
    lookback : int
        The number of previous time steps to use as input for each output.

    Returns
    -------
    tuple
        A tuple containing:
        - x : list
            A list of input sequences of length `lookback`.
        - y : list
            A list of output values corresponding to each input sequence.
        - final_segment : array-like
            The last segment of the data of length `lookback`.

    Example
    -------
    >>> data = np.arange(10)
    >>> x, y, final_segment = data_slicing(data, lookback=3)
    >>> x
    [array([0, 1, 2]), array([1, 2, 3]), ...]
    >>> y
    [3, 4, 5, ...]
    >>> final_segment
    array([7, 8, 9])
    """
    x = []
    y = []
    for i in range(len(data) - lookback):
        x.append(data[i:i + lookback])
        y.append(data[i+lookback])
    final_segment = data[-lookback:]
    return x, y, final_segment
def transformer_slice(
        data,
        lookback:int
    ):
    """
    Slice a dataset into overlapping sequences for transformer-based models.

    Parameters
    ----------
    data : array-like
        The dataset to be sliced.
    lookback : int
        The number of time steps to include in each sequence.

    Returns
    -------
    tuple
        A tuple containing:
        - x : list
            A list of input sequences of length `lookback`.
        - y : list
            A list of output sequences of length `lookback`, offset by one step.
        - final_segment : array-like
            The last segment of the data of length `lookback`.

    Example
    -------
    >>> data = np.arange(10)
    >>> x, y, final_segment = transformer_slice(data, lookback=3)
    >>> x
    [array([0, 1, 2]), array([1, 2, 3]), ...]
    >>> y
    [array([1, 2, 3]), array([2, 3, 4]), ...]
    >>> final_segment
    array([7, 8, 9])
    """
    x = []
    y = []
    for i in range(len(data) - lookback - 1):
        x.append(data[i:i + lookback])
        y.append(data[i+1: i+1+lookback])
    final_segment = data[-lookback:]
    return x, y, final_segment