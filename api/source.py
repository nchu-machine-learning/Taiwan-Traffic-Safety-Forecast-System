import warnings
warnings.simplefilter(action='ignore')
import os
import sys
parent_dir = os.path.join(os.getcwd(), '..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from pypinyin import pinyin, Style
from utility.data import data_imputer, data_slicing, \
    transformer_slice, get_dict_from_pd, train_test_split
from utility.utils import select_model, autoregressive_predicting
from model.gpt import GPT_predict, get_desired_sequence_by_len, GPT_fit
import torch
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
ML_MODELS = {"RandomForest": RandomForestRegressor, "AdaBoost": AdaBoostRegressor, "GradientBoost": GradientBoostingRegressor}
LLM_MODELS = {"GPT"}

def forecast(
        df,
        city_name: str = '臺北市',
        model_name: str = 'GPT',
        params: dict[str, object] = {
            'lookback': 150,
            'config': {}
        },
        num_of_days: int = 30,
        device: torch.device = torch.device('cpu'),
        model_params_directory: str = '../output/result/'
    ) -> tuple[list, str, list, tuple]:
    '''
    Forecast future values for a specified city using a selected model (ML or LLM-based).

    Parameters
    ----------
    df : pandas.DataFrame or similar data structure
        The data used for training and forecasting. It should contain time-series data
        from various cities. The exact structure should align with what `data_imputer`
        and subsequent data processing functions expect.
    
    city_name : str, default = "臺北市"
        The target city for which you want to make predictions.
    
    model_name : str, default = "GPT"
        The name of the model to use. It must be one of:
            - ML_MODELS: {"RandomForest", "AdaBoost", "GradientBoost"}
            - LLM_MODELS: {"GPT"}
        Using "GPT" triggers a Large Language Model (LLM) approach, while the ML options
        will use traditional machine learning methods.
    
    params : dict[str, object], optional
        A dictionary of parameters. At minimum, it should include:
            'lookback': int
                The number of historical time steps to use as input for the model.
            'config': dict, optional
                Additional configuration parameters for the chosen model.
        For ML models, 'config' might include hyperparameters. For the LLM (GPT),
        'config' could be used to define model architecture or loading checkpoints.
    
    num_of_days : int, default = 30
        The number of future days to forecast.
    
    device : torch.device, default = torch.device('cpu')
        The computing device on which the model runs. Useful for LLM models (e.g., "cuda:0").

    model_params_directory : str, default = '../output/result/'
        The target directory where the GPT model's parameters are placed.
    Returns
    -------
    tuple[list, str, list, tuple]
        A tuple containing:
        
        pred : list
            The forecasted values for the specified number of days into the future.
            For ML models, this will be a list of numeric predictions.
            For LLM (GPT), this will be a list of predicted numeric values extracted
            from the generated sequence.
        
        roman_repre : str
            The city name represented in pinyin with tone numbers, joined by underscores.
            For example, "臺北市" -> "tai2_bei3_shi4".

        training_data : list
            The data from the selected city and also used for model training

        date_num : tuple(datetime, datetime)
            The starting date and ending day for the training dataset

    Examples
    --------
    >>> import pickle
    >>> with open('../data/source_data.pkl', 'rb') as f:
    ...     data = pickle.load(f)
    >>> pred, pinyin_representation, training_data, (start_date, end_date) = forecast(
    ...     df=data,
    ...     city_name='臺北市',
    ...     model_name='GPT',
    ...     num_of_days=100
    ... )
    >>> len(pred)
    100
    >>> pinyin_representation
    'tai2_bei3_shi4'

    Notes
    -----
    - Ensure that `df` is in a format compatible with `data_imputer` and `get_dict_from_pd`.
    - The LLM approach (`model_name="GPT"`) expects a certain directory structure for checkpointing.
    - ML models are selected through `select_model`, which uses 'params["config"]' for configuration.
    '''

    # Determine model type based on model_name
    if model_name in ML_MODELS:
        model_type = "ML"
    elif model_name in LLM_MODELS:
        model_type = "LLM"
    else:
        raise ValueError(f"{model_name} is not available for now.")
    
    # Data processing: impute missing values and retrieve sequences
    df_list, date_range = data_imputer(df)
    date_num = date_range.values
    sequences = get_dict_from_pd(df_list, 'address1', '受傷')
    training_data = sequences[city_name]

    roman_repre = "_".join([item[0] for item in pinyin(city_name, style=Style.TONE3)])
    if model_type == "ML":
        # For ML models, prepare data and fit a selected traditional ML model
        train_x, train_y, final_segment = data_slicing(
            training_data,
            params['lookback']
        )
        model = select_model(ML_MODELS[model_name], params['config'], train_x, train_y)
        pred = autoregressive_predicting(model, final_segment, num_of_days)
    else:
        # For LLM (GPT), convert city name to pinyin and load/fine-tune GPT model
        trainer, model, final_segment = GPT_fit(
            training_data, 
            checkpoint_dir=model_params_directory + roman_repre,
            trained=True,
            device=device
        )
        pred = GPT_predict(
            model=model,
            final_segment=final_segment,
            max_length=num_of_days + params['lookback'] + 10,
            device=device
        )
        pred = get_desired_sequence_by_len(pred, final_segment, length=num_of_days)

    return pred, roman_repre, training_data.tolist(), (str(date_num[0])[:10], str(date_num[-1])[:10]) 
    # as the date_num contains a number of datetime-typed elements, we parse them into strings and take the first 7 elements which contain the date information
