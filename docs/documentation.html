<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Forecasting Utilities Documentation</title>
<style>
    body {
        margin: 0;
        font-family: "Helvetica Neue", Arial, sans-serif;
        line-height: 1.6;
        background: #f9f9f9;
        color: #333;
    }
    header, nav, main, footer {
        padding: 20px;
        max-width: 1200px;
        margin: 0 auto;
    }
    header h1 {
        margin-bottom: 0;
        font-size: 2.5em;
        font-weight: 700;
    }
    header p {
        margin-top: 0;
        font-size: 1.1em;
        color: #555;
    }
    nav {
        background: #fff;
        border-bottom: 1px solid #eee;
        margin-bottom: 30px;
    }
    nav ul {
        list-style-type: none;
        padding-left: 0;
    }
    nav ul li {
        display: inline-block;
        margin-right: 20px;
    }
    nav ul li a {
        text-decoration: none;
        color: #0066cc;
        font-weight: 500;
    }
    nav ul li a:hover {
        text-decoration: underline;
    }
    main h2 {
        margin-top: 50px;
        font-size: 2em;
        border-bottom: 2px solid #eee;
        padding-bottom: 10px;
    }
    main h3 {
        margin-top: 30px;
        font-size: 1.5em;
        color: #333;
        border-bottom: 1px solid #eee;
        padding-bottom: 5px;
    }
    main p, main li {
        font-size: 1em;
        line-height: 1.8em;
    }
    main code, pre code {
        font-family: "Source Code Pro", monospace;
        background: #fafafa;
        padding: 2px 4px;
        font-size: 0.95em;
        border-radius: 4px;
        color: #c7254e;
    }
    pre {
        background: #f0f0f0;
        padding: 15px;
        border-radius: 5px;
        overflow-x: auto;
    }
    footer {
        text-align: center;
        color: #999;
        font-size: 0.9em;
        margin-top: 50px;
    }
    .parameters, .returns, .examples, .notes {
        margin-top: 10px;
        background: #fff;
        border: 1px solid #eee;
        border-radius: 5px;
        padding: 15px;
    }
    .parameters h4,
    .returns h4,
    .examples h4,
    .notes h4 {
        margin-top: 0;
        color: #333;
        font-size: 1.2em;
    }
    .parameters ul,
    .returns ul,
    .examples pre,
    .notes p {
        font-size: 0.95em;
    }
    .highlight {
        color: #0066cc;
        font-weight: bold;
    }
</style>
</head>
<body>
<header>
    <h1>Forecasting Utilities Documentation</h1>
    <p>A set of functions for training, predicting, and managing forecasting models using both traditional ML approaches and Large Language Models (GPT).</p>
</header>

<nav>
    <ul>
        <li><a href="#forecast">forecast</a></li>
        <li><a href="#gpt_fit">GPT_fit</a></li>
        <li><a href="#gpt_predict">GPT_predict</a></li>
        <li><a href="#get_desired_sequence">get_desired_sequence</a></li>
        <li><a href="#get_desired_sequence_by_len">get_desired_sequence_by_len</a></li>
        <li><a href="#autoregressive_predicting">autoregressive_predicting</a></li>
        <li><a href="#select_model">select_model</a></li>
    </ul>
</nav>

<main>
    <h2 id="forecast">Function: forecast</h2>
    <p>Forecast future values for a specified city using a selected model (ML or LLM-based).</p>
    <div class="parameters">
        <h4>Parameters</h4>
        <ul>
            <li><code>df</code> : <em>DataFrame-like</em><br/>Time-series data aligned with <code>data_imputer</code> functions.</li>
            <li><code>city_name</code> : <em>str</em>, default = "臺北市"<br/>Target city name for prediction.</li>
            <li><code>model_name</code> : <em>str</em>, default = "GPT"<br/>Must be one of {"RandomForest", "AdaBoost", "GradientBoost"} or {"GPT"}.</li>
            <li><code>params</code> : <em>dict</em>, default includes 'lookback':150, 'config':{}<br/>Model hyperparameters and lookback window.</li>
            <li><code>num_of_days</code> : <em>int</em>, default = 30<br/>Number of future days to forecast.</li>
            <li><code>device</code> : <em>torch.device</em>, default = cpu<br/>Computational device for model execution.</li>
        </ul>
    </div>
    <div class="returns">
        <h4>Returns</h4>
        <ul>
            <li><code>pred</code> : <em>list</em><br/>List of predicted values for the forecast horizon.</li>
            <li><code>roman_repre</code> : <em>str</em><br/>City name in pinyin with tone numbers.</li>
        </ul>
    </div>
    <div class="examples">
        <h4>Examples</h4>
        <pre><code>>> import pickle
>>> with open('../data/source_data.pkl', 'rb') as f:
...     data = pickle.load(f)
>>> pred, pinyin_representation = forecast(
...     df=data,
...     city_name='臺北市',
...     model_name='GPT',
...     num_of_days=100
... )
>>> len(pred)
100
>>> pinyin_representation
'tai2_bei3_shi4'</code></pre>
    </div>
    <div class="notes">
        <h4>Notes</h4>
        <p>- Ensure <code>df</code> matches the format expected by <code>data_imputer</code>.</p>
        <p>- GPT model expects proper directory structures for checkpoints.</p>
        <p>- ML models are configured via <code>params['config']</code>.</p>
    </div>

    <h2 id="gpt_fit">Function: GPT_fit</h2>
    <p>Train a GPT-like model or load from a checkpoint.</p>
    <div class="parameters">
        <h4>Parameters</h4>
        <ul>
            <li><code>train</code> : <em>array-like</em><br/>Training data as a sequence of tokens.</li>
            <li><code>params</code> : <em>dict</em><br/>Model config (embedding size, #layers, #heads).</li>
            <li><code>checkpoint_dir</code> : <em>str</em><br/>Directory for model checkpoints.</li>
            <li><code>trained</code> : <em>bool</em><br/>If True, loads model from checkpoint.</li>
            <li><code>device</code> : <em>torch.device</em><br/>Computational device.</li>
        </ul>
    </div>
    <div class="returns">
        <h4>Returns</h4>
        <ul>
            <li><code>trainer</code> : <em>Trainer</em><br/>The Hugging Face Trainer object.</li>
            <li><code>model</code> : <em>GPT2LMHeadModel</em><br/>The trained GPT model.</li>
            <li><code>final_segment</code> : <em>list</em><br/>The final input segment used for prediction.</li>
        </ul>
    </div>

    <h2 id="gpt_predict">Function: GPT_predict</h2>
    <p>Generate predictions using a trained GPT model.</p>
    <div class="parameters">
        <h4>Parameters</h4>
        <ul>
            <li><code>model</code> : <em>GPT2LMHeadModel</em><br/>The trained GPT model.</li>
            <li><code>final_segment</code> : <em>list</em><br/>Prompt sequence for generation.</li>
            <li><code>max_length</code> : <em>int</em><br/>Max length of generated sequence.</li>
            <li><code>num_return_sequences</code> : <em>int</em><br/>Number of sequences to generate.</li>
            <li><code>device</code> : <em>torch.device</em><br/>Computational device.</li>
        </ul>
    </div>
    <div class="returns">
        <h4>Returns</h4>
        <ul>
            <li><code>outputs</code> : <em>torch.Tensor</em><br/>Generated sequences as a tensor.</li>
        </ul>
    </div>

    <h2 id="get_desired_sequence">Function: get_desired_sequence</h2>
    <p>Extract a portion of the generated sequence matching the test sequence length.</p>
    <div class="parameters">
        <h4>Parameters</h4>
        <ul>
            <li><code>pred</code> : <em>torch.Tensor</em></li>
            <li><code>final_segment</code> : <em>list</em></li>
            <li><code>test</code> : <em>array-like</em><br/>Ground truth sequence for comparison.</li>
        </ul>
    </div>
    <div class="returns">
        <h4>Returns</h4>
        <ul>
            <li><em>np.ndarray</em>: Extracted sequence of length <code>len(test)</code>.</li>
        </ul>
    </div>

    <h2 id="get_desired_sequence_by_len">Function: get_desired_sequence_by_len</h2>
    <p>Extract a portion of the generated sequence by specifying a target length.</p>
    <div class="parameters">
        <h4>Parameters</h4>
        <ul>
            <li><code>pred</code> : <em>torch.Tensor</em></li>
            <li><code>final_segment</code> : <em>list</em></li>
            <li><code>length</code> : <em>int</em><br/>Desired length of output sequence.</li>
        </ul>
    </div>
    <div class="returns">
        <h4>Returns</h4>
        <ul>
            <li><em>np.ndarray</em>: Extracted sequence of specified length.</li>
        </ul>
    </div>

    <h2 id="autoregressive_predicting">Function: autoregressive_predicting</h2>
    <p>Perform autoregressive predictions step-by-step for a specified number of timesteps.</p>
    <div class="parameters">
        <h4>Parameters</h4>
        <ul>
            <li><code>predictor</code> : <em>object</em><br/>A trained model with a <code>predict</code> method.</li>
            <li><code>final_segment</code> : <em>np.array</em><br/>Last segment of training data as initial input.</li>
            <li><code>test_len</code> : <em>int</em><br/>Number of future steps to predict.</li>
        </ul>
    </div>
    <div class="returns">
        <h4>Returns</h4>
        <ul>
            <li><em>list</em>: A list of predicted values for the given number of steps.</li>
        </ul>
    </div>

    <h2 id="select_model">Function: select_model</h2>
    <p>Initialize, configure, and train a regression model (compatible with scikit-learn API).</p>
    <div class="parameters">
        <h4>Parameters</h4>
        <ul>
            <li><code>model</code> : <em>object</em><br/>Model class (e.g., RandomForestRegressor).</li>
            <li><code>config</code> : <em>dict</em><br/>Hyperparameters passed to the model's constructor.</li>
            <li><code>train_x</code> : <em>np.array</em><br/>Training feature matrix.</li>
            <li><code>train_y</code> : <em>np.array</em><br/>Training labels.</li>
        </ul>
    </div>
    <div class="returns">
        <h4>Returns</h4>
        <ul>
            <li><em>object</em>: The trained regression model instance.</li>
        </ul>
    </div>
</main>

<footer>
    <p>© 2024 Forecasting Utilities</p>
</footer>
</body>
</html>
