

# Utilizing Generative Pre-trained Transformers for Predictive Modeling of Traffic Casualties in Taiwan
<!-- [![View Paper Utilizing Generative Pre-trained Transformers for
Predictive Modeling of Traffic Casualties in Taiwan](https://img.shields.io/badge/View-Paper-blue)](https://github.com/nchu-machine-learning/Taiwan-Traffic-Safety-Forecast-System/blob/main/paper/Utilizing_Generative_Pre_trained_Transformers_for_Predictive_Modeling_of_Traffic_Casualties_in_Taiwan.pdf)
-->
## The purpose of the system
* To implement a forecasting system that allows you to choose the models you want to make predictions
## Components of the repo
1. Frontend Vue
2. Backend Flask: for testing, templates\form.html is provided. 
3. Python notebooks: also for testing, training models, or running experiments.
# Running the server
- main.py: main.py contains the flask code is placed
- Activate the virtual environment containing all the dependencies. In the command, type
```bash
python main.py
```
for Linux users, type
```bash
python3 main.py
```
in the command
# Running the Experiment

To explore the experiment setup, navigate to the `experiment` directory:

- **E_2.ipynb**: Use this notebook for training models. It demonstrates how to prepare data, configure parameters, and run the training process.
- **E_3.ipynb**: Use this notebook for application development. It provides an example of how to call the prediction functions from your chosen models and visualize or utilize the results.

Before running these notebooks, ensure that the following directories exist in the project’s root:

- `figure`  
- `output`  

Make sure to activate the appropriate virtual environment and then run the desired notebook.

---

# Installation

To install all required dependencies, execute:

```bash
pip install -r requirements.txt
```

# Reference
This repository used the GPT-2 model from hugging face.
[https://huggingface.co/docs/transformers/model_doc/gpt2](https://huggingface.co/docs/transformers/model_doc/gpt2)
