import warnings
warnings.simplefilter(action='ignore')
import os
import sys
import pickle
parent_dir = os.path.join(os.getcwd(), '..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from api.source import forecast

with open('../data/source_data.pkl', 'rb') as f:
    data = pickle.load(f) 
pred, pinyin = forecast(data, model_name='RandomForest', city_name='臺北市', num_of_days=100)
print(pred.__len__(), pinyin, pred[:4])

pred, pinyin = forecast(data, model_name='GPT', city_name='臺北市', num_of_days=100)
print(pred.__len__(), pinyin, pred[:4])