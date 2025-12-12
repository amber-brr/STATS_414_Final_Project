# 0. Install (from a shell, not inside Python)
# pip install realtabformer pandas scikit-learn numpy

import pandas as pd
import numpy as np

from realtabformer import REaLTabFormer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


# load data 
df = pd.read_csv("final_synth_gen_df.csv")


# train model
rtf = REaLTabFormer(
    model_type="tabular",
    epochs=50,
    gradient_accumulation_steps=4,
    logging_steps=100,
    mask_rate=0.1 # implement 10% mask rate from paper
)

rtf.fit(df, qt_max="compute", num_bootstrap=10, resume_from_checkpoint=True)

# generate synthetic data 
sample_size=len(df)*2
print('Saving synthetic data.')
synthetic = rtf.sample(n_samples=sample_size)
synthetic.to_csv('rtf_synth.csv')
print('Saved synthetic data.')

# save model
print('Saving RTF model.')
rtf.save("rtf_model/")
print ('Saved RTF model')