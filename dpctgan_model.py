import pandas as pd
import numpy as np
import pickle
from snsynth import Synthesizer

# load the dataset
file_path = 'final_synth_gen_df.csv'
df = pd.read_csv(file_path)


# define columns
categorical_cols = [
    'label',
    'u_refreshTimes',
    'u_feedLifeCycle',
    'city',
    'device_name',
    'series_group',
    'interaction_type_freq',
    'age',
    'residence',
    'user_num_click_apps',
    'user_num_click_advers'
]

continuous_cols = [
    "article_like_rate",
    "user_total_views",
    'device_size',
    "article_ctr",
    "slot_freq",
    "num_news_interest_st",
    "adv_prim_freq",
    "user_articledoc_clicks",
    "adv_freq",
    "user_category_clicks"
]

discrete_columns = df.select_dtypes(include=['int']).columns.tolist()
continuous_cols=[]
for c in df.columns:
    if c not in discrete_columns:
        continuous_cols.append(c)


# train models
synth1_5 = Synthesizer.create('dpctgan', epochs=150, pac=10, epsilon=1.5, verbose=True)
print("Training DPCTGAN w epsilon=1.5")

synth1_5.fit(
  df,
  categorical_columns=discrete_columns,
  continuous_columns=continuous_cols,
  preprocessor_eps=0.5,
  nullable=True
)

synth3 = Synthesizer.create('dpctgan', epochs=150, pac=10, epsilon=3, verbose=True)
print("Training DPCTGAN w epsilon=3")

synth3.fit(
  df,
  categorical_columns=discrete_columns,
  continuous_columns=continuous_cols,
  preprocessor_eps=0.5,
  nullable=True
)


# sample synthetic data
sample_size = len(df)*2

synthetic_data1_5 = synth1_5.sample(sample_size)
output_file = 'final_synth_gen_df_dp_synthetic1_5.csv'
synthetic_data1_5.to_csv(output_file, index=False)
print(f"Synthetic data generated and saved to {output_file}")

synthetic_data3 = synth3.sample(sample_size )
output_file = 'final_synth_gen_df_dp_synthetic3.csv'
synthetic_data3.to_csv(output_file, index=False)
print(f"Synthetic data generated and saved to {output_file}")


# save models
model_filename = 'dpctgan_model_eps1_5.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(synth1_5, f)
print(f"Model saved to {model_filename}")


model_filename = 'dpctgan_model_eps3.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(synth3, f)
print(f"Model saved to {model_filename}")