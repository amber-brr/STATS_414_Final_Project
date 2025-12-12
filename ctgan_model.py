import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from ctgan import CTGAN

df = pd.read_csv("final_synth_gen_df.csv")

discrete_columns = df.columns.tolist()

ctgan = CTGAN(
    epochs=150,        
    batch_size=256,
    generator_dim=(128, 128),
    discriminator_dim=(128, 128),
    pac=5,
    verbose=True
)

print("Training CTGAN")
ctgan.fit(df, discrete_columns=discrete_columns)

num_samples = len(df)*2  

synthetic_df = ctgan.sample(num_samples)

# Save
synthetic_df.to_csv("ctgan_data.csv", index=False)