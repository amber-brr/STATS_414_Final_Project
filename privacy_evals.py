import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import kstest, mannwhitneyu
import matplotlib.pyplot as plt

# distance calculation ---
def calculate_nearest_neighbor_distances(source_data, target_data, k=2):

    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='brute')
    nn.fit(target_data)
    
    distances, _ = nn.kneighbors(source_data, n_neighbors=k)
    return distances

# nearest neighbor distance ratio (NNDR) 
def calculate_nndr(synthetic_data, real_data):
    k_max = 5 
    
    # distances for synthetic points to their k_max neighbors in real data
    distances_synth_to_real = calculate_nearest_neighbor_distances(
        synthetic_data, real_data, k=k_max
    )
    
    distance_to_nnr = distances_synth_to_real[:, 0]
    
    average_distance_k_nnr = distances_synth_to_real.mean(axis=1)

    epsilon = 1e-9
    nndr_values = distance_to_nnr / (average_distance_k_nnr + epsilon)
    
    return nndr_values

# distribution of DCR values
def calculate_dcr_distribution_comparison(synthetic_data, real_data, num_bins=50):
    distances_synth_to_real = calculate_nearest_neighbor_distances(
        synthetic_data, real_data, k=1
    )
    dcr_synth_to_real = distances_synth_to_real[:, 0]
    
    distances_real_to_real = calculate_nearest_neighbor_distances(
        real_data, real_data, k=2
    )

    dcr_real_to_real = distances_real_to_real[:, 1]
    
    return dcr_synth_to_real, dcr_real_to_real

# load datasets
REAL_DF_PATH='synthdata/realdf.csv'
CTGAN_DF_PATH='synthdata/ctgan_df.csv'
DPCTGAN_1_5_DF_PATH='synthdata/dpctgan1_5_df.csv'
DPCTGAN_3_DF_PATH='synthdata/dpctgan3_df.csv'
GREAT_DF_PATH='synthdata/greatdf.csv'
RTF_DF_PATH='synthdata/rtfdf.csv'
real_data=pd.read_csv(REAL_DF_PATH)
ctgandf=pd.read_csv(CTGAN_DF_PATH)
dp15df=pd.read_csv(DPCTGAN_1_5_DF_PATH)
dp3df=pd.read_csv(DPCTGAN_3_DF_PATH)
greatdf=pd.read_csv(GREAT_DF_PATH)
rtfdf=pd.read_csv(RTF_DF_PATH)

# fill nans
real_data.fillna(0, inplace=True)
ctgandf.fillna(0, inplace=True)
dp15df.fillna(0, inplace=True)
dp3df.fillna(0, inplace=True)
greatdf.fillna(0, inplace=True)
rtfdf.fillna(0, inplace=True)

# drop unnecessary column from rtfdf
if 'Unnamed: 0' in rtfdf.columns:
    rtfdf = rtfdf.drop(columns=['Unnamed: 0']).values # .values converts DataFrame to NumPy array

synthetic_datasets = {
    'REaLTabFormer': rtfdf,
    'CTGAN': ctgandf,
    'DP-CTGAN (ε=1.5)': dp15df,
    'DP-CTGAN (ε=3)': dp3df,
    'GReaT': greatdf,
}

results = {}

print("Calculating privacy evals.")

for name, synth_data in synthetic_datasets.items():
    print(f"\nProcessing dataset: **{name}**")
    
    # NNDR calculation 
    nndr_values = calculate_nndr(synth_data, real_data)
    mean_nndr = np.mean(nndr_values)
    
    # DCR distribution comparison
    dcr_s_to_r, dcr_r_to_r = calculate_dcr_distribution_comparison(synth_data, real_data)
    
    # store results
    results[name] = {
        'NNDR_Mean': mean_nndr,
        'DCR_S_to_R': dcr_s_to_r,
        'DCR_R_to_R': dcr_r_to_r 
    }

    print(f"Mean NNDR: {mean_nndr:.4f}")
    print(f"Mean DCR (Synthetic to Real): {np.mean(dcr_s_to_r):.4f}")
    print(f"Mean DCR (Real to Real Baseline): {np.mean(dcr_r_to_r):.4f}")

print("done")

