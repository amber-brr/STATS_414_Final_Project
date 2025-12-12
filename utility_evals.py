import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_synthetic_datasets(real_data, synthetic_datasets, target_column):
    
    # split real data into train and test
    _, X_real_test, _, y_real_test = train_test_split(
        real_data.drop(target_column, axis=1),
        real_data[target_column],
        test_size=0.3,
        random_state=42
    )

    results = []

    for i, synth_df in enumerate(synthetic_datasets):
        dataset_name = f"Synthetic_Dataset_{i+1}"
        print(f"Evaluating {dataset_name}...")

        # prepare synthetic training data
        X_train = synth_df.drop(target_column, axis=1)
        y_train = synth_df[target_column]

        # calculate scale_pos_weight for XGBoost 
        num_pos = y_train.sum()
        num_neg = len(y_train) - num_pos
        scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

        # define models
        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            ),
            "XGBoost": XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                scale_pos_weight=scale_pos_weight,
                eval_metric="auc",
                tree_method="hist",
                n_jobs=-1,
                random_state=42
            )
        }

        # train and evaluate models
        for model_name, model in models.items():
            # train on Synthetic
            model.fit(X_train, y_train)

            # predict on real
            y_pred = model.predict(X_real_test)
            
            # predict probs for AUC
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_real_test)[:, 1]
            else:
                y_pred_proba = None

            # calculate metrics
            metrics = {
                "Dataset": dataset_name,
                "Model": model_name,
                "Accuracy": accuracy_score(y_real_test, y_pred),
                "Precision": precision_score(y_real_test, y_pred, zero_division=0),
                "Recall": recall_score(y_real_test, y_pred, zero_division=0),
                "F1 Score": f1_score(y_real_test, y_pred, zero_division=0),
            }
            print(metrics)
            
            if y_pred_proba is not None:
                try:
                    metrics["AUC"] = roc_auc_score(y_real_test, y_pred_proba)
                except ValueError:
                    metrics["AUC"] = np.nan
            else:
                metrics["AUC"] = np.nan

            results.append(metrics)

    # return results 
    return pd.DataFrame(results)

# load datasets
REAL_DF_PATH='synthdata/realdf.csv'
CTGAN_DF_PATH='synthdata/ctgan_df.csv'
DPCTGAN_1_5_DF_PATH='synthdata/dpctgan1_5_df.csv'
DPCTGAN_3_DF_PATH='synthdata/dpctgan3_df.csv'
GREAT_DF_PATH='synthdata/greatdf.csv'
RTF_DF_PATH='synthdata/rtfdf.csv'
real_df=pd.read_csv(REAL_DF_PATH)
ctgandf=pd.read_csv(CTGAN_DF_PATH)
dp15df=pd.read_csv(DPCTGAN_1_5_DF_PATH)
dp3df=pd.read_csv(DPCTGAN_3_DF_PATH)
greatdf=pd.read_csv(GREAT_DF_PATH)
rtfdf=pd.read_csv(RTF_DF_PATH)

# fill nans
real_df.fillna(0, inplace=True)
ctgandf.fillna(0, inplace=True)
dp15df.fillna(0, inplace=True)
dp3df.fillna(0, inplace=True)
greatdf.fillna(0, inplace=True)
rtfdf.fillna(0, inplace=True)

# drop unnecessary column from rtfdf
if 'Unnamed: 0' in rtfdf.columns:
    rtfdf.drop(columns=['Unnamed: 0'],inplace=True) 

# run utility evals
synthetic_dfs = [rtfdf,ctgandf,dp15df,dp3df,greatdf,rtfdf]
evaluation_results = evaluate_synthetic_datasets(
    real_data=real_df,
    synthetic_datasets=synthetic_dfs, 
    target_column='label'
)

# view results
evaluation_results.to_csv('utility_evals.csv',index=False)
print('Saved results df.')
print(evaluation_results)