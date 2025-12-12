# NECESSARY IMPORTS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


# LOAD DATA
ADS_FILE_PATH='train_data_ads.csv' 
FEEDS_FILE_PATH='train_data_feeds.csv'

try:
    df_ads = pd.read_csv(ADS_FILE_PATH)
    df_feeds = pd.read_csv(FEEDS_FILE_PATH)
    print("\nData loaded successfully.")
except FileNotFoundError:
    print("Error: One or both of the file paths not found.")
    print("Please ensure the files are in the same directory as the script.")
    exit()


# UTILITY FUNCTION FOR MEMORY OPTIMIZATION 
def optimize_data_types(df):
    """
    Downcasts numerical columns to smaller types (e.g., int64 -> int8) 
    to significantly reduce memory footprint before encoding.
    """
    initial_mem = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"\nOptimizing Data Types")
    print(f"Initial DataFrame memory usage: {initial_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            if 'int' in str(col_type):
                c_min = df[col].min()
                c_max = df[col].max()
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            elif 'float' in str(col_type):
                df[col] = df[col].astype(np.float32) # Use float32 to save memory
        # Convert object columns that are effectively categorical to category type
        elif df[col].nunique() < 50:
             df[col] = df[col].astype('category')


    final_mem = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"Final DataFrame memory usage: {final_mem:.2f} MB (Reduced by: {(initial_mem - final_mem):.2f} MB)")
    return df


# PREPROCESSING FUNCTION
def preprocess_data(df_ads, df_feeds):
    """
    Performs the required filtering, column dropping, and prepares interaction labels.
    """
    print("\nStarting initial preprocessing and filtering...")

    # Keep only users who clicked (label=1) in the ads data
    clicking_user_ids = df_ads[df_ads['label'] == 1]['user_id'].unique()
    print(f"\nTotal unique clicking users found: {len(clicking_user_ids)}")

    # Apply filtering to df_ads
    df_ads = df_ads[df_ads['user_id'].isin(clicking_user_ids)].copy()
    # Drop unncessary columns from df_ads 
    df_ads = df_ads.drop(columns=['site_id', 'pt_d', 'log_id'], inplace=False)
    # Only get creat_type_cd=5 data
    df_ads=df_ads[df_ads['creat_type_cd']==5]
    print(f"df_ads shape after filtering: {df_ads.shape}")

    
    # Prepare feeds data for aggregation: rename userid label    
    df_feeds = df_feeds.rename(columns={'u_userId': 'user_id'})
    # Apply filtering to df_feeds
    df_feeds = df_feeds[df_feeds['user_id'].isin(clicking_user_ids)].copy()
    print(f"df_feeds shape after filtering and renaming: {df_feeds.shape}")

    return df_ads, df_feeds

#HELPER FUNCTION FOR PROCESSING/FEATURE ENGINEERING FOR LIST FEATURES
def parse_list(col):
    """Convert '21489^34426' → ['21489', '34426'], handle empty."""
    if pd.isna(col) or col == "":
        return []
    return col.split("^")

#FEATURE ENGINEERING FUNCTION
def feature_engineering(df_ads, df_feeds):
    """
    Performs feature engineering for ads and feeds data.
    """
    print("\nStarting feature engineering...")

    # Feature Engineering on List Columns
    list_cols_ads = ['ad_click_list_v001','ad_click_list_v002','ad_click_list_v003','ad_close_list_v001','ad_close_list_v002','ad_close_list_v003']

    for c in list_cols_ads:
        df_ads[c] = df_ads[c].astype(str).apply(parse_list)
    print(f"\nExtracted list length features from Ads data and dropped original list columns.")

    list_cols_feeds = ['u_newsCatInterests','u_newsCatDislike','u_newsCatInterestsST','u_click_ca2_news']

    for c in list_cols_feeds:
        df_feeds[c] = df_feeds[c].astype(str).apply(parse_list)
    print(f"Extracted list length features from Feeds data and dropped original list columns.")

    # Various Feature Engineering for Ads Data

    # Nfadv_pr and closes 
    df_ads['user_num_click_tasks'] = df_ads['ad_click_list_v001'].apply(len)
    df_ads['user_num_click_advers']   = df_ads['ad_click_list_v002'].apply(len)
    df_ads['user_num_sclick_app']  = df_ads['ad_click_list_v003'].apply(len)
    df_ads['user_num_close_tasks'] = df_ads['ad_close_list_v001'].apply(len)
    df_ads['user_num_close_advers']   = df_ads['ad_close_list_v002'].apply(len)
    df_ads['user_num_rec_apps_ads_disable']   = df_ads['ad_close_list_v003'].apply(len)

    # Total clicks + closes
    df_ads['user_total_interactions'] = (
        df_ads['user_num_click_tasks'] +
        df_ads['user_num_close_tasks']
    )

    # Task frequency
    task_freq = df_ads['task_id'].value_counts()
    df_ads['task_freq'] = df_ads['task_id'].map(task_freq)

    # Advertiser frequency
    adv_freq = df_ads['adv_id'].value_counts()
    df_ads['adv_freq'] = df_ads['adv_id'].map(adv_freq)

    # Application ID corresponding to the ad delivery task frequency
    app_freq = df_ads['spread_app_id'].value_counts()
    df_ads['app_freq'] = df_ads['spread_app_id'].map(app_freq)

    # Ad placement ID frequency
    slot_freq = df_ads['slot_id'].value_counts()
    df_ads['slot_freq'] = df_ads['slot_id'].map(slot_freq)
    
    # Advertiser ID corresponding to the advertising task frequency
    adv_prim_freq = df_ads['adv_prim_id'].value_counts()
    df_ads['adv_prim_freq'] = df_ads['adv_prim_id'].map(adv_prim_freq)

    # Interaction type of the creative material corresponding to the advertising task
    interaction_type_freq = df_ads['inter_type_cd'].value_counts()
    df_ads['interaction_type_freq'] = df_ads['inter_type_cd'].map(interaction_type_freq)
    
    # Drop columns used in feature engineering for ads data
    df_ads=df_ads.drop(columns=['inter_type_cd','adv_id','task_id','spread_app_id','slot_id','adv_prim_id','creat_type_cd','ad_click_list_v001','ad_click_list_v002','ad_click_list_v003','ad_close_list_v001','ad_close_list_v002','ad_close_list_v003'])
    

    # Various Cleaning/Feature Engineering for Feeds Data

    # Convert u_refreshTimes to numeric for aggregation
    df_feeds['u_refreshTimes'] = pd.to_numeric(df_feeds['u_refreshTimes'], errors='coerce')

    # Convert label/cillabel values from -1 and 1 to 0 and 1
    df_feeds['label'] = df_feeds['label'].replace({-1: 0, 1: 1})
    df_feeds['cillabel'] = df_feeds['cillabel'].replace({-1: 0, 1: 1})

    # Number of user liked news feeds categories, short-term interest classification preferences, dislikes
    df_feeds['num_news_interest_cat'] = df_feeds['u_newsCatInterests'].apply(len)
    df_feeds['num_news_interest_st'] = df_feeds['u_newsCatInterestsST'].apply(len)
    df_feeds['num_dislike_cats'] = df_feeds['u_newsCatDislike'].apply(len)

    # Views, clicks, likes on source domain
    user_agg = df_feeds.groupby('user_id').agg(
        user_total_views = ('label', 'count'),
        user_total_clicks = ('label', 'sum'),
        user_total_likes = ('cillabel', 'sum'),
    )

    # User ctr and like rate (smoothed)
    ALPHA = 10
    user_agg['user_article_ctr'] = (
        (user_agg['user_total_clicks'] + ALPHA) /
        (user_agg['user_total_views'] + 2*ALPHA)
    )
    user_agg['user_article_like_rate'] = (
        (user_agg['user_total_likes'] + ALPHA) /
        (user_agg['user_total_views'] + 2*ALPHA)
    )

    # Weighted click interest per category
    cat_click_weighted = (
        df_feeds[df_feeds['label'] == 1]
        .groupby(['user_id', 'i_cat'])
        .size()
        .reset_index(name='cat_clicks')
    )

    # Weighted like interest per category
    cat_like_weighted = (
        df_feeds[df_feeds['cillabel'] == 1]
        .groupby(['user_id', 'i_cat'])
        .size()
        .reset_index(name='cat_likes')
    )
    cat_click_user = cat_click_weighted.groupby('user_id')['cat_clicks'].sum()
    cat_like_user = cat_like_weighted.groupby('user_id')['cat_likes'].sum()

    user_agg['user_category_clicks'] = user_agg.index.map(cat_click_user).fillna(0)
    user_agg['user_category_likes'] = user_agg.index.map(cat_like_user).fillna(0)

    # Weighted click interest per article
    doc_click_weighted = (
        df_feeds[df_feeds['label'] == 1]
        .groupby(['user_id', 'i_docId'])
        .size()
        .reset_index(name='doc_clicks')
    )

    # Weighted like interest per article
    doc_like_weighted = (
        df_feeds[df_feeds['cillabel'] == 1]
        .groupby(['user_id', 'i_docId'])
        .size()
        .reset_index(name='doc_likes')
    )

    doc_click_user = doc_click_weighted.groupby('user_id')['doc_clicks'].sum()
    doc_like_user = doc_like_weighted.groupby('user_id')['doc_likes'].sum()

    user_agg['user_articledoc_clicks'] = user_agg.index.map(doc_click_user).fillna(0)
    user_agg['user_articledoc_likes'] = user_agg.index.map(doc_like_user).fillna(0)

    # Weighted click interest per source
    source_click_weighted = (
        df_feeds[df_feeds['label'] == 1]
        .groupby(['user_id', 'i_s_sourceId'])
        .size()
        .reset_index(name='source_clicks')
    )

    # Weighted like interest per article
    source_like_weighted = (
        df_feeds[df_feeds['cillabel'] == 1]
        .groupby(['user_id', 'i_s_sourceId'])
        .size()
        .reset_index(name='source_likes')
    )

    source_click_user = source_click_weighted.groupby('user_id')['source_clicks'].sum()
    source_like_user = source_like_weighted.groupby('user_id')['source_likes'].sum()

    user_agg['user_source_clicks'] = user_agg.index.map(source_click_user).fillna(0)
    user_agg['user_source_likes'] = user_agg.index.map(source_like_user).fillna(0)

    

    # Add engineered features back to feeds df
    df_feeds['user_total_views'] = df_feeds['user_id'].map(user_agg['user_total_views'])
    df_feeds['article_ctr'] = df_feeds['user_id'].map(user_agg['user_article_ctr'])
    df_feeds['article_like_rate'] = df_feeds['user_id'].map(user_agg['user_article_like_rate'])
    df_feeds['user_category_clicks'] = df_feeds['user_id'].map(user_agg['user_category_clicks'])
    df_feeds['user_category_likes'] = df_feeds['user_id'].map(user_agg['user_category_likes'])
    df_feeds['user_articledoc_clicks'] = df_feeds['user_id'].map(user_agg['user_articledoc_clicks'])
    df_feeds['user_articledoc_likes'] = df_feeds['user_id'].map(user_agg['user_articledoc_likes'])
    df_feeds['user_source_clicks'] = df_feeds['user_id'].map(user_agg['user_source_clicks'])
    df_feeds['user_source_likes'] = df_feeds['user_id'].map(user_agg['user_source_likes'])
    


    # drop columns used in feature engineering
    df_feeds=df_feeds.drop(columns=['i_docId', 'i_cat','i_entities', 'e_m', 'e_po', 'e_pl', 'e_rn', 'e_section', 'pro','i_s_sourceId', 'u_newsCatInterests','u_newsCatInterestsST','u_newsCatDislike','u_click_ca2_news','label', 'cillabel'])
    
    # Aggregate feeds data
    df_feeds_agg = df_feeds.groupby('user_id').agg({
    'num_news_interest_cat': 'mean',
    'num_news_interest_st': 'mean',
    'num_dislike_cats': 'mean',
    'user_total_views': 'mean',
    'user_source_clicks': 'mean',
    'user_source_likes': 'mean',
    'article_ctr': 'mean',
    'article_like_rate': 'mean',
    'user_category_clicks': 'first',
    'user_category_likes': 'mean',
    'user_articledoc_clicks': 'mean',
    'user_articledoc_likes': 'mean',
    'user_source_clicks': 'mean',
    'user_source_likes': 'mean',
    }).reset_index()

    print("\nFinished preprocessing and feature engineering.")

    # Merge Ads and Feeds Data
    df_merged = pd.merge(optimize_data_types(df_ads), optimize_data_types(df_feeds_agg), on='user_id', how='left')
    df_final=df_merged.drop(columns=['u_newsCatInterestsST'])
    print(f"\nMerged Ads and Feeds data. Final shape: {df_merged.shape}")

    print("\nFinished creating cleaned, feature engineered, and merged dataframe.")

    # Set X,y for training
    X = df_final.drop(columns=['label'])
    y = df_final['label']

    return X, y, df_final



def run_feature_importance(X, y, model=None):
    """
    Trains Random Forests and XGBoost and extracts feature importance.
    """
    if model is None:
        print("\nStarting Random Forest Feature Importance Analysis...")
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
        rf_model.fit(X, y)
    else:
        print("\nUsing trained Random Forest Model for Feature Importance Analysis...")
        rf_model = model

    # Extract feature importances
    rf_importances = rf_model.feature_importances_
    feature_names = X.columns
    rf_imp = pd.Series(rf_importances, index=feature_names, name='rf_importance')

    # Sort and display top features
    top_n = 20
    sorted_importances = rf_imp.sort_values(ascending=False).head(top_n)

    print(f"\nTop {top_n} Feature Importances (Random Forest)")
    print(sorted_importances.to_string())

    pos = y.sum()
    neg = len(y) - pos
    scale_pos_weight = neg / pos    
    if model is None:
        print("\nStarting XGBoost Feature Importance Analysis...")
        xgb_model = XGBClassifier(
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
        xgb_model.fit(X, y)
    else:
        print("\nUsing trained Random Forest Model for Feature Importance Analysis...")
        xgb_model = model

    # Extract feature importances
    xgbimportances = xgb_model.feature_importances_
    feature_names = X.columns
    xgb_imp = pd.Series(xgbimportances, index=feature_names, name='xgb_importance')

    # Sort and display top features
    top_n = 20
    sorted_importances = xgb_imp.sort_values(ascending=False).head(top_n)
    print(f"\nTop {top_n} Feature Importances (XGBoost)")
    print(sorted_importances.to_string())

    # Create importance df for both rf and xgb
    imp_df = pd.concat([rf_imp, xgb_imp], axis=1)
    imp_df['rf_norm']  = imp_df['rf_importance'] / imp_df['rf_importance'].sum()
    imp_df['xgb_norm'] = imp_df['xgb_importance'] / imp_df['xgb_importance'].sum()

    # Average normalized importance
    imp_df['mean_norm'] = imp_df[['rf_norm', 'xgb_norm']].mean(axis=1)

    # Ranks (1 = most important)
    imp_df['rf_rank']  = imp_df['rf_importance'].rank(ascending=False)
    imp_df['xgb_rank'] = imp_df['xgb_importance'].rank(ascending=False)
    imp_df['mean_rank'] = imp_df[['rf_rank', 'xgb_rank']].mean(axis=1)

    imp_df.to_csv('imp_df.csv',index=False)
    print("\nSaved df with feature importance ranking.")
    K = 80
    selected_features_topk = (
        imp_df
        .sort_values('mean_rank', ascending=True)
        .head(K)
        .index
        .tolist()
    )
    top20=selected_features_topk[:20]
    print(f"\nFinal Top 20 Feature Importances: {selected_features_topk[:20]}")
    return top20


# MAIN EXECUTION
if __name__ == "__main__":
    # read files
    # df_ads = pd.read_csv(ADS_FILE_PATH)
    # df_feeds = pd.read_csv(FEEDS_FILE_PATH)

    # # apply preprocessing
    # df_ads_pre, df_feeds_pre = preprocess_data(df_ads, df_feeds)

    # # apply feature engineering 
    # X_final, y_final, df_final = feature_engineering(df_ads_pre, df_feeds_pre)
    df_final=pd.read_csv('synthdata/realdf.csv')
    # generate X, y
    X_final = df_final.drop(columns=['label'])
    y_final = df_final['label']

    # onluy use important columns from top 20 random forests + xgb feature importances
    top20=run_feature_importance(X_final, y_final)
    print(top20)
    df_final=df_final[['label','u_refreshTimes','article_like_rate','user_total_views','device_size','article_ctr','slot_freq','num_news_interest_st','u_feedLifeCycle','user_num_click_apps','city','adv_prim_freq','device_name','user_articledoc_clicks','series_group','adv_freq','user_category_clicks','interaction_type_freq','age','residence','user_num_click_advers']]
                        
    
    # Create final dataset for synthetic data generation
    df_final.to_csv('final_synth_gen_df.csv', index=False) 
    print("\n Saved dataframe with top 20 important features for synthetic data generation.")

    print("\ndone")