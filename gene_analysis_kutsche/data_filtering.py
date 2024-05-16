import pandas as pd
from scipy.stats import trim_mean

import pandas as pd

def filter_data_wt(df):
    """
    Filter, organize, and compute a gene expression data by day.
    """
    wt_columns = [col for col in df.columns if 'WT' in col]
    day_map = {col: int(col.split('d')[1].split('_')[0]) for col in wt_columns}
    ordered_columns = sorted(wt_columns, key=lambda x: day_map[x])
    df_filtered_wt = df[ordered_columns]
    
    return df_filtered_wt, day_map, wt_columns

def filter_data_arithmetic_mean(df):
    """
    Filter, organize, and compute an arithmetic mean gene expression data by day.
    """

    df_filtered_wt, day_map, wt_columns = filter_data_wt(df)
    
    mean_per_day = {}
    for day in set(day_map.values()):
        columns_for_day = [col for col in wt_columns if day_map[col] == day]
        daily_data = df_filtered_wt[columns_for_day]
        mean_per_day[day] = daily_data.mean(axis=1)

    df_mean_per_day = pd.DataFrame(mean_per_day)
    return df_mean_per_day, df_filtered_wt, day_map


def filter_data_median(df):
    """
    Filter, organize, and compute a median gene expression data by day.
    """
    df_filtered_wt, day_map, wt_columns = filter_data_wt(df)

    median_per_day = {}
    for day in set(day_map.values()):
        columns_for_day = [col for col in wt_columns if day_map[col] == day]
        daily_data = df_filtered_wt[columns_for_day]
        median_per_day[day] = daily_data.median(axis=1)

    df_median_per_day = pd.DataFrame(median_per_day)
    return df_median_per_day, df_filtered_wt, day_map


def filter_data_proximity_based_weights(df):
    """
    Filter, organize, and compute a robust weighted mean gene expression data by day.
    """
    df_filtered_wt, day_map, wt_columns = filter_data_wt(df)

    robust_mean_per_day = {}
    for day in set(day_map.values()):
        columns_for_day = [col for col in wt_columns if day_map[col] == day]
        daily_data = df_filtered_wt[columns_for_day]

        # Compute median and assign weights based on proximity to median
        median = daily_data.median(axis=1)
        weights = 1 / (1 + daily_data.subtract(median, axis=0).abs())
        sum_weights = weights.sum(axis=1)
        
        # Since DataFrame operations broadcast based on index and columns, ensure sum_weights is a DataFrame
        # This allows division to broadcast properly across the weights DataFrame
        sum_weights_df = sum_weights.to_frame('sum')
        normalized_weights = weights.div(sum_weights_df['sum'], axis=0)

        # Compute weighted mean using normalized weights
        weighted_mean = (daily_data.multiply(normalized_weights)).sum(axis=1)
        robust_mean_per_day[day] = weighted_mean

    df_mean_per_day = pd.DataFrame(robust_mean_per_day)
    return df_mean_per_day, df_filtered_wt, day_map

