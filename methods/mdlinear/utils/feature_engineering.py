import pandas as pd


def get_id(df):
    ys = df['y'].unique()
    ys = {y: i for i, y in enumerate(sorted(ys), start=0)}
    ids = df['y'].map(ys)
    return ids


def compute_locs(loc_file):
    locs = pd.read_csv(loc_file)
    locs['cluster'] = 0
    locs.loc[locs['x'] > 500, 'cluster'] = 1
    locs.loc[locs['x'] > 1500, 'cluster'] = 2
    locs.loc[locs['x'] > 2500, 'cluster'] = 3
    locs.loc[locs['x'] > 3500, 'cluster'] = 4
    locs.loc[locs['x'] > 4500, 'cluster'] = 5

    locs['id_in_cluster'] = locs.groupby('cluster', group_keys=False).apply(get_id)
    locs = locs.drop(columns=['x', 'y'])
    return locs


def set_cluster_and_location_id(df, loc_file):
    # Compute clusters depending on the x values and give new ids based on the y values
    # (monotonically increasing within cluster).
    locs = compute_locs(loc_file)
    df = df.merge(locs, on='TurbID')
    return df


def construct_features(df, loc_file):
    df = set_cluster_and_location_id(df, loc_file)

    # Merge Pab features
    df['Pab_max'] = df[['Pab1', 'Pab2', 'Pab3']].max(axis=1)
    df = df.drop(columns=['Pab1', 'Pab2', 'Pab3'])

    # Drop low relevance features
    df = df.drop(columns=['Wdir', 'Ndir', 'Itmp', 'Etmp'])

    # Mean value of cluster
    df['patv_cluster_mean'] = df.groupby(['cluster', 'date'])['Patv'].transform('mean')

    # Set the target column last in the dataframe.
    df = df[[c for c in df if c not in ['Patv']] + ['Patv']]
    return df
