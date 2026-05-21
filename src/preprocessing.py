from sklearn.preprocessing import MinMaxScaler

def add_rul(df):

    max_cycle = df.groupby('engine_id')['cycle'].max()

    df['RUL'] = df.apply(
        lambda row: max_cycle[row['engine_id']] - row['cycle'],
        axis=1
    )

    return df


def prepare_features(df):

    features = df.drop(['engine_id', 'cycle', 'RUL'], axis=1)

    target = df['RUL']

    scaler = MinMaxScaler()

    features_scaled = scaler.fit_transform(features)

    return features_scaled, target, scaler