import pandas as pd

def load_data(path):

    columns = ['engine_id', 'cycle']

    settings = ['setting1', 'setting2', 'setting3']

    sensors = [f's{i}' for i in range(1, 22)]

    columns += settings + sensors

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )

    df = df.iloc[:, :26]

    df.columns = columns

    return df