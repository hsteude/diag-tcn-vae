# imports
from kfp import dsl
from kfp.dsl import Artifact, Input, Output, Dataset


@dsl.component(
    packages_to_install=["pandas==1.5.3", "pyarrow", "scikit-learn"],
    base_image="python:3.9",
)
def fit_scaler(
    df_train: Input[Dataset],
    scaler: Output[Artifact],
):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import joblib

    # read data
    df_train = pd.read_parquet(df_train.path)

    # fit scaler
    sc = StandardScaler()
    sc.fit(df_train.values)
    joblib.dump(sc, scaler.path)

