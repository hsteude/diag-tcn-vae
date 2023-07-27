# imports
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, HTML
from typing import Dict, List


@dsl.component(
    packages_to_install=["pandas==1.5.3", "pyarrow", "plotly"],
    base_image="python:3.9",
)
def visualize_predictions(
    df_results: Input[Dataset],
    subsystems_map: Dict[str, List[str]],
    plot: Output[HTML],
    split: str,
):
    import pandas as pd
    import plotly.subplots as sp
    import plotly.graph_objects as go

    # read data
    result_df = pd.read_parquet(df_results.path)

    # Create subplots: nrows is the number of keys in subsystems_map
    fig = sp.make_subplots(rows=len(subsystems_map.keys()), cols=1)

    for idx, col in enumerate(sorted(subsystems_map.keys())):
        fig.add_trace(
            go.Box(x=result_df["faultNumber"], y=result_df[col], name=col),
            row=idx+1,
            col=1
        )
    fig.update_xaxes(title_text="faultNumber", row=idx+1, col=1)
    fig.update_yaxes(title_text=col, row=idx+1, col=1)

    fig.update_traces(boxpoints='all', jitter=0.5)
    fig.update_layout(height=400*len(subsystems_map.keys()),
                      title_text=f"Boxplots of recon errors faultNumber (Split: {split}")
    fig.write_html(plot.path)
