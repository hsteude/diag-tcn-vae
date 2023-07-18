# imports
from kfp import dsl
from kfp.dsl import Artifact, Output

@dsl.component(
    base_image="python:3.9",
)
def create_logging_dir(
    logging_dir: Output[Artifact],
):
    pass
