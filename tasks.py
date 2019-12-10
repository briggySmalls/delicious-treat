"""
Tasks for the image processing repository
"""
from pathlib import Path

from invoke import task

_ROOT_DIR = Path(__file__).parent
_NOTEBOOK_DIR = _ROOT_DIR.joinpath('notebooks')


@task()
def clean(c):
    """
    Clean notebook output
    """
    notebooks = [f"'{n}'" for n in _NOTEBOOK_DIR.glob('*.ipynb')]
    c.run(("jupyter nbconvert"
           " --ClearOutputPreprocessor.enabled=True"
           " --inplace {}").format(" ".join(notebooks)))
