"""
Tasks for maintaining the project.

Execute 'invoke --list' for guidance on using Invoke
"""
import platform
import shutil

from invoke import task

try:
    from pathlib import Path
    Path().expanduser()
except (ImportError, AttributeError):
    from pathlib2 import Path

ROOT_DIR = Path(__file__).parent
SETUP_FILE = ROOT_DIR.joinpath("setup.py")
TEST_DIR = ROOT_DIR.joinpath("tests")
SOURCE_DIR = ROOT_DIR.joinpath("delicious_treat")
TOX_DIR = ROOT_DIR.joinpath(".tox")
COVERAGE_FILE = ROOT_DIR.joinpath(".coverage")
COVERAGE_DIR = ROOT_DIR.joinpath("htmlcov")
COVERAGE_REPORT = COVERAGE_DIR.joinpath("index.html")
DOCS_DIR = ROOT_DIR.joinpath("docs")
DOCS_BUILD_DIR = DOCS_DIR.joinpath("_build")
DOCS_INDEX = DOCS_BUILD_DIR.joinpath("index.html")
NOTEBOOK_DIR = ROOT_DIR.joinpath("notebooks")
PYTHON_DIRS = [str(d) for d in [SOURCE_DIR, TEST_DIR, Path(__file__)]]


def _delete_file(file):
    try:
        file.unlink(missing_ok=True)
    except TypeError:
        # missing_ok argument added in 3.8
        try:
            file.unlink()
        except FileNotFoundError:
            pass


@task(help={'check': "Checks if source is formatted without applying changes"})
def format(c, check=False):
    """
    Format code
    """
    python_dirs_string = " ".join(PYTHON_DIRS)
    # Run yapf
    yapf_options = '--recursive {}'.format('--diff' if check else '--in-place')
    c.run("yapf {} {}".format(yapf_options, python_dirs_string))
    # Run isort
    isort_options = '--recursive {}'.format('--check-only' if check else '')
    c.run("isort {} {}".format(isort_options, python_dirs_string))


@task
def lint_flake8(c):
    """Run flake8 linter"""
    c.run("flake8 {}".format(SOURCE_DIR))


@task
def lint_pylint(c):
    """Run flake8 linter"""
    c.run("pylint {}".format(SOURCE_DIR))


@task
def lint_mypy(c):
    """Run type checker"""
    c.run("mypy {} --strict".format(SOURCE_DIR))


@task(pre=[lint_flake8, lint_pylint, lint_mypy])
def lint(c):
    """Lint code"""
    pass


@task
def test(c):
    """
    Run tests
    """
    pty = platform.system() == 'Linux'
    c.run("python {} test".format(SETUP_FILE), pty=pty)


@task(help={'publish': "Publish the result via coveralls"})
def coverage(c, publish=False):
    """
    Create coverage report
    """
    c.run("coverage run --source {} -m pytest".format(SOURCE_DIR))
    c.run("coverage report")
    if publish:
        # Publish the results via coveralls
        c.run("coveralls")
    else:
        # Build a local report
        c.run("coverage html")
        webbrowser.open(COVERAGE_REPORT.as_uri())


@task
def clean_python(c):
    """
    Clean up python file artifacts
    """
    c.run("find . -name '*.pyc' -exec rm -f {} +")
    c.run("find . -name '*.pyo' -exec rm -f {} +")
    c.run("find . -name '*~' -exec rm -f {} +")
    c.run("find . -name '__pycache__' -exec rm -fr {} +")


@task
def clean_tests(c):
    """
    Clean up files from testing
    """
    _delete_file(COVERAGE_FILE)
    shutil.rmtree(TOX_DIR, ignore_errors=True)
    shutil.rmtree(COVERAGE_DIR, ignore_errors=True)


@task
def clean_notbooks(c):
    notebooks = [f"'{n}'" for n in NOTEBOOK_DIR.glob('*.ipynb')]
    c.run(("jupyter nbconvert"
           " --ClearOutputPreprocessor.enabled=True"
           " --inplace {}").format(" ".join(notebooks)))


@task(pre=[clean_python, clean_tests, clean_notbooks])
def clean(c):
    """
    Runs all clean sub-tasks
    """
    pass
