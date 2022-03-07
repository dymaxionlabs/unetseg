# Installation

## Stable release

Use pip to install from PyPI:

Install from pip:

```
pip install unetseg
```

## From source

The source for satproc can be installed from the GitHub repo.

```
python -m pip install git+https://github.com/dymaxionlabs/unetseg.git
```

To install for local development, you can clone the repository:

```
git clone https://github.com/dymaxionlabs/unetseg.git
```

If you don't have [Poetry](https://python-poetry.org/) installed, follow 
[these instructions](https://python-poetry.org/docs/master/#installing-with-the-official-installer) first.

Then, install all dependencies. Poetry will automatically create a virtual environment for you:

```
cd unetseg
poetry install
```

Whenever you want to bring the latest changes, just run `git pull` from the
cloned repository.
