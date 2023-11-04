## Environment Setup

Run:
```shell
conda env create --name YOUR_ENV_NAME --file=environments.yml
conda activate YOUR_ENV_NAME
export PYTHONDONTWRITEBYTECODE=1
```

You can also use something like `direnv` or some other `rc` runner to run the conten ts of `.fridayrc` based on directory. Up to you.

## Usage

Run:
```shell
python3 run.py
```

Ideal hyperparameters so far:
```shell
Enter window size: 3
Enter number of epochs: 30
Enter batch size: 2
Enter initial learning rate: 0.001
```

> **TODO**: Still have not written actual `model.predict` or `model.evaluate` to test and fine-tune model for validation / test sets.