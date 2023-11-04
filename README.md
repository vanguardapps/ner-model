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

Example hyperparameters (not the greatest probably):
```shell
Enter window size: 3
Enter number of epochs: 30
Enter batch size: 2
Enter initial learning rate: 0.001
```

The printed output at the end corresponds to the following label associations (also in the `data.py`):
```
0   'O',        # Untagged
1   'B-geo',    # Geographical Entity (Beginning)
2   'I-geo',    # Geographical Entity (Inside)
3   'B-org',    # Organization (Beginning)
4   'I-org',    # Organization (Inside)
5   'B-per',    # Person (Beginning)
6   'I-per',    # Person (Inside)
7   'B-gpe',    # Geopolitical Entity (Beginning)
8   'I-gpe',    # Geopolitical Entity (Inside)
9   'B-tim',    # Time indicator (Beginning)
10  'I-tim',    # Time indicator (Inside)
11  'B-art',    # Artifact (Beginning)
12  'I-art',    # Artifact (Inside)
13  'B-eve',    # Event (Beginning)
14  'I-eve',    # Event (Inside)
16  'B-nat',    # Natural Phenomenon (Beginning)
17  'I-nat',    # Natural Phenomenon (Inside)
```

> **TODO**: Still have not written actual `model.predict` or `model.evaluate` to test and fine-tune model for validation / test sets.