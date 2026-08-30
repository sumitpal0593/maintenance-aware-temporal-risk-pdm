# Scania Component X data

The raw Scania Component X dataset is **not distributed in this repository**.

Download the public dataset from its official source and store the CSV files in
a local directory. Then copy `.env.example` to `.env` and set:

```text
SCANIA_DATA_DIR=C:/your/local/path/to/the/data
```

The revision pipeline expects the following files:

```text
train_operational_readouts.csv
validation_operational_readouts.csv
train_specifications.csv
validation_specifications.csv
train_tte.csv
validation_labels.csv
```

Additional test files can also be placed in the same local directory if needed.

The dataset directory is intentionally excluded by `.gitignore`.
