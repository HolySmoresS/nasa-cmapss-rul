# First script (Claude Code will extend this with you). Run it with:  python src/01_look_at_data.py
import pandas as pd

columns = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
train = pd.read_csv("data/train_FD001.txt", sep=r"\s+", header=None, names=columns)

print(train.shape)          # (rows, columns)
print(train.head())         # first 5 rows
print(train["unit"].nunique(), "engines")

# Next step: add an RUL column (last cycle of each engine minus current cycle).
