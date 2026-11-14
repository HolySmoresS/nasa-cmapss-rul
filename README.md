# NASA C-MAPSS Remaining Useful Life Prediction

**Goal:** predict how many flight cycles a jet engine has left before failure, using simulated NASA sensor data (C-MAPSS, subset FD001).

**Status:** Project 0 in progress.

## Results
| Model | Test RMSE (cycles) | Notes |
|---|---|---|
| Mean baseline | | |

## What I'm learning
(fill in as you go)

## Data
NASA Prognostics Center of Excellence data repository, Turbofan Engine Degradation Simulation Data Set. Not included here; download it and place `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` in `data/`.

## Setup
```bash
python3 -m venv .venv                       # private package folder for this project
.venv/bin/pip install pandas scikit-learn matplotlib
.venv/bin/python src/01_look_at_data.py     # quick check that the data loads
```
