# Project: NASA C-MAPSS Remaining Useful Life (Project 0)

## Who I am
Community college student, complete beginner at coding.
Long-term goal: AI/ML spike built around NASA Earth-observation work, aiming at
NASA internships (NCAS, L'SPACE NPWEE, OSTEM, EarthRISE) and transfer to Stanford/MIT.
This repo is the first of several projects (Project 0 -> NC flood/burn-scar mapper -> research write-up).

## How you should work with me (IMPORTANT)
- You write the code. I read it and learn from it. Do the work, but teach as you go.
- Before each step: say in plain language WHAT you are about to do and WHY (what problem it solves, how it connects to the goal).
- After each step: explain what the output means and what we learned. Define any term (RMSE, train/test split, overfitting, data leakage) with an everyday analogy the first time it appears.
- Add short comments in the code explaining the "why", not just the "what".
- I am a beginner. No unexplained jargon. Ask me before making big design choices and give 2-3 options with a recommendation.
- Build in small steps and run the code yourself to check it works. Never claim a result you did not actually run.
- Every so often, quiz me with one short question ("why did we split by engine instead of by row?") so I can explain this work out loud later (essays, interviews, recommenders).
- Keep me honest: if a result looks too good, check for data leakage and tell me.
- Keep README.md and LOG.md up to date as we go. Commit after each working step with a clear message, and tell me what the commit contains.
- Never commit data files or secrets.

## The task
Predict Remaining Useful Life (cycles left before failure) for simulated turbofan engines.
Dataset: NASA C-MAPSS, subset FD001 only (100 train engines, 100 test engines, 1 operating condition, 1 fault mode).
Files go in data/ (git-ignored): train_FD001.txt, test_FD001.txt, RUL_FD001.txt.
Columns: unit, cycle, op1, op2, op3, s1..s21 (space-separated, no header).
Train labels: RUL = (max cycle of that unit) - cycle.
Test: evaluate ONLY on the last row of each test engine vs RUL_FD001.txt.
Metric: RMSE (lower is better).

## Reference results (verified Nov 13, 2026, on FD001 test set)
- Predict training-mean RUL for everything: RMSE ~52.6  (the baseline to beat)
- Linear regression, raw RUL labels: ~32.0
- Linear regression, RUL clipped at 125: ~21.9
- Gradient boosting, RUL clipped at 125: ~18.4
- Published deep-learning results on FD001 are typically ~12-16 (goal for later weeks).
Constant columns (zero variance) should be dropped: op3, s1, s5, s10, s16, s18, s19.
That leaves 17 usable features (op1, op2 + 15 sensors), measured Nov 13, 2026.

## Milestones
1. Week 1: load data, create RUL labels, plot a few engines, mean baseline.
2. Week 2: linear regression + proper train/validation split by ENGINE (not by row).
3. Week 3: clip RUL at ~125, gradient boosting, rolling-window features.
4. Week 4: results table, README write-up, honest "what failed" section.
5. Later: 1D-CNN/LSTM in PyTorch; then move to NASA HLS satellite data (Project 1).

## Current status (updated Nov 14, 2026)
- Weeks 1 to 4 are DONE, plus one follow-up investigation. Read README.md and LOG.md for the full story before starting anything.
- Results on the FD001 test set (all run and logged): mean baseline 52.62, linear regression 32.04, linear clipped 21.90, gradient boosting clipped 18.36, gradient boosting clipped + "change from each engine's own start" features 15.18 (current best).
- Key finding: engines start at different sensor levels, so comparing each engine to its own first 20 flights matters more than smoothing.
- Layout: shared helpers in src/cmapss.py; each step is a numbered script in src/ AND an executed notebook in notebooks/; plots in figures/. Run code with .venv/bin/python.
- NEXT: the "Later" milestone, a 1D-CNN or LSTM in PyTorch on FD001 (PyTorch is not installed yet). Ask me before choosing the design.

## Rules for this repo
- Log every experiment in LOG.md (date, model, RMSE, what changed, what I learned).
- Small commits with clear messages. Commit after each working step.
- Never commit the data files.

## Repo
GitHub: https://github.com/HolySmoresS/nasa-cmapss-rul  (rename on GitHub if the URL differs; keep `git remote -v` in sync)
Setup commands (git init, first commit, push) are fine for Claude Code to run, one at a time, explaining each.
