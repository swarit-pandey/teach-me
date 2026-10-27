# AI from Scratch

A visual, bottom-up course: linear algebra → calculus → probability → ML in NumPy →
deep learning (your own autograd) → transformers (a GPT you train on CPU) → RL on
transformers (PPO/RLHF, DPO, GRPO).

```sh
./course setup      # .venv with numpy, torch (CPU), pytest
open index.html     # course home: map + progress
./course check 01   # grade workbook 01
```

- `index.html`: the course map and your progress (stored in the browser).
- `lessons/NN-*.html`: interactive lessons (open locally; no server or network needed).
- `workbooks/NN_*/`: `exercises.py` (yours), `test_exercises.py` (the grader),
  `solutions.py` (look after a genuine attempt).
- `./course help`: all commands. `./course validate` and `./course pages` are author checks.
- `AUTHORING.md`: conventions, if you want to add lessons.
