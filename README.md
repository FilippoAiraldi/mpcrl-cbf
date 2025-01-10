# TODO: write readme

## 2D Constrained LTI System

### Commands

```bash
python lti/eval.py scmpc --horizon=12 --dcbf --soft --n-eval=100 --save=${save_filename} --n-jobs=${number_of_parallel_jobs}
python lti/train.py lstd-ql --horizon=1 --dcbf --soft --terminal-cost=pwqnn --save=${training_results_filename} --n-jobs=${number_of_parallel_jobs}
python lti/eval.py scmpc --horizon=1 --dcbf --soft --terminal-cost=pwqnn --from-file=${training_results_filename} --n-eval=100 --save=${save_filename} --n-jobs=${number_of_parallel_jobs}
```
