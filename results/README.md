# Results

Committed output from the measurement harnesses. Every figure in the project
README is reproducible from these files, and the commands that produced them are
listed beside each.

| file | produced by |
|---|---|
| `robustness.json` | `python -m sim.robustness --seeds 16 --sessions 5000` |
| `sensitivity.json` | `python -m sim.sensitivity --workers 8` |
| `transfer.json` | `python -m sim.transfer` |
| `uplift_result.json` | `python -m sim.uplift_experiment --sessions 20000` |
| `uplift_at_40.json` | `python -m sim.uplift_experiment --sessions 20000 --discount 0.40` |
| `experiment_ab.json` | `python -m sim.experiment --sessions 43000` |
| `experiment_aa.json` | `python -m sim.experiment --sessions 43000 --aa` |

`experiment_ab.json` and `experiment_aa.json` are single-seed runs, kept because
the dashboard's audit-trail panel reads that experiment. The headline figure is
the pooled one in `robustness.json` — a single run at this size carries a
standard deviation near 1.7 percentage points, so it is not quotable on its own.

The response curves the dashboard draws live at `visuals/counterfactual_grid.json`.
