# SimplePad Palace AMR study driver

Use this driver to prepare, run, and analyze the four SimplePad Palace adaptive-mesh refinement cases. Run commands from the SCGSim checkout with the study dependencies installed.

## Run all four prepared cases

The default `run` command requires both Route A and Route B to have conformal and nonconformal handoffs prepared:

```bash
uv run python -m research.palace_amr_simplepad run /path/to/study
```

## Run complete prepared route pairs

If preparation records a route failure, use `--prepared-routes-only` to run another route only when both of its handoffs are sealed:

```bash
uv run python -m research.palace_amr_simplepad run /path/to/study --prepared-routes-only
```

Cases from failed routes are recorded as not run and receive no result payload. The command returns a nonzero status when cases are skipped or a solver run fails. The default command continues to require all four cases.

## Analyze completed runs

After the selected route runs finish, analyze the study directory:

```bash
uv run python -m research.palace_amr_simplepad analyze /path/to/study
```
