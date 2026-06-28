# Contributing

## Setup

```bash
just setup
```

## Develop

```bash
just build-maturin   # compile the plugin into the venv
just test            # rust + python tests
just ci              # full local CI (lock-check, fmt, lint, test, package dry-runs)
```

See `justfile` for all targets.
