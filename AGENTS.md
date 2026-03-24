# AGENTS.md

## Project Structure

This project is a Home Assistant custom integration for Mistral AI
conversation services. The structure follows standard Home Assistant
integration conventions:

```bash
custom_components/mistral_conversation/
├── __init__.py          # Integration setup and entry point
├── config_flow.py       # Configuration flow handler
├── manifest.json        # Integration metadata
├── strings.json         # Translation strings
├── services.yaml        # Service definitions (if any)
└── ...                  # Other Python files
scripts/
└── ...                  # Helper scripts for development
```

## Developer Setup

### uv

Run commands using `uv run`:

```bash
# Run a script
uv run python script.py

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest

# Run linting
uv run ruff check .
```

Dev dependencies (testing, linting, etc.) should use the `--group dev` flag:

```bash
# Add development dependencies
uv add --group dev package-name

# Example: Add pytest with coverage
uv add --group dev pytest pytest-cov
```

This keeps production dependencies clean and separates development tools.

### Linting

Run linting manually:

```bash
# Run all checks
uv run pre-commit run --all-files

# Specific checks
uv run ruff check .
uv run ty .
```

## Documentation Lookup

Use [chub](https://pypi.org/project/chub/) to quickly access Python package documentation:

```bash
uv run chub -h                                 # CLI help
uv run chub search "stripe"                    # BM25 search
uv run chub search "auth" --limit 5            # limit results
uv run chub get stripe/api --lang python       # fetch a doc
uv run chub get openai/chat --version 4.0      # specific version
uv run chub list                               # list all docs
uv run chub list --json                        # JSON output
```

**Note:** Chub does not yet support Home Assistant documentation.
For Home Assistant docs, use:

- [Official Home Assistant Developer Docs](https://developers.home-assistant.io/)

## Testing

Run tests with:

```bash
uv run pytest
```
