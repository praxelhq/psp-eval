# Publishing psp-eval to PyPI

This file documents the publish flow. **Do not publish from CI
automatically** — each release is a conscious act.

## One-time setup

1. Create a PyPI account at https://pypi.org/account/register/.
2. (Optional but recommended) Register on https://test.pypi.org/ as well
   so you can dry-run uploads.
3. Generate a scoped API token (PyPI → Account settings → API tokens →
   Scope: "Project: psp-eval" *after* the first upload, or
   "Entire account" for the very first push) and save it in `~/.pypirc`:

   ```ini
   [pypi]
   username = __token__
   password = pypi-AgEI...

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = pypi-AgENdGVzd...
   ```

## Release steps

```bash
# 0. Make sure you're in a clean working tree on main.
git status  # should be empty

# 1. Bump the version in pyproject.toml (and psp_eval/__init__.py
#    if you kept a __version__ there). Commit it.
#    Use SemVer: 0.1.0 → 0.2.0 for new features; 0.1.1 for patches.

# 2. Build wheel + sdist into ./dist/.
uv pip install --system build twine
python -m build .

# 3. Sanity-check the build.
twine check dist/*

# 4. Dry-run to TestPyPI first.
twine upload --repository testpypi dist/*
# Verify at https://test.pypi.org/project/psp-eval/.
# Install + smoke test:
python -m venv /tmp/psp-eval-test && source /tmp/psp-eval-test/bin/activate
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ psp-eval
python -c "from psp_eval import score_directory; print(score_directory.__doc__)"
deactivate && rm -rf /tmp/psp-eval-test

# 5. Publish to real PyPI.
twine upload dist/*

# 6. Tag the release in git.
git tag -a v0.1.0 -m "psp-eval 0.1.0 — first public PyPI release"
git push origin v0.1.0
```

## What NOT to publish

- Centroid tensor files (`huggingface/psp-centroids/*.pt`) — too big for a
  wheel; they belong on Hugging Face Hub at `Praxel/psp-native-centroids`.
- The paper PDF (`paper/psp.pdf`) — browsable from the repo.
- Benchmark result JSON files — users re-generate their own with their
  own systems.

The `MANIFEST.in` already prunes these paths.

## Post-publish smoke test

After the real upload completes, run:

```bash
pip install --upgrade psp-eval
psp-score --help
```

`psp-score` is the `[project.scripts]` entry point exposed by this package.
