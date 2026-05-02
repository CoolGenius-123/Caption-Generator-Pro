# Repository Guidelines

## Project Structure & Module Organization

- `code/caption_generator.py` contains the Tkinter UI, model loading, generation logic, batch processing, and save helpers.
- `assets/` contains UI assets such as `logo.ico` and the README screenshot.
- `requirements.txt` lists Python runtime dependencies.
- `install.bat` creates/uses `.venv` and installs CUDA PyTorch plus app dependencies.
- `startui.bat` activates `.venv` and launches the app.

There is no dedicated `tests/` directory. Keep new source code under `code/` unless a clear module split is needed.

## Build, Test, and Development Commands

Run from the repository root:

```cmd
install.bat
```

Creates `.venv`, installs CUDA-enabled PyTorch when possible, and installs `requirements.txt`.

```cmd
startui.bat
```

Starts the Tkinter app using the local virtual environment.

```cmd
.venv\Scripts\python.exe code\caption_generator.py
```

Runs the app directly during development.

```cmd
.venv\Scripts\python.exe -c "import ast, pathlib; ast.parse(pathlib.Path('code/caption_generator.py').read_text(encoding='utf-8'))"
```

Performs a lightweight syntax check without opening the GUI.

## Coding Style & Naming Conventions

Use Python 3.10+ syntax and 4-space indentation. Follow the existing style: constants in `UPPER_SNAKE_CASE`, helper functions in `snake_case`, and internal app methods prefixed with `_`. Keep comments brief.

Prefer `pathlib.Path` for filesystem paths and avoid hard-coded absolute paths in app logic. Keep UI text short and specific.

## Testing Guidelines

No formal test framework is configured yet. For changes, at minimum run the syntax check above and import-check key dependencies:

```cmd
.venv\Scripts\python.exe -c "import torch, transformers, qwen_vl_utils, PIL, psutil; print('imports ok')"
```

For UI or generation changes, manually launch `startui.bat` and verify single-image mode, batch mode, output saving, and model load error handling as relevant. If tests are added later, place them under `tests/` and name files `test_*.py`.

## Commit & Pull Request Guidelines

Existing commits use short title-case summaries, for example `Update Readme File`. Keep commit messages concise and imperative when possible, such as `Fix Qwen Launch Script`.

Pull requests should include a brief description, checks run, and screenshots for visible UI changes. Note model, CUDA, or dependency changes because they affect install size and runtime compatibility.

## Security & Configuration Tips

Do not commit `.venv/`, model caches, Hugging Face tokens, generated captions, or local image datasets. Tokens should be entered through the UI only when needed. Treat model downloads and dependency upgrades as user-visible changes.
