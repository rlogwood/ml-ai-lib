# fs-ml-lib

Machine learning helpers used in FullStack ML/AI course.

## Quick Start (Colab)

Add this to the top of each notebook:
```python
# === COLAB BOOTSTRAP ===
BOOTSTRAP_URL = 'https://raw.githubusercontent.com/rlogwood/fs-ml-lib/main/colab_bootstrap.py'
import urllib.request; exec(urllib.request.urlopen(BOOTSTRAP_URL).read().decode())
# === END BOOTSTRAP ===

upload_lib()

# Your imports
import lib.text_util as tu
import lib.wrangler as wr

# Reload after code changes
reload_lib_modules()
```

## Functions

### `is_colab()`
Returns `True` if running in Google Colab, `False` otherwise.

### `upload_lib(repo_url=None, lib_dir=None, force_refresh=False)`
Clones or pulls lib files from GitHub.

- `repo_url`: Repository URL (defaults to this repo)
- `lib_dir`: Destination directory (default: `/content/lib`)
- `force_refresh`: If `True`, deletes and re-clones fresh
- No-op when running locally

### `setup_paths()`
Adds lib to Python path. Called automatically when bootstrap loads.

- Colab: adds `/content` to path
- Local: adds `../../` to path

### `reload_lib_modules()`
Reloads all currently imported `lib.*` modules. Run after pulling updates.

## Workflow

### Initial Setup (once per Colab session)
1. Run bootstrap code
2. `upload_lib()` clones the repo
3. Import modules you need

### When Lib Code Changes
1. Push changes to this repo
2. In Colab: `upload_lib()` pulls latest
3. Run `reload_lib_modules()` to pick up changes

### Pin to a Version (optional)
```python
BOOTSTRAP_URL = 'https://raw.githubusercontent.com/rlogwood/fs-ml-lib/v1.0/colab_bootstrap.py'
upload_lib('https://github.com/rlogwood/fs-ml-lib.git@v1.0')
```

## Project Structure
```
fs-ml-lib/
├── colab_bootstrap.py
├── README.md
├── text_util.py
├── wrangler.py
├── data_cleaner.py
└── ...
```

Local notebooks expect:
```
assignments/
├── lib -> ~/fs-ml-lib (symlink) or copy
├── assignment1/
│   └── notebook.ipynb
└── assignment2/
    └── notebook.ipynb
```