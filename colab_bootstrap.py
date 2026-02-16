import os
import sys
import importlib


def is_colab():
    """Check if running in Google Colab."""
    return 'COLAB_RELEASE_TAG' in os.environ


def upload_lib(repo_url='https://github.com/rlogwood/fs-ml-lib.git', lib_dir=None, force_refresh=False):
    """Clone/pull lib files from public GitHub repo."""
    if not is_colab():
        print("Running locally, NOT in Google Colab, using local lib files")
        return


    lib_dir = lib_dir or '/content/lib'
    print("Running in Google Colab, uploading lib files to:", lib_dir)

    if force_refresh and os.path.exists(lib_dir):
        import shutil
        shutil.rmtree(lib_dir)

    git_dir = os.path.join(lib_dir, '.git')

    if os.path.exists(git_dir):
        os.system(f'cd {lib_dir} && git pull')
        print(f"✓ Pulled latest lib files to {lib_dir}")
        reload_lib_modules()
    else:
        # Remove empty/corrupt directory if it exists
        if os.path.exists(lib_dir):
            import shutil
            shutil.rmtree(lib_dir)
        os.system(f'git clone {repo_url} {lib_dir}')
        print(f"✓ Cloned lib files to {lib_dir}")

def setup_paths():
    """Add lib to Python path."""
    if is_colab():
        if '/content' not in sys.path:
            sys.path.insert(0, '/content')
    else:
        # Local: lib is 2 levels up from notebook
        assignments_path = os.path.abspath('../../')
        if assignments_path not in sys.path:
            sys.path.insert(0, assignments_path)


def reload_lib_modules():
    """Reload all currently imported lib.* modules."""
    lib_modules = [name for name in sys.modules if name.startswith('lib.')]

    for module_name in lib_modules:
        importlib.reload(sys.modules[module_name])
        print(f"  ✓ Reloaded {module_name}")

    if not lib_modules:
        print("  No lib modules loaded yet")


# Auto-run on import
setup_paths()
