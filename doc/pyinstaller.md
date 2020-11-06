## Build a single executable using pyinstaller
Enter the project directory and run:
```
pyinstaller --onefile --hidden-import 'sklearn' \
--hidden-import 'appdirs' \
--hidden-import 'packaging.requirements' \
--hidden-import 'sklearn.utils._cython_blas' \
--paths venv/lib/python3.8/site-packages run_detection.py
```
In this example `--path` adds the project virtual environment.
Remove option `--onefile` to not archive the files and speed up the program start by this.

Tested on `5.8.16-2-MANJARO x86_64`.