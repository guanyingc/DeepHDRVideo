find . -name '*.pyc' -exec rm {} \;
find . -name '*.pyo' -exec rm {} \;
find . -name '__pycache__' -exec rm -r {} \;
