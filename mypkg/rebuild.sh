python setup.py bdist_wheel
pip uninstall mypkg
pip install mypkg dist/mypkg-0.0.1-py3-none-any.whl  --no-build-isolation