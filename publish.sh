#!/bin/bash
set -o errexit

# Get project path.
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd ${PROJECT_PATH}

# If local virtualenv exists and none is activated, use it.
if [ -d ".venv" ] && [ -z "${VIRTUAL_ENV}" ]; then
    source .venv/bin/activate
fi

# clean up
rm -rf build
rm -rf dist
rm -rf wav2x.egg-info

# build and upload
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/* --verbose

popd
