#!/bin/bash
set -o errexit

# Get project path.
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd ${PROJECT_PATH}

# If local virtualenv exists and none is activated, use it.
if [ -d ".venv" ] && [ -z "${VIRTUAL_ENV}" ]; then
    source .venv/bin/activate
fi

# Add project modules to PYTHONPATH.
if [[ "${PYTHONPATH}" != *"${PROJECT_PATH}"* ]]; then
    export PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}"
fi

rm -f .coverage

# Run tests.
for TEST_FILE in $(find tests -name "*_test.py"); do
    echo "Running tests in ${TEST_FILE}"
    python3 -m coverage run -a ${TEST_FILE}
done
echo "All tests passed!"

popd
