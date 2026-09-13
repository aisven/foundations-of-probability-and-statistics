#!/bin/sh

# enable shell tracing
#set -x
# treat unset variables as an error and exit
set -u
# exit immediately if any command fails
set -e


# ----- determine project module name -----

# absolute path of this shell script
SCRIPT_PATH=$(realpath "$0" 2>/dev/null || readlink -f "$0" 2>/dev/null || (cd "$(dirname "$0")" && pwd -P))
# absolute path of the directory in which this shell script resides
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
# name of the directory in which this shell script resides
DIR_NAME=$(basename "$SCRIPT_DIR")
# name of main project module as per convention
PROJECT_MODULE_NAME=$(echo "$DIR_NAME" | tr '-' '_')

echo "SCRIPT_PATH=${SCRIPT_PATH}"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "DIR_NAME=${DIR_NAME}"
echo "PROJECT_MODULE_NAME=${PROJECT_MODULE_NAME}"


# ----- delete project build artifacts -----

if [ -d "./build" ]; then
    rm -rf ./build
    echo "Deleted ./build"
else
    echo "Directory ./build does not exist"
fi

if [ -d "./${PROJECT_MODULE_NAME}.egg-info" ]; then
    rm -rf ./${PROJECT_MODULE_NAME}.egg-info
    echo "Deleted ./${PROJECT_MODULE_NAME}.egg-info"
else
    echo "Directory ./${PROJECT_MODULE_NAME}.egg-info does not exist"
fi


# ----- delete Python build artifacts -----

# certain files
find . -name "*.pyc" -type f -delete

# certain directories
find . -name '__pycache__' -type d -exec rm -rf {} +
find . -name '.mypy_cache' -type d -exec rm -rf {} +
find . -name '*.egg-info' -type d -exec rm -rf {} +


# ----- delete pytest artifacts -----

# certain files
find . -name ".coverage" -type f -delete

# certain directories
find . -name '.pytest_cache' -type d -exec rm -rf {} +


# ----- delete other artifacts -----

# macOS cache files
find . -name ".DS_Store" -type f -delete
