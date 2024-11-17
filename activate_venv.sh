#!/bin/bash

# Path of the venv

VENV_PATH="../../.venv311_new/bin/activate"

if [ -f "$VENV_PATH" ]; then
	echo "Activating virtual Environment..."
	source "$VENV_PATH"
	echo "Virtual Environment Activate"
else
	echo "Error: VIrtual environent not found at $VENV_PATH"
	exit 1
fi

