#!/bin/bash

# --- Configuration ---
VENV_NAME="venv"
MODULE_DIR="key_manager_module"
WHEEL_FILE="dist/api_key_rotater-0.1.0-py3-none-any.whl"

echo "Starting deployment script..."

# 1. Check and Activate Virtual Environment
# Check if VIRTUAL_ENV is set (standard way to check if a venv is active)
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment not active. Attempting to activate '$VENV_NAME'..."

    # Check for the common activation script path
    VENV_ACTIVATE_SCRIPT="./$VENV_NAME/bin/activate"

    if [[ -f "$VENV_ACTIVATE_SCRIPT" ]]; then
        # Use 'source' or '.' to execute the activation script in the current shell context
        source "$VENV_ACTIVATE_SCRIPT"
        
        # Verify activation (optional, but good practice)
        if [[ -n "$VIRTUAL_ENV" ]]; then
            echo "Successfully activated virtual environment."
        else
            echo "Error: Activation script ran, but VIRTUAL_ENV is still not set. Aborting."
            exit 1
        fi
    else
        echo "Error: Cannot find activation script at '$VENV_ACTIVATE_SCRIPT'. Have you run 'python3 -m venv $VENV_NAME'?"
        exit 1
    fi
else
    echo "Virtual environment is already active: $VIRTUAL_ENV"
fi


# 2. Change Directory into the module folder
echo "Changing directory to '$MODULE_DIR'..."
if [[ -d "$MODULE_DIR" ]]; then
    cd "$MODULE_DIR"
else
    echo "Error: Directory '$MODULE_DIR' not found. Aborting."
    exit 1
fi


# 3. Run Python Build command
echo "Running 'python -m build' to create the distribution..."
python -m build

if [[ $? -ne 0 ]]; then
    echo "Error: 'python -m build' failed. Aborting."
    cd .. # Go back to original directory before exiting
    exit 1
fi


# 4. Install the wheel file using uv
echo "Installing wheel using 'uv pip install $WHEEL_FILE'..."
uv pip install "$WHEEL_FILE"

if [[ $? -ne 0 ]]; then
    echo "Error: 'uv pip install' failed."
    cd ..
    exit 1
fi

echo "--- Deployment Complete ---"
echo "Module built and installed successfully."

cd.. # Return to the original directory

# chmod +x install_key_manager.sh