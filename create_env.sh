#!/bin/bash

# Name of the virtual environment directory
VENV_NAME=".prompt-opt"

# Python packages to include in requirements.txt
PACKAGES=(
    "datasets"
    "python-dotenv"
    "anthropic"
    "regex"     # For GSM8K answer extraction
    "numpy"
    "pandas"
)

# --- Script Start ---

echo "Setting up Python environment in current directory..."

# 1. Check if we are in a directory (sanity check)
if [ ! -d "." ]; then
    echo "Error: Please run this script from within your project directory."
    exit 1
fi

# 2. Create Python virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating Python virtual environment '$VENV_NAME'..."
    python3 -m venv "$VENV_NAME"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment. Make sure python3 and venv are installed."
        exit 1
    fi
    echo "Virtual environment '$VENV_NAME' created."
else
    echo "Virtual environment '$VENV_NAME' already exists."
fi

# 3. Create requirements.txt
echo "Creating/updating requirements.txt..."
REQUIREMENTS_FILE="requirements.txt"
touch "$REQUIREMENTS_FILE" # Clear the file or create it if it doesn't exist

for pkg in "${PACKAGES[@]}"; do
    echo "$pkg" >> "$REQUIREMENTS_FILE"
done
echo "requirements.txt created with the following packages:"
cat "$REQUIREMENTS_FILE"

# 4. Print activation and installation instructions
echo ""
echo "---------------------------------------------------------------------"
echo "Environment Setup Complete!"
echo ""
echo "Next Steps:"
echo "1. Activate the virtual environment:"
echo "   source \"$VENV_NAME/bin/activate\""  # Quoted to handle the dot
echo ""
echo "2. Install the required packages (if you haven't already):"
echo "   pip install -r requirements.txt"
echo ""
echo "3. To deactivate the virtual environment later, simply type:"
echo "   deactivate"
echo "---------------------------------------------------------------------"

# --- Script End ---