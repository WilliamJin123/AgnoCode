# install_key_manager.ps1

# --- Configuration ---
$VenvName = ".venv"
$ModuleDir = "key_manager_module"
$WheelFile = "dist/api_key_rotater-0.1.0-py3-none-any.whl"

Write-Host "Starting deployment script..."

# 1. Check and Activate Virtual Environment
if (-not (Get-Variable -Name VIRTUAL_ENV -ErrorAction SilentlyContinue)) {
    Write-Host "Virtual environment not active. Attempting to activate '$VenvName'..."

    # Standard PowerShell activation script path
    $VenvActivateScript = ".\$VenvName\Scripts\activate"

    if (Test-Path $VenvActivateScript) {
        # Use the dot-sourcing operator (.) to run the script in the current scope
        . $VenvActivateScript

        if (Get-Variable -Name VIRTUAL_ENV -ErrorAction SilentlyContinue) {
            Write-Host "Successfully activated virtual environment."
        } else {
            Write-Host "Error: Activation script ran, but VIRTUAL_ENV is still not set. Aborting."
            exit 1
        }
    } else {
        Write-Host "Error: Cannot find activation script at '$VenvActivateScript'. Have you run 'python3 -m venv $VenvName'?"
        exit 1
    }
} else {
    Write-Host "Virtual environment is already active: $env:VIRTUAL_ENV"
}

# 2. Change Directory into the module folder
Write-Host "Changing directory to '$ModuleDir'..."
if (Test-Path $ModuleDir -PathType Container) {
    Set-Location $ModuleDir
} else {
    Write-Host "Error: Directory '$ModuleDir' not found. Aborting."
    exit 1
}

# 3. Run Python Build command
Write-Host "Running 'python -m build' to create the distribution..."
python -m build

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: 'python -m build' failed. Aborting."
    Set-Location .. # Go back before exiting
    exit 1
}

# 4. Install the wheel file using uv
Write-Host "Installing wheel using 'uv pip install $WheelFile'..."
uv pip install $WheelFile

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: 'uv pip install' failed."
    Set-Location ..
    exit 1
}

Write-Host "--- Deployment Complete ---"
Write-Host "Module built and installed successfully."

Set-Location .. # Return to the original directory

