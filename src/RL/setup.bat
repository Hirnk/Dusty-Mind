@echo off
echo Setting up the Mindustry RL Project...

:: Check if the virtual environment exists, if not, create one
IF NOT EXIST venv (
    echo Creating virtual environment...
    py -m venv venv
)

:: Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

:: Install required dependencies from requirements.txt
echo Installing dependencies...
py -m pip install -r requirements.txt

:: Install torch (CPU version by default)
echo Installing PyTorch...
py -m pip install torch torchvision

:: Optional: GPU version of PyTorch (uncomment if you need CUDA)
:: python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

:: Success message
echo Setup complete. You can now run your RL agent training.
pause
