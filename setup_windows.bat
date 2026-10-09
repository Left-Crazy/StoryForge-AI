@echo off
setlocal
cd /d "%~dp0"
if not exist .venv (
  py -3.11 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Choose an AI runtime in the StoryForge sidebar:
echo   Ollama: use a model already installed by Ollama, e.g. gemma3:4b
echo   GGUF: point the app to a standalone .gguf model file

streamlit run app.py
endlocal
