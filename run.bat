@echo off
call venv\scripts\activate.bat
python app.py  --use-gradio
call venv\scripts\deactivate.bat
cd..
