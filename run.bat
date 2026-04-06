@echo off
call venv\scripts\activate.bat

@REM If artifacts appear, "1" is recommended.
set TRELLIS_RENDER_SYNC_BEFORE_CPU=1

@REM Log levels: "quiet", "normal"(default value), "verbose", "debug"
set TRELLIS_LOG_LEVEL=normal

python app.py  --use-gradio
call venv\scripts\deactivate.bat
cd..
