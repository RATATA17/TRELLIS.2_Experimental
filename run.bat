@echo off
cd TRELLIS.2
call venv\scripts\activate.bat
set "ATTN_BACKEND="
set "SPARSE_ATTN_BACKEND="
REM python app.py --use-gradio
REM python app_mod.py --worker-backend inprocess --use-gradio
python app_mod.py --worker-backend separate --worker-refresh-interval 2 --use-gradio
call venv\scripts\deactivate.bat
cd..
