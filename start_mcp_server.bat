@echo off
call C:\Users\pan\anaconda3\Scripts\activate
call conda activate a3X
cd /d "%~dp0"
python -m src.mcp_server
