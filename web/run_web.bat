@echo off
rem ============================================================
rem  AgriVision Web Launcher
rem  KEY: force Python UTF-8 mode (PYTHONUTF8=1)
rem  WHY: model config file is UTF-8 with Chinese comments;
rem       PaddleDetection reads it via open() without encoding,
rem       which defaults to GBK on Chinese Windows and fails with
rem       "'gbk' codec can't decode byte ..." during model load.
rem ============================================================
chcp 65001 >nul
setlocal

rem Move to project root (this script lives in web\, one level up = root)
cd /d "%~dp0.."

rem Force UTF-8 mode (MUST be set before launching Python)
set PYTHONUTF8=1

rem Pass through launch args, e.g.:
rem   run_web.bat --device gpu --port 7860
rem Defaults: auto / 7860 ; public share via env GRADIO_SHARE=true
echo [launch] project dir: %CD%
echo [launch] UTF-8 mode: ON (PYTHONUTF8=1)
echo [launch] command: python web\app.py %*
echo ------------------------------------------------------------

python web\app.py %*

if errorlevel 1 (
    echo.
    echo [launch failed] see error above.
    pause
)

endlocal
