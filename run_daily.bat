@echo off
REM Daily foolwatch run. Register with Task Scheduler (see README).
cd /d "%~dp0"
py -m foolwatch daily
