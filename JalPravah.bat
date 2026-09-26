@echo off
rem JalPravah: local web app for the SPH and Delft3D dam-break results
cd /d "%~dp0"
python jalpravah.py %*
pause
