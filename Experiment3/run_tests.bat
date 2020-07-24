@echo off
set TEST_DIR=%cd%\test_scripts

for %%i in (%TEST_DIR%\test*.py) do (
    echo RUNNING %%i
    python %%i
) 