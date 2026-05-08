@echo off
REM Regenerate mock JSON files used by the demo mode.
REM Usage:
REM   regen_mock.bat                  (pure p=3, N=40, k=100, seed=42)
REM   regen_mock.bat --model mixed    (mixed 2+3 spin)
REM   regen_mock.bat --p 4 --N 60    (pure p=4, N=60)

python tools/mock_run.py frontend/public/mock_run.json %*
python tools/mock_run.py frontend/public/mock_mixed.json --model mixed
