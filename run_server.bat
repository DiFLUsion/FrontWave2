@echo off
set PY_EXE=C:\Users\Pporras\AppData\Local\anaconda3\envs\geocondapy\python.exe
set PROJ_LIB=C:\Users\Pporras\AppData\Local\anaconda3\envs\geocondapy\Library\share\proj

set GDAL_DATA=C:\Users\Pporras\AppData\Local\anaconda3\envs\geocondapy\Library\share\gdal
set PATH=C:\Users\Pporras\AppData\Local\anaconda3\envs\geocondapy\Library\bin;%PATH%
"%PY_EXE%" -m pip install --upgrade pip
"%PY_EXE%" -m pip install -r requirements.txt
set PROJ_LIB=%PROJ_LIB%
"%PY_EXE%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
