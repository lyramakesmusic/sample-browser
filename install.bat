@echo off

python -m venv venv
call "venv/Scripts/activate"
pip install flask laion_clap librosa numpy torch
pip install -U torch==2.4.0+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo @echo off > run.bat
echo call "venv/Scripts/activate" >> run.bat
echo python sound-similarity-browser.py >> run.bat
echo start http://localhost:5000/ >> run.bat

call run.bat