# sample-browser

sound-similarity sample browser via [CLAP](https://github.com/LAION-AI/CLAP) embeddings cosine distance

 requires ~4.2 GB VRAM to run (you will need a GPU)

## Install:

Download this repo, unzip it into a folder. Make sure python is installed on your system. I used python 3.10. Then:

to run after first install, run `run.bat` (win) or `./run.sh` (mac).

### Windows:
**run `install.bat`.** you will need [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64) installed.

### Mac:
navigate to the folder, open it in terminal, and **run: `chmod +x install.sh` then `./install.sh`**

### Other OSs:
create a venv `python -m venv venv`, enter it `source venv/Scripts/activate`, run `pip install flask laion_clap librosa numpy torch`, make sure the CUDA version of torch is installed `pip install -U torch==2.4.0+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`, then run `python sound-similarity-browser.py` and go to http://localhost:5000/

## Usage:

Paste a local filepath into the Cache Management input and press Process Folder. When complete, upload a sound (use upload button or drag-drop an audio file) or type a sound description and press search.

![screenshot of sample browser showing a list of matching samples](demos/inference.png)

![screenshot of sample browser showing a progress bar caching audio latents](demos/caching.png)
