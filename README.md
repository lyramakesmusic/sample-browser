# sample-browser
sound-similarity sample browser via [CLAP](https://github.com/LAION-AI/CLAP) embeddings cosine distance

code is multiplatform, but installer and run file are windows-only

### Install:

download this repo, unzip it into a folder, and run `install.bat`. you will need [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64) and python installed.

to run after first install, run `run.bat`. it will open localhost for you

(other OSs: create a venv `python -m venv venv`, enter it, run `pip install flask laion_clap librosa numpy torch`, make sure the CUDA version of torch is installed, then run `python sound-similarity-browser.py` and go to http://localhost:5000/)

### Usage:

Paste a local filepath into the Cache Management input and press Process Folder. When complete, upload a sound or type a sound description and press search

![screenshot of sample browser showing a list of matching samples](inference.png)

![screenshot of sample browser showing a progress bar caching audio latents](caching.png)
