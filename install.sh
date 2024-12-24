#!/bin/bash

# Create and activate virtual environment
python3.10 -m venv --prompt samplebrowser venv
source venv/bin/activate

# Install base requirements
pip install --upgrade pip setuptools wheel packaging
pip install flask laion_clap librosa

# Install PyTorch with MPS support (nightly version for best M1/M2/M3 support)
pip install torch==2.4.0 torchvision torchaudio

# Create run script
cat > run.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python3 sound-similarity-browser.py &
PID=$!  # Capture the background process ID
sleep 15
open http://127.0.0.1:5000/
# Wait for Ctrl+C
trap "kill $PID" INT
wait $PID
EOF

# Make run script executable
chmod +x run.sh

# Run the application
./run.sh