# NVIDIA GPU & Docker Setup Guide

This guide details how to configure a Linux host (e.g., Ubuntu) with NVIDIA GPUs (like the RTX 4090) to run this project with hardware acceleration.

## 1. Prerequisites: NVIDIA Drivers
Ensure you have the official NVIDIA drivers installed on your host machine.
```bash
nvidia-smi
```
If this command fails, you must install the drivers first (e.g., `sudo apt install nvidia-driver-550`).

## 2. Install NVIDIA Container Toolkit
The toolkit is required for Docker to "see" and use your GPUs.

### Step A: Setup the Repository
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
*Note: If asked to overwrite the keyring, choose **y**.*

### Step B: Install the Package
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

## 3. Configure Docker Runtime
Register the NVIDIA runtime with Docker and restart the daemon.

```bash
# Configure Docker to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker to apply changes
sudo systemctl restart docker
```

## 4. Verification
Run a test container to verify that Docker can access the GPUs.

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
You should see a table showing your RTX 4090 GPUs.

## 5. Running the Project
Once configured, you can use the provided runner script which handles device detection automatically:

```bash
chmod +x run.sh
./run.sh
```

Or run via Docker Compose directly:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

---

### Troubleshooting
- **"nvidia-ctk: command not found"**: The installation in Step 2 failed or didn't finish.
- **"could not select device driver"**: Ensure you restarted Docker after running the configuration command.
- **Permission Denied**: Ensure your user is in the `docker` group: `sudo usermod -aG docker $USER` (requires logout/login).
