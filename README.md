# MinecraftAI

Capstone project for training reinforcement learning agents in Minecraft using the [MineRL](https://github.com/minerllabs/minerl) framework.

## What This Project Does

This repo pulls together:
- Behavioral Cloning (BC) from human key inputs
- Recurrent PPO (RPPO) for fine-tuning the agent
- Custom reward shaping
- A full training + evaluation pipeline via `run_pipeline.py` (name can be changed if needed)

You’ll be able to collect data, train an agent, and run evaluations — all from one UI.

---

## Installation (MineRL on Ubuntu using Miniconda)

If you're running Windows, open PowerShell and type:
```bash
wsl --install
```

Once inside Ubuntu, follow these steps:

### 1. Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Then:
```bash
conda init
# If that fails:
~/miniconda3/bin/conda init
```

Confirm conda is working:
```bash
conda --version
```

### 2. Set Up Your Project

```bash
mkdir minecraftai
cd minecraftai
```

Make a `requirements.txt`:
```txt
gym>=0.19.0,<=0.23.1
opencv-python>=4.1.0.25
setuptools>=49.2.0
tqdm>=4.32.2
numpy==1.24
requests>=2.20.0
ipython>=7.5.0
typing>=3.6.6
lxml>=4.3.3
psutil>=5.6.2
Pyro4>=4.76
coloredlogs>=10.0
pillow>=8.0.0
dill>=0.3.1.1
daemoniker>=0.2.3
xmltodict==0.12.0
inflection>=0.3.1
jinja2>=2.11.2
imagehash>=4.0.0
flaky
pyglet
```

### 3. Set Up Conda Environment

```bash
conda create -n minerl python=3.10
conda activate minerl
```

### 4. Install Java 8

```bash
sudo apt update && sudo apt install openjdk-8-jdk
```

Then:
```bash
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
```

Verify:
```bash
java -version
python --version  # should be 3.10
```

### 5. Install Everything

```bash
sudo apt install build-essential libncurses5-dev libncursesw5-dev
pip install getch
pip install -r requirements.txt
pip install git+https://github.com/minerllabs/minerl
```

---

## Running the Pipeline

Once installed, just clone the repo and run the pipeline:

```bash
git clone https://github.com/your_username/minecraftai.git
cd minecraftai
python run_pipeline.py
```

This launches a simple UI where you can:
- Collect demonstration data
- Train a BC model
- Fine-tune with RL
- Evaluate your trained agent

---

## Optional: Google Colab Setup

We also created a Colab notebook to make it easier to install MineRL dependencies (especially useful for testing or demos).

➡️ [**Minerl_Installation.ipynb**](link-to-colab-if-public)

---

## Questions?

Ping us on Discord or submit an issue in the repo.
