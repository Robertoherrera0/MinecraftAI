# MinecraftAI

Capstone project: Redstone AI Systems  
University of Missouri - Columbia

Capstone project for training reinforcement learning agents in Minecraft using the [MineRL](https://github.com/minerllabs/minerl) framework.

## What This Project Does

The pipeline includes:
- **Behavioral Cloning (BC)** for learning from human keyboard inputs  
- **Recurrent PPO (RPPO)** for fine-tuning the agent's behavior  
- **Custom reward shaping** to guide learning (e.g., logs collected, camera control, health penalties)  
- A simple **GUI launcher** to collect data, train models, and evaluate results

You’ll be able to collect data, train an agent, and run evaluations — all from one UI.

---

## Setup Guides

**Installing MineRL on WSL (via Google Colab notebook)**  
➡️ https://colab.research.google.com/drive/1IUSLeHmbyKUxkNX9qvjPYGkryU15gsao?usp=sharing

**Installing MineRL on Windows using Bash (via Google Doc)**  
➡️ https://docs.google.com/document/d/1C0SI3YShkjO1t-yqURLVF_7Uj5ynvLqrDkMA8uZinqw/edit?usp=sharing

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
git clone https://github.com/Robertoherrera0/MinecraftAI.git
cd minecraftai
python run_pipeline.py
```

This launches a simple UI where you can:
- Collect demonstration data
- Train a BC model
- Fine-tune with RL
- Evaluate your trained agent

---

## Questions?

Submit issue or if a grader send us an email
