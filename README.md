# MinecraftAI

Capstone project: Redstone AI Systems  
University of Missouri - Columbia


## Team

- Luke Domalewski  
- Landis Bargatze  
- Roberto Herrera  
- Brennan Scheel  
- Paul Maschhoff

---

This project explores how to train an AI agent to perform useful tasks in Minecraft using reinforcement learning and behavioral cloning.

We originally set out to train an agent to autonomously mine diamonds. But after running into major limitations with the MineRL dataset and environment support, we pivoted to something more achievable: training an agent to chop down trees.

To do this, we built our own dataset from scratch and developed a full pipeline for data collection, imitation learning, fine-tuning with reinforcement learning, and evaluation all tied together with a simple GUI.

---

## Project Pipeline

**Data Collection**  
- `controller.py` lets a human control the Minecraft agent  
- Actions, observations, and rewards are recorded into custom episodes

**Preprocessing**  
- `data_processing.py` formats recorded episodes into `.npz` files for training  
- Observations include flattened POV images and inventory vectors

**Behavior Cloning (BC)**  
- `train_bc.py`: trains a simple MLP model  
- `train_bc_cnn.py`: trains a CNN model using visual data

**Reinforcement Learning**  
- `train_rppo_cnn.py`: fine-tunes using Recurrent PPO with the BC encoder as a feature extractor

**Evaluation**  
- `evaluate_rppo_cnn.py`: runs trained models inside the MineRL environment and displays real-time feedback

**GUI**  
- `run_pipeline.py` provides a one-click interface to run each part of the system


---

## Technologies Used

- Python 3.10  
- MineRL 1.0.2 (custom install)  
- Stable-Baselines3 + sb3-contrib  
- OpenCV (for image handling)  
- Tkinter (for GUI)  
- WSL or Ubuntu (preferred install environment)
 ( Most packages are installed when installing MineRL in the guides below ) 
 
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

## What Didn’t Go as Planned

- MineRL’s official dataset loaders were deprecated  
- Older versions of MineRL were unstable and poorly documented  
- Installation was painful — Python, Java, and Gradle versions had to be guessed  
- We had to collect all data manually  
- Some teammates needed to run MineRL in VMs or Docker, which caused lag  
- Models took 30–60 minutes to train, and long runs would crash around 18,000 steps

---


## Questions?

Submit issue or if a grader send us an email
