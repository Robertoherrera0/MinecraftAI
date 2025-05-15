import tkinter as tk
from tkinter import PhotoImage
import subprocess
import os
from PIL import Image, ImageTk

# Launch a script in a new process (non-blocking)
def run_script(script_path):
    print(f"Running: {script_path}")
    subprocess.Popen(["python", script_path])

# Run data collection and preprocessing in sequence
def collect_data_pipeline():
    subprocess.run(["python", "data_collector_agent.py"])
    subprocess.run(["python", "data_processing.py"])

# Initialize main GUI window
root = tk.Tk()
root.title("Minecraft AI Pipeline")
root.geometry("400x520")
root.resizable(False, False)

# Load and display logo image
image_path = os.path.join(os.path.dirname(__file__), "image/logo.png")
logo_img = Image.open(image_path)
logo_img = logo_img.resize((300, 200))
logo = ImageTk.PhotoImage(logo_img)
tk.Label(root, image=logo).pack(pady=10)

# Heading text
tk.Label(root, text="Select an action", font=("Arial", 14)).pack(pady=5)

# Buttons to trigger different stages of the pipeline
tk.Button(root, text="Collect Data", width=25,
          command=collect_data_pipeline).pack(pady=6)

tk.Button(root, text="Train BC Model", width=25,
          command=lambda: run_script("bc_model/train_bc.py")).pack(pady=6)

tk.Button(root, text="Train RL Model", width=25,
          command=lambda: run_script("choptree/training_rl/train_rppo_cnn.py")).pack(pady=6)

tk.Button(root, text="Evaluate Agent", width=25,
          command=lambda: run_script("choptree/evaluate_models/evaluate_rppo_cnn.py")).pack(pady=6)

# Start the GUI loop
root.mainloop()
