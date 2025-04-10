import minerl
import gym
import os
import cv2
import numpy as np
from pynput import keyboard
from datetime import datetime

env = gym.make("MineRLObtainDiamondShovel-v0") 

keys_pressed = set()
label_dir = "labeled_data"
os.makedirs(label_dir, exist_ok=True)

# Key mappings to actions
def get_action():
    act = env.action_space.no_op()
    if 'w' in keys_pressed:
        act['forward'] = 1
    if 'a' in keys_pressed:
        act['left'] = 1
    if 'd' in keys_pressed:
        act['right'] = 1
    if ' ' in keys_pressed:
        act['jump'] = 1
    if 's' in keys_pressed:
        act['back'] = 1
    if 'm' in keys_pressed:
        act['attack'] = 1
    return act

# Key press handling
def on_press(key):
    try:
        keys_pressed.add(key.char)
    except AttributeError:
        pass

def on_release(key):
    try:
        keys_pressed.discard(key.char)
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

obs = env.reset()
done = False
step = 0

while True:
    action = get_action()
    obs, reward, done, _ = env.step(action)
    frame = obs["pov"]

    env.render()

    if done:
        obs = env.reset()
