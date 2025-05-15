import sys, os, types
import numpy as np

# ensure project root is on PYTHONPATH so `import data_collector_agent` works
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# stub out heavy external deps so the module can import
sys.modules.setdefault('minerl', types.ModuleType('minerl'))
sys.modules.setdefault('gym',  types.ModuleType('gym'))

# stub cv2.resize to just return an array of the right shape/dtype
cv2 = types.ModuleType('cv2')
def resize(img, size):
    h, w = size[1], size[0]
    c = img.shape[2]
    return np.zeros((h, w, c), dtype=img.dtype)
cv2.resize = resize
sys.modules['cv2'] = cv2

# stub pynput.keyboard.Listener so it doesn't actually spin up a listener
pynput = types.ModuleType('pynput')
keyboard = types.ModuleType('pynput.keyboard')
class Listener:
    def __init__(self, on_press=None, on_release=None):
        pass
    def start(self): 
        pass
keyboard.Listener = Listener
pynput.keyboard = keyboard
sys.modules['pynput'] = pynput
sys.modules['pynput.keyboard'] = keyboard

# stub CustomRewardWrapper
choptree = types.ModuleType('choptree')
wrappers = types.ModuleType('choptree.wrappers')
crw = types.ModuleType('choptree.wrappers.custom_reward_wrapper')
crw.CustomRewardWrapper = object
wrappers.custom_reward_wrapper = crw
choptree.wrappers = wrappers
sys.modules['choptree'] = choptree
sys.modules['choptree.wrappers'] = wrappers
sys.modules['choptree.wrappers.custom_reward_wrapper'] = crw
