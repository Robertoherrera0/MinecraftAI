import numpy as np
import data_collector_agent as dca

def test_flatten_observation_default_inventory():
    # make a dummy 64×64×3 image so cv2.resize is a no‐op
    pov = np.zeros((64,64,3), dtype=np.uint8)
    obs = {"pov": pov}
    flat = dca.flatten_observation(obs)
    # pov should stay 64×64×3 uint8
    assert flat["pov"].shape == (64,64,3)
    assert flat["pov"].dtype == np.uint8
    # inv vector length == len(INVENTORY_KEYS) == 1, float32 zeros
    assert flat["inv"].shape == (len(dca.INVENTORY_KEYS),)
    assert flat["inv"].dtype == np.float32
    assert flat["inv"][0] == 0.0

def test_flatten_observation_with_inventory_value():
    pov = np.zeros((64,64,3), dtype=np.uint8)
    # give it a nonzero inventory count
    obs = {"pov": pov, "inventory": {"log": 7}}
    flat = dca.flatten_observation(obs)
    assert flat["inv"][0] == 7.0
