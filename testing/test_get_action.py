import numpy as np
import pytest
import data_collector_agent as dca

def test_get_action_no_keys():
    dca.keys_pressed.clear()
    act = dca.get_action()
    # all movement / attack / jump should be 0
    assert all(act[k] == 0 for k in dca.ACTION_KEYS)
    # camera should be zero float32 vector
    assert isinstance(act["camera"], np.ndarray)
    assert act["camera"].dtype == np.float32
    np.testing.assert_array_equal(act["camera"],
                                  np.array([0.0, 0.0], dtype=np.float32))

def test_get_action_all_keys_and_camera():
    # press every movement key + camera keys
    dca.keys_pressed.clear()
    for key in ['w','a','s','d','b','m','j','l','i','k']:
        dca.keys_pressed.add(key)
    act = dca.get_action()
    # movement flags
    assert act["forward"] == 1
    assert act["back"]    == 1
    assert act["left"]    == 1
    assert act["right"]   == 1
    assert act["jump"]    == 1
    assert act["attack"]  == 1
    # j,l cancel yaw, i,k cancel pitch → both back to 0
    np.testing.assert_allclose(act["camera"], [0.0, 0.0], atol=1e-6)
