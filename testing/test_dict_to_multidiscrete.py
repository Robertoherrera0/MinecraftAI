import numpy as np
import data_collector_agent as dca

def test_dict_to_multidiscrete_centered():
    action = {k: 0 for k in dca.ACTION_KEYS}
    action["camera"] = np.array([0.0, 0.0], dtype=np.float32)
    vec = dca.dict_to_multidiscrete(action)
    # length = 6 action flags + 2 camera bins
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.int32
    assert vec.shape == (len(dca.ACTION_KEYS) + 2,)
    # first six should be zeros
    assert all(int(v)==0 for v in vec[:len(dca.ACTION_KEYS)])
    # both bins should be CAMERA_CENTER
    c = dca.CAMERA_CENTER
    assert int(vec[len(dca.ACTION_KEYS)])   == c   # yaw
    assert int(vec[len(dca.ACTION_KEYS)+1]) == c   # pitch

def test_dict_to_multidiscrete_extremes():
    # movement flags all on
    action = {k: 1 for k in dca.ACTION_KEYS}
    # extreme camera values
    action["camera"] = np.array([dca.CAMERA_RANGE, -dca.CAMERA_RANGE],
                                dtype=np.float32)
    vec = dca.dict_to_multidiscrete(action)
    # first six should be ones
    assert all(int(v)==1 for v in vec[:len(dca.ACTION_KEYS)])
    idx = len(dca.ACTION_KEYS)
    # yaw = camera[1] = -range → round(-10)+10 = 0
    assert int(vec[idx]) == 0
    # pitch = camera[0] = +range → round(+10)+10 = 20 = CAMERA_BINS-1
    assert int(vec[idx+1]) == dca.CAMERA_BINS-1
