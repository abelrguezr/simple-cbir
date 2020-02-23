import pickle
import os
import numpy as np
import errno


def save_pickle(path, obj):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    print('Saving object in: ' + path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    print('Loading object from: ' + path)
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def reshape_data(data):
    nsamples, nx, ny, c = data.shape
    data2d = data.reshape(nsamples, nx * ny * c)

    return data2d


def apply_func_to_data(func, data, mode='v', **kwargs):
    results = np.array([])
    if mode=='h':
        stack=np.hstack
    else:
        stack = np.vstack
    for x in data:
        result = func(x, **kwargs)
        results = stack([results, result]) if results.size else result

    return results
