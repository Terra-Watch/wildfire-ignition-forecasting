"""python apply_cube_patch.py CUBE.npy cube_patch.npz -- writes the clean fire-history channels into a copy of the original cube, in place."""
import sys
import numpy as np
cube = np.load(sys.argv[1], mmap_mode="r+"); z = np.load(sys.argv[2]); ch, p = list(z["channels"]), z["patch"]
assert list(cube.shape) == list(z["shape"]), "patch is for a different cube"
for a in range(0, cube.shape[0], 128):
    cube[a:a + 128][:, :, ch] = p[a:a + 128].astype(np.float32)
cube.flush(); print("patched channels", ch)
