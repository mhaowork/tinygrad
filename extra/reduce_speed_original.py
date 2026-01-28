import numpy as np

from tinygrad import Tensor

a = Tensor(np_array:=np.random.default_rng().random((4096, 4096), dtype=np.float32)).realize()

out = a.sum().realize()

print(out)