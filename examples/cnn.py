import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jdl import Tensor
from jdl.nn import Conv2d, get_model_params
import numpy as np

class Model:
    def __init__(self):
        self.l1 = Conv2d(1, 1, 3)
    def __call__(self, x):
        return self.l1(x)

X = Tensor(np.random.random(size=(1,4,4,1)))

model = Model()
y = model(X)
print(y.data)
