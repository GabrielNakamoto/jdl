# https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jdl import Tensor
from jdl.nn import Conv2d, Linear, BatchNorm

class Model:
    def __init__(self):
        # Conv block 1
        self.l1 = Conv2d(1, 32, 5)
        self.l2 = Conv2d(32, 32, 5)
        self.l3 = BatchNorm(32)

        # Conv block 2
        self.l4 = Conv2d(32, 64, 3)
        self.l5 = Conv2d(64, 64, 3)
        self.l6 = BatchNorm(64)

        # Fully-connected
        self.l7 = Linear(576, 10)
    def __call__(self, x):
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        x = self.l3(x)

        x = self.l4(x).relu()
        x = self.l5(x).relu()
        x = self.l6(x).flatten()
        return self.l7(x)


for epoch in range(20):
    pass
