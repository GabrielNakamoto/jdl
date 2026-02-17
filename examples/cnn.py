import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jdl import Tensor

x = Tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
k = Tensor([[0.0, 1.0], [2.0, 3.0]])
y = x.corr2d(k)

print(y.data)
