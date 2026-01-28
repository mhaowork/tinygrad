from tinygrad import Tensor

print(Tensor.empty(4, 4).sum(1).realize().numpy())
