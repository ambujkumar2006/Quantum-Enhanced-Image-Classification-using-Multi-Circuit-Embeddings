import torch
from model import ClassicalBlock
from model import SFEModule
from model import QuantumLayer
from model import QuCNet

# if __name__ == "__main__":
#     x = torch.randn(
#         2,
#         64,
#         32,
#         32
#     )

#     block = ClassicalBlock()
#     print(block)
#     y = block(x)
#     print("Input shape", x.shape )
#     print("Output shape", y.shape )

# if __name__ == "__main__":

#     x = torch.randn(
#         8,
#         3,
#         32,
#         32
#     )

# sfe = SFEModule()

# y = sfe(x)

# print("Input Shape : ", x.shape)
# print("Output Shape: ", y.shape)

# if __name__ == "__main__":

#     x = torch.randn(
#         8,
#         4
#     )

#     qlayer = QuantumLayer()

#     y = qlayer(x)

#     print("Input Shape: ", x.shape)
#     print("Output Shape: ", y.shape)

#     print(y)

# if __name__ == "__main__":
#     x = torch.randn(
#         8,
#         3,
#         32,
#         32
#     )
#     model = QuCNet(
#         num_classses=10
#     )
#     y = model(x)

#     print("Input Shape :", x.shape)
#     print("Output Shape :", y.shape)

#     # print(y.dtype)

model = QuCNet()
x = torch.randn(8,3,32,32)
y = model(x)
print(y.shape)
loss = y.mean()
loss.backward()
print("Backward pass successful")