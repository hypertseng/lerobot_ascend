# import mindspore as ms
# import mindspore.numpy as mnp

# class Beta:
#     """A minimal beta distribution stub used for Ascend."""
#     def __init__(self, alpha, beta):
#         self.alpha = alpha
#         self.beta = beta

#     def sample(self, shape=None):
#         # 使用 Gamma 分布实现 Beta 分布：X ~ Gamma(a), Y ~ Gamma(b), return X / (X + Y)
#         x = mnp.random.gamma(self.alpha, 1.0, shape)
#         y = mnp.random.gamma(self.beta, 1.0, shape)
#         return x / (x + y)