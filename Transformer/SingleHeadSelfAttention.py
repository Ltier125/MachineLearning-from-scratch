import math
import torch
from torch import nn

# T = 3       # token 数量
# D = 4       # 输入向量维度
# d_k = 2     # Q、K 的维度
# d_v = 4     # V 的维度

# X = torch.randn(T, D) # 3x4

# W_q = torch.randn(D, d_k) #4x2
# W_k = torch.randn(D, d_k) #4x2
# W_v = torch.randn(D, d_v) #4x4

# Q = X @ W_q   # [3, 2]
# K = X @ W_k   # [3, 2]
# V = X @ W_v   # [3, 4]

# # ① 计算所有 token 两两之间的分数，并缩放
# scores = Q @ K.T /math.sqrt(d_k)                    # [3, 3]

# # ② 每个 query 对应一行，让每行权重之和为 1
# weights = torch.softmax(scores, dim=-1) # [3, 3]

# # ③ 根据权重，对 V 的各行做加权求和
# output = weights @ V                         # [3, 4]

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()

        self.q_proj = nn.Linear(d_model, d_k, bias=False)
        self.k_proj = nn.Linear(d_model, d_k, bias=False)
        self.v_proj = nn.Linear(d_model, d_v, bias=False)

        self.scale = math.sqrt(d_k)

    def forward(self, x):
        # x: [B, T, d_model]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = (q @ k.transpose(-2,-1)) / self.scale
        weights = torch.softmax(scores,dim=-1)

        output = weights @ v
        return output, weights


torch.manual_seed(42)

model = SingleHeadSelfAttention(
    d_model=8,
    d_k=4,
    d_v=8,
)

x = torch.randn(2, 5, 8)# 2句话，每句话 5 个 token，每个 token 是 8 维向量

# output, weights =model(x)

# print("输出形状：", output.shape)
# print("权重形状：", weights.shape)
# print("第一句话的注意力权重：\n", weights[0])
# print("每行权重之和：\n", weights.sum(dim=-1))


# 下面尝试让权重更新一次
#首先先查看一个参数
print("Q 投影参数的形状：", model.q_proj.weight.shape)
print("是否需要计算梯度：", model.q_proj.weight.requires_grad)

target = x
output, weights =model(x)

error = output - target
squared_error = error.square()
loss =squared_error.mean()

print("误差形状：", error.shape)
print("平方误差形状：", squared_error.shape)
print("损失形状：", loss.shape)
print("损失数值：",loss.item())

# 清除可能留下的旧梯度
model.zero_grad()

# 从 loss 出发，反向计算各个参数的梯度
loss.backward()

# 查看Q投影向量对应的梯度
print("Q 参数的梯度：")
print(model.q_proj.weight.grad)

print("梯度形状:", model.q_proj.weight.grad.shape)

learning_rate = 0.1

# 保存一个参数更新前的数值，方便检查
before = model.q_proj.weight[0,0].item()
grad = model.q_proj.weight.grad[0,0].item()

# 更新模型的全部参数： Q，K，V三个投影矩阵
with torch.no_grad():
    for parameter in model.parameters():
        # parameter = parameter-learning_rate*parameter.grad 错误写法
        parameter-=learning_rate*parameter.grad

after = model.q_proj.weight[0, 0].item()

print("更新前:", before)
print("这个参数的梯度:", grad)
print("更新后:", after)
print("实际变化:", after - before)
print("预期变化:", -learning_rate * grad)

with torch.no_grad():
    new_output, _ = model(x)
    new_loss = (new_output - target).square().mean()

print("更新前的损失:", loss.item())
print("更新后的损失:", new_loss.item())

scores_zero = torch.tensor([2.0, 0.0, 0.0])
scores_masked = torch.tensor([2.0, float("-inf"), float("-inf")])


