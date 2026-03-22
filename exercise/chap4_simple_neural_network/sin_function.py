import numpy as np
import matplotlib.pyplot as plt

# 定义神经网络的底层基础组件
class Linear:
    """全连接线性层 (y = xW + b)"""
    def __init__(self, in_features, out_features):
        # 初始化权重 W 和偏置 b (使用正态分布初始化，乘以 0.1 减小初始方差)
        self.W = np.random.randn(in_features, out_features) / np.sqrt(in_features)
        self.b = np.zeros((1, out_features))
        self.mem = {} # 用于存储前向传播时的 x，方便反向传播算梯度

    def forward(self, x):
        self.mem['x'] = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_y):
        x = self.mem['x']
        # 计算对输入 x 的梯度，传给上一层
        grad_x = np.dot(grad_y, self.W.T)
        # 计算对权重 W 的梯度
        self.grad_W = np.dot(x.T, grad_y)
        # 计算对偏置 b 的梯度 (按列求和)
        self.grad_b = np.sum(grad_y, axis=0, keepdims=True)
        return grad_x

    def step(self, learning_rate):
        # 梯度下降更新参数
        self.W -= learning_rate * self.grad_W
        self.b -= learning_rate * self.grad_b


class ReLU:
    """ReLU 激活函数"""
    def __init__(self):
        self.mem = {}

    def forward(self, x):
        self.mem['x'] = x
        return np.maximum(0, x)

    def backward(self, grad_y):
        x = self.mem['x']
        # 当 x > 0 时，梯度原样传递；否则梯度为 0
        grad_x = grad_y * (x > 0)
        return grad_x


class MSELoss:
    """均方误差损失函数 (Mean Squared Error)"""
    def forward(self, y_pred, y_true):
        self.mem = {'y_pred': y_pred, 'y_true': y_true}
        # 计算 MSE: mean((y_pred - y_true)^2)
        return np.mean((y_pred - y_true) ** 2)

    def backward(self):
        y_pred = self.mem['y_pred']
        y_true = self.mem['y_true']
        N = y_pred.shape[0]
        # MSE 的导数公式: 2 * (y_pred - y_true) / N
        return 2.0 * (y_pred - y_true) / N


# 搭建两层 ReLU 神经网络模型
class TwoLayerNet:
    """两层神经网络，用于拟合任意函数"""
    def __init__(self, hidden_size=64):
        # 输入维度是1 (x)，隐藏层维度默认64，输出维度是1 (y)
        self.fc1 = Linear(1, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, 1)

    def forward(self, x):
        # 前向传播：Linear -> ReLU -> Linear
        h1 = self.fc1.forward(x)
        a1 = self.relu.forward(h1)
        y_pred = self.fc2.forward(a1)
        return y_pred

    def backward(self, grad_loss):
        # 反向传播：按相反顺序逐层求导
        grad_a1 = self.fc2.backward(grad_loss)
        grad_h1 = self.relu.backward(grad_a1)
        self.fc1.backward(grad_h1)

    def update(self, learning_rate):
        # 更新每一层的参数
        self.fc1.step(learning_rate)
        self.fc2.step(learning_rate)


# 生成数据、训练模型与结果可视化
# 1. 定义目标函数及数据采集
def target_function(x):
    # y = sin(x)
    return np.sin(x)

# 生成 500 个自变量 x，范围在 -pi 到 pi 之间
X_all = np.linspace(-np.pi, np.pi, 500).reshape(-1, 1)
# 计算真实的 y 值，并加入一点高斯噪声
Y_all = target_function(X_all) + np.random.normal(0, 0.05, size=X_all.shape)

# 打乱数据并划分为训练集 (80%) 和测试集 (20%)
indices = np.random.permutation(500)
train_idx, test_idx = indices[:400], indices[400:]

X_train, Y_train = X_all[train_idx], Y_all[train_idx]
X_test, Y_test = X_all[test_idx], Y_all[test_idx]

# 2. 实例化模型与损失函数
model = TwoLayerNet(hidden_size=100) # 使用100个隐藏层神经元增强拟合能力
criterion = MSELoss()

# 3. 训练循环
learning_rate = 0.01  # 学习率设置较小，保证训练稳定；如果过大可能会发散
epochs = 8000
loss_history = []

for epoch in range(epochs):
    # 前向传播
    Y_pred = model.forward(X_train)
    # 计算损失
    loss = criterion.forward(Y_pred, Y_train)
    loss_history.append(loss)
    
    # 反向传播
    grad_loss = criterion.backward()
    model.backward(grad_loss)
    
    # 更新参数
    model.update(learning_rate)
    
    # 每 500 次迭代打印一次进度
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# 4. 在测试集上验证并计算最终 Loss
Y_test_pred = model.forward(X_test)
test_loss = criterion.forward(Y_test_pred, Y_test)
print(f"Final Test MSE Loss: {test_loss:.4f}")

# 5. 可视化拟合效果 (画图)
plt.figure(figsize=(12, 5))

# 图 1：Loss 下降曲线
plt.subplot(1, 2, 1)
plt.plot(loss_history, color='blue')
plt.title("Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")

# 图 2：函数拟合效果对比
plt.subplot(1, 2, 2)
# 画出全量数据的真实平滑曲线
plt.plot(X_all, target_function(X_all), color='green', label='True Function', linewidth=2)
# 画出带有噪声的训练集散点
plt.scatter(X_train, Y_train, color='gray', s=10, alpha=0.5, label='Train Data (Noisy)')
# 画出神经网络预测的曲线
Y_pred_all = model.forward(X_all)
plt.plot(X_all, Y_pred_all, color='red', label='Neural Net Prediction', linestyle='dashed', linewidth=2)

plt.title("Function Approximation via 2-Layer ReLU Net")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
plt.show()