# main.py — JAX 入门：函数式深度学习框架实战
# 依赖：jax>=0.4, jaxlib, optax, scikit-learn
# 安装：pip install jax jaxlib optax scikit-learn
# 对应课程：阶段 03 · 12（JAX 入门）

import time
import jax
import jax.numpy as jnp
from jax import random
import optax

# ============================================================
# 第 1 部分：jax.grad — 函数式自动微分
# ============================================================

def demo_grad():
    """演示 jax.grad 的基础用法：梯度就是函数的一种变换。"""
    print("=" * 60)
    print("第 1 部分：jax.grad — 函数式自动微分")
    print("=" * 60)

    # 定义一个纯函数：输入一个标量，返回其立方
    def f(x):
        return x ** 3

    # jax.grad 将 f 变换为它的导函数 f'
    df = jax.grad(f)

    # 链式调用：对导函数再求导，得到二阶导数 f''
    d2f = jax.grad(df)

    x_val = 2.0
    print(f"f(x)  = x**3")
    print(f"f({x_val})   = {f(x_val)}")       # 8.0
    print(f"f'({x_val})  = {df(x_val)}")       # 12.0 (3 * 2**2)
    print(f"f''({x_val}) = {d2f(x_val)}")      # 12.0 (6 * 2)
    print()

    # 多参数求导：指定对哪个参数求导
    def linear(w, b, x):
        """线性函数：y = w * x + b"""
        return w * x + b

    # argnums=0 表示对第一个参数 w 求偏导
    grad_w = jax.grad(linear, argnums=0)
    w_val, b_val, x_val = 3.0, 1.0, 2.0
    print(f"linear(w,b,x) = w*x + b")
    print(f"∂/∂w at w={w_val} = {grad_w(w_val, b_val, x_val)}")  # x_val = 2.0

    # jax.value_and_grad 同时返回函数值和梯度（更高效）
    loss_and_grad = jax.value_and_grad(linear, argnums=0)
    loss_val, grad_val = loss_and_grad(w_val, b_val, x_val)
    print(f"loss={loss_val}, gradient={grad_val}")
    print()


# ============================================================
# 第 2 部分：jax.vmap — 自动向量化
# ============================================================

def demo_vmap():
    """演示 jax.vmap：写单样本函数，自动获得批次版本。"""
    print("=" * 60)
    print("第 2 部分：jax.vmap — 自动向量化")
    print("=" * 60)

    key = random.PRNGKey(42)
    k1, k2 = random.split(key)

    # 参数是简单的 pytree
    params = {
        'w': random.normal(k1, (4,)),  # 4 维权重
        'b': 0.0,                      # 标量偏置
    }

    # 只处理一个样本的函数
    def predict_single(params, x):
        """单个样本的预测函数。

        Args:
            params: 包含 'w' 和 'b' 的字典
            x: 形状 (4,) 的输入向量

        Returns:
            标量预测值
        """
        return jnp.dot(params['w'], x) + params['b']

    # 生成 8 个样本的批次
    batch_x = random.normal(k2, (8, 4))

    # vmap 将其提升为批次函数
    # in_axes=(None, 0) 表示 params 不批量（共享），x 沿第 0 维批量
    batch_predict = jax.vmap(predict_single, in_axes=(None, 0))

    results = batch_predict(params, batch_x)

    print(f"单样本输入形状: (4,)")
    print(f"批次输入形状:   {batch_x.shape}")
    print(f"预测结果形状:   {results.shape}")  # (8,)
    print(f"预测结果:       {results}")
    print()

    # vmap 与 grad 可以组合：一次性计算每个样本的梯度
    def loss_fn(params, x, y):
        pred = predict_single(params, x)
        return (pred - y) ** 2

    # 为批次中的每个样本独立计算梯度
    per_sample_grads = jax.vmap(
        jax.grad(loss_fn), in_axes=(None, 0, 0)
    )(params, batch_x, jnp.zeros(8))

    print(f"per-example 梯度形状: {jax.tree.map(lambda x: x.shape, per_sample_grads)}")
    print(f"per-example 梯度 w 的第一行: {per_sample_grads['w'][0]}")
    print()


# ============================================================
# 第 3 部分：jax.jit — 即时编译
# ============================================================

def demo_jit():
    """演示 jax.jit 的加速效果。"""
    print("=" * 60)
    print("第 3 部分：jax.jit — 即时编译")
    print("=" * 60)

    key = random.PRNGKey(0)
    x = random.normal(key, (1000, 1000))

    def heavy_computation(x):
        """一个计算密集型函数：矩阵乘法 + 归一化，重复多次。"""
        for _ in range(10):
            x = jnp.dot(x, x)
            # 用 L2 范数归一化，防止数值溢出
            x = x / jnp.linalg.norm(x)
        return x

    # 编译版本：第一次调用会触发 JIT 编译
    fast_fn = jax.jit(heavy_computation)

    # 预热：第一次调用包含编译开销
    print("正在 JIT 编译...")
    _ = fast_fn(x)
    print("编译完成。")
    print()

    # 非编译版本的基准测试
    start = time.perf_counter()
    for _ in range(10):
        _ = heavy_computation(x)
    eager_time = time.perf_counter() - start

    # JIT 版本的基准测试
    # 注意：block_until_ready() 确保异步计算的完成
    start = time.perf_counter()
    for _ in range(10):
        _ = fast_fn(x).block_until_ready()
    jit_time = time.perf_counter() - start

    print(f"Python 解释执行 × 10:    {eager_time:.4f}s")
    print(f"JIT 编译执行 × 10:        {jit_time:.4f}s")
    print(f"加速比:                   {eager_time / jit_time:.1f}x")
    print()


# ============================================================
# 第 4 部分：PRNG — 显式随机数管理
# ============================================================

def demo_prng():
    """演示 JAX 的 PRNG 系统：无全局状态，完全可复现。"""
    print("=" * 60)
    print("第 4 部分：PRNG — 显式随机数管理")
    print("=" * 60)

    # JAX 没有全局随机种子，必须显式传递 PRNGKey
    key = random.PRNGKey(42)  # 从种子创建根密钥

    print("根密钥:", key)

    # split 从根密钥分出独立子密钥
    # 每次使用前必须 split，不能重复使用同一个密钥
    key1, key2 = random.split(key)

    # 两个独立随机数
    a = random.normal(key1, shape=(3,))
    b = random.normal(key2, shape=(3,))

    print(f"key1 生成的随机数: {a}")
    print(f"key2 生成的随机数: {b}")
    print()

    # 验证可复现性
    same_key = random.PRNGKey(42)
    _, key1_copy = random.split(same_key)
    a_copy = random.normal(key1_copy, shape=(3,))
    print(f"可复现？{jnp.allclose(a, a_copy)}")  # True
    print()


# ============================================================
# 第 5 部分：完整训练 — MNIST 上的 3 层 MLP
# ============================================================

def get_mnist_data():
    """加载 MNIST 数据集并进行归一化。"""
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    # 前 60000 张为训练集，后 10000 张为测试集
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test


def init_params(key):
    """用 He 初始化创建网络参数（返回 pytree）。

    网络结构: 784 → 256 → 128 → 10
    - 输入层: 784（MNIST 图片展平）
    - 隐藏层 1: 256 神经元
    - 隐藏层 2: 128 神经元
    - 输出层: 10 神经元（对应 0-9 数字）
    """
    k1, k2, k3 = random.split(key, 3)

    # He 初始化（ReLU 友好）：scale = sqrt(2 / fan_in)
    params = {
        'layer1': {
            'w': jnp.sqrt(2.0 / 784) * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': jnp.sqrt(2.0 / 256) * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': jnp.sqrt(2.0 / 128) * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params


def forward(params, x):
    """前向传播：三层的全连接 + ReLU 激活。

    Args:
        params: 包含 'layer1', 'layer2', 'layer3' 的 pytree
        x: 形状 (batch_size, 784) 的输入

    Returns:
        logits: 形状 (batch_size, 10) 的输出
    """
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x


def loss_fn(params, x, y):
    """交叉熵损失函数。

    Args:
        params: 模型参数
        x: 输入数据
        y: 标签（整型）

    Returns:
        标量损失值
    """
    logits = forward(params, x)
    # one-hot 编码 + log_softmax 交叉熵
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))


# 创建优化器（Optax 的梯度变换链）
# 先归一化梯度，再用 Adam 更新
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),   # 梯度裁剪：防止爆炸
    optax.adam(learning_rate=1e-3),   # Adam 优化器
)


@jax.jit
def train_step(params, opt_state, x, y):
    """单步训练：计算损失和梯度 → 更新参数。

    被 @jax.jit 装饰，整个函数编译为 XLA 内核。
    第一次调用较慢（编译），后续调用极快。
    """
    # 同时返回损失值和梯度
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    # Optax 处理梯度变换（裁剪 + Adam）
    updates, opt_state = optimizer.update(grads, opt_state, params)
    # 将更新应用到参数（返回新参数，不修改原参数）
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@jax.jit
def accuracy(params, x, y):
    """计算准确率（编译版本）。"""
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)


def train():
    """完整训练循环。"""
    print("=" * 60)
    print("第 5 部分：MNIST 训练 — 函数式训练循环")
    print("=" * 60)

    # 加载数据
    print("正在加载 MNIST 数据集...")
    X_train, y_train, X_test, y_test = get_mnist_data()

    # 转换为 JAX 数组
    X_train = jnp.array(X_train)
    X_test = jnp.array(X_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)

    print(f"训练集: {X_train.shape[0]} 张图片")
    print(f"测试集: {X_test.shape[0]} 张图片")
    print()

    # 初始化参数和优化器状态
    key = random.PRNGKey(0)
    params = init_params(key)
    opt_state = optimizer.init(params)

    batch_size = 128
    n_epochs = 10

    print(f"{'轮次':>5} | {'损失':>8} | {'训练准确率':>12} | {'测试准确率':>12}")
    print("-" * 50)

    for epoch in range(n_epochs):
        # 每个轮次前打乱数据（使用新的 PRNG 密钥）
        key, subkey = random.split(key)
        perm = random.permutation(subkey, len(X_train))
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        epoch_loss = 0.0
        n_batches = len(X_train) // batch_size

        for i in range(n_batches):
            start = i * batch_size
            xb = X_shuffled[start:start + batch_size]
            yb = y_shuffled[start:start + batch_size]
            params, opt_state, loss = train_step(params, opt_state, xb, yb)
            epoch_loss += loss

        # 用前 5000 个样本评估训练准确率
        train_acc = accuracy(params, X_train[:5000], y_train[:5000])
        test_acc = accuracy(params, X_test, y_test)

        print(f"{epoch + 1:5d} | {epoch_loss / n_batches:8.4f} | "
              f"{train_acc:12.4f} | {test_acc:12.4f}")

    print()

    # 输出关键差异：与 PyTorch 对比
    print("训练完成。与 PyTorch 的关键差异：")
    print("  - 参数是显式传递的 pytree，不是 .parameters()")
    print("  - 优化器状态由 opt_state 管理，不是 optimizer.state_dict()")
    print("  - train_step 是纯函数：params in → new_params out，无突变")
    print("  - 整个 train_step 被编译为单个 XLA 内核")
    print()

    return params


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  JAX 入门 — 函数式深度学习框架实战              ║")
    print("╚" + "═" * 58 + "╝")
    print()

    # 1. 梯度变换
    demo_grad()

    # 2. 自动向量化
    demo_vmap()

    # 3. JIT 编译
    demo_jit()

    # 4. 随机数管理
    demo_prng()

    # 5. 完整训练
    train()
