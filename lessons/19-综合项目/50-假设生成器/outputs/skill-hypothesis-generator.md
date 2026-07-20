# 假设生成器配方

## 配置

```python
config = GeneratorConfig(
    n_passes=6,          # 采样次数
    t_min=0.2,           # 最低温度
    t_max=1.2,           # 最高温度
    novelty_threshold=0.25,  # 新颖性阈值
    w_novelty=0.4,       # 新颖性权重
    w_specificity=0.3,   # 特异性权重
    w_testability=0.3,   # 可测试性权重
)
```

## 假设格式

生成模型输出必须使用结构化标签：

```xml
<hypothesis>
<text>假设的断言文本</text>
<variables>var1, var2, var3</variables>
<metric>要测量的指标</metric>
<baseline>基线参考</baseline>
</hypothesis>
```

## 排序分数公式

```
rank_score = 0.4 × novelty + 0.3 × specificity + 0.3 × testability
```

## 生产部署替换

| 组件 | 开发版本 | 生产版本 |
|------|----------|----------|
| 嵌入 | 哈希嵌入（零依赖） | Sentence-Transformer |
| 语言模型 | MockLLM（脚本化） | API 调用或本地推理 |
| 种子 | 固定种子 | 系统时钟或计数器 |
