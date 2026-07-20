# 滑动窗口数据集技能

## 目标
构建将编码后的token ID流转换为PyTorch训练数据集的管道。

## 评估标准

| 权重 | 标准 | 测量方法 |
|:---:|------|----------|
| 25 | Dataset正确 | __len__和__getitem__返回正确形状的input/target |
| 20 | 目标移位 | target = input向左移一位断言 |
| 20 | 确定性洗牌 | 相同种子→相同批次，不同epoch→不同顺序 |
| 20 | 窗口计数公式 | count_windows与len(dataset)一致 |
| 15 | stride权衡展示 | stride变化时窗口数的可观察变化 |

## 构建检查清单

- [ ] SlidingWindowDataset（__len__/__getitem__）
- [ ] count_windows静态方法
- [ ] make_dataloader（带种子训练的Generator）
- [ ] input/target移位（window[:-1]/window[1:]）
- [ ] 形状断言（B,T）
- [ ] 确定性洗牌测试
- [ ] stride权衡演示
