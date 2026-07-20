# HDF5 语料库配方

## HDF5 分片创建

```python
import h5py
f = h5py.File("shard.h5", "w", libver="latest")
f.swmr_mode = True
ds = f.create_dataset("tokens", shape=(0,), maxshape=(None,),
                      dtype=np.uint16, chunks=(8192,))
```

## 生产配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| Chunk size | 4096-16384 | 太小增加元数据开销，太大浪费 I/O |
| Compression | gzip level 4 | 约 30-40% 压缩率，300-500MB/s 解压 |
| SWMR | True | 单写入器多读取器模式 |

## 滑动窗口数据加载

```python
# 训练序列长度 = window_size + 1
# 输入为前 window_size 个词元，目标为后 window_size 个词元偏移一位
# 例如：window_size=4
# 样本: 输入=[10,20,30,40], 目标=[20,30,40,50]
```

## 中文词元类型

- 词表 ≤ 65535: 使用 `np.uint16`
- 词表 > 65535: 使用 `np.uint32`
