# GPU 配置配方

## 快速验证

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## 通用设备句柄

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 显存估算

fp16 可容参数数 ≈ 显存(GB) × 0.5 × 10⁹

## 云服务

| 服务 | 价格/h | 最低配置 |
|:----|:-------|:--------|
| Colab | 免费 | T4 15GB |
| Lambda | $0.30 | A10 24GB |
| RunPod | $0.19 | RTX 3090 |
