# 开发环境检查配方

## 快速验证

```bash
uv python list | grep "3.12"
node --version
rustc --version
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## 包管理器

- Python: uv (快 10-100x)
- Node.js: pnpm (磁盘空间高效)
- Rust: cargo (官方)

## 设备

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
