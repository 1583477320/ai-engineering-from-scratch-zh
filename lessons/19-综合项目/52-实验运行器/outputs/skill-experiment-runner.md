# 实验运行器配方

## 子进程配置

```python
proc = subprocess.Popen([sys.executable, script, config_path],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
```

## 超时与内存

- 墙上时钟: communicate(timeout=60)
- 内存: 轮询 /proc/{pid}/status VmRSS
- 超时后必须 proc.kill()

## 消融表

```python
specs = ablate(base_spec, "lr", [1e-3, 5e-4, 1e-4])
```
