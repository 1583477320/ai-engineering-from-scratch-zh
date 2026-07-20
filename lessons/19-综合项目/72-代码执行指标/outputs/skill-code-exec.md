# 代码执行指标配方

## 子进程模板

```python
script = safe_builtins + denied_imports + code + assertion_loop
subprocess.run([sys.executable, "-c", script], timeout=3.0)
```

## 退出码

pass / assertion_fail / syntax_error / timeout / error

## pass@k

```
pass_at_k(n, c, k) = 1 - C(n-c, k) / C(n, k)
```
