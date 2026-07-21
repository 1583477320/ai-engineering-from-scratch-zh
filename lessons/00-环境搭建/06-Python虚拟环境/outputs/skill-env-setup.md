# Python 虚拟环境配方

## uv 虚拟环境

```bash
uv python install 3.12
uv venv
source .venv/bin/activate
uv pip install torch numpy
```

## pyproject.toml

```toml
[project.optional-dependencies]
torch = ["torch>=2.3"]
llm = ["anthropic>=0.39"]
```

## 验证

```bash
which python  # 应该指向 .venv/bin/python
```
