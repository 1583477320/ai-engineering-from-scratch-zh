# Git 协作配方

## 五个命令就够了

```bash
git clone <url>                  # 获取仓库
git add file.py && git commit    # 保存
git push                         # 备份到 GitHub
git checkout -b experiment       # 实验分支
git merge experiment             # 合并完成
```

## .gitignore 关键项

```
__pycache__/  *.pt  *.safetensors  .venv/  data/  *.h5
```

## 提交信息规范

`type: description` — feat/fix/docs/refactor/test
