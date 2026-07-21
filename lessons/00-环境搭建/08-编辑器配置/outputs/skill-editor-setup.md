# VS Code AI 工程配置

## 必装扩展

```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension eamodio.gitlens
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension charliermarsh.ruff
```

## 关键设置

```json
{
    "python.analysis.typeCheckingMode": "basic",
    "editor.formatOnSave": true,
    "editor.rulers": [88, 120],
    "notebook.output.scrolling": true
}
```

## Remote SSH

```bash
ssh-keygen -t ed25519 -C "email"
ssh-copy-id user@gpu-box
```

VS Code: Ctrl+Shift+P → Remote-SSH: Connect to Host
