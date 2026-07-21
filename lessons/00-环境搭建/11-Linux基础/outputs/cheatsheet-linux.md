# Linux 速查卡（AI 工程师）

## 导航
```bash
pwd; ls -la; cd ~; cd ..; find . -name "*.py"
```

## 文件
```bash
cp -r src/ backup/; mv old.txt new.txt; rm -rf dir/
cat file.txt; head -20 file.txt; tail -f log.txt; less file.txt
```

## 权限
```bash
chmod +x script.sh; chmod 755 deploy.sh; chown user:group file
```

## 包管理
```bash
sudo apt update; sudo apt install -y htop tmux
```

## 进程
```bash
htop; ps aux | grep python; kill PID
nvidia-smi; watch -n1 nvidia-smi
```

## 磁盘
```bash
df -h; du -sh ./data/*; du -h --max-depth=1 /
```

## 网络
```bash
wget URL; curl -O URL
scp file user@remote:/data/; rsync -avz ./data/ user@remote:/data/
```

## 会话
```bash
tmux new -s name; tmux attach -t name; tmux ls
```
