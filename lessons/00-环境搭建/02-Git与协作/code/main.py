"""检查 Git 配置。"""
import subprocess, sys

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""

print("=== Git 配置检查 ===")
for cfg in ["user.name", "user.email"]:
    v = run(["git", "config", "--global", cfg])
    print(f"  {'✓' if v else '✗'} {cfg}: {v or '(未设置)'}")
branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
print(f"  当前分支: {branch or '(非 Git 仓库)'}")
print("\n提示: 确保 user.name 和 user.email 已设置")
sys.exit(0)
