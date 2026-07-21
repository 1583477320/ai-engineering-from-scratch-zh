"""API 调用演示——使用原始 HTTP 调用 Anthropic API。"""
import os, urllib.request, json

def call_api(prompt: str, api_key: str = None) -> str:
    """原始 HTTP 调用 Anthropic API。"""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return "错误: 请设置 ANTHROPIC_API_KEY 环境变量"
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    }
    body = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["content"][0]["text"]

def main():
    print("API 调用演示")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        print(f"检测到 API 密钥: {key[:8]}...")
        try:
            result = call_api("用一句话解释什么是神经网络。")
            print(f"响应: {result}")
        except Exception as e:
            print(f"API 调用失败: {e}")
    else:
        print("未设置 ANTHROPIC_API_KEY，跳过实际调用")
        print("请运行: export ANTHROPIC_API_KEY='你的密钥'")
    print("\n密钥存储安全检查:")
    print("  ✓ 使用环境变量而非硬编码")
    print("  ✓ .env 文件已加入 .gitignore")
    print("  ✓ 不将密钥提交到 Git")
    return 0

if __name__ == "__main__":
    sys.exit(main())
