# MCP OAuth 2.1 状态机

import hashlib
import secrets


class MCPAuthState:
    """MCP OAuth 2.1 状态机。"""
    STATES = ["unauthenticated", "authorization_required", "authorized", "error"]

    def __init__(self):
        self.state = "unauthenticated"
        self.tokens = {}
        self.code_verifier = None

    def start_authorization(self):
        """启动 PKCE 授权流程。"""
        self.code_verifier = secrets.token_urlsafe(32)
        code_challenge = hashlib.sha256(self.code_verifier.encode()).hexdigest()
        self.state = "authorization_required"
        return {"authorization_url": f"https://auth.example.com/authorize?code_challenge={code_challenge}"}

    def handle_callback(self, code):
        """处理授权回调。"""
        self.state = "authorized"
        self.tokens = {"access_token": f"token_{code[:8]}"}
        return self.tokens


if __name__ == "__main__":
    print("MCP OAuth 2.1 状态机演示\n")
    auth = MCPAuthState()

    # 启动授权
    result = auth.start_authorization()
    print(f"状态: {auth.state}")
    print(f"授权URL: {result['authorization_url'][:50]}...")

    # 处理回调
    tokens = auth.handle_callback("auth_code_123")
    print(f"状态: {auth.state}")
    print(f"令牌: {tokens}")

    print("\n渐进授权流程:")
    print("  1. Client 检测需要 OAuth")
    print("  2. Client 重定向到授权服务器 (PKCE)")
    print("  3. 用户授权")
    print("  4. Client 获取令牌")
    print("  5. 如果 403 → 请求扩展 scope → 重复")
