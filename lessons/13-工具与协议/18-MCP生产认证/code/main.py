# JWKS 缓存 + 受众绑定验证

import time
import hashlib


class JWKSProvider:
    """JWKS 缓存和刷新。"""
    def __init__(self, ttl=3600):
        self.jwks = {}
        self.last_fetch = 0
        self.ttl = ttl

    def get_key(self, kid):
        if time.time() - self.last_fetch > self.ttl:
            self._refresh()
        return self.jwks.get(kid)

    def _refresh(self):
        self.jwks = {"key-1": {"kty": "RSA", "kid": "key-1", "use": "sig"}}
        self.last_fetch = time.time()
        print("  JWKS 已刷新")


def verify_audience(token_aud, server_aud):
    if token_aud != server_aud:
        return False, f"受众不匹配: 令牌={token_aud}, 服务器={server_aud}"
    return True, "受众匹配"


if __name__ == "__main__":
    print("MCP 生产认证演示\n")

    jwks = JWKSProvider()
    key = jwks.get_key("key-1")
    print(f"JWKS 密钥: {key}")

    ok, msg = verify_audience("https://my-server.com/mcp", "https://my-server.com/mcp")
    print(f"受众验证 (匹配): {msg}")

    ok, msg = verify_audience("https://other.com/mcp", "https://my-server.com/mcp")
    print(f"受众验证 (不匹配): {msg}")
