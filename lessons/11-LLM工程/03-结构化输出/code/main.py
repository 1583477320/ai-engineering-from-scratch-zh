# 结构化输出：Schema 定义 + 验证 + 重试

import json
from typing import Optional, List, Any
from dataclasses import dataclass


# ============================================================================
# 第 1 步：Pydantic 风格 Schema 定义（无外部依赖）
# ============================================================================

@dataclass
class Product:
    name: str
    price: float
    in_stock: bool
    category: Optional[str] = None

    @classmethod
    def schema(cls):
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "产品名称"},
                "price": {"type": "number", "description": "产品价格"},
                "in_stock": {"type": "boolean", "description": "库存状态"},
                "category": {"type": "string", "description": "产品类别（可选）"},
            },
            "required": ["name", "price", "in_stock"],
        }


# ============================================================================
# 第 2 步：JSON 解析和验证
# ============================================================================

def parse_json(json_str: str) -> Optional[dict]:
    """解析 JSON——处理常见 LLM 输出问题。"""
    json_str = json_str.strip()
    # 移除 markdown 代码块
    if json_str.startswith("```"):
        json_str = "\n".join(json_str.split("\n")[1:-1])
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def validate_product(data: dict) -> Optional[Product]:
    """验证并构造 Product。"""
    try:
        return Product(
            name=data.get("name", ""),
            price=float(data.get("price", 0)),
            in_stock=bool(data.get("in_stock", False)),
            category=data.get("category"),
        )
    except (TypeError, ValueError):
        return None


# ============================================================================
# 第 3 步：带验证和重试的提取
# ============================================================================

def extract_product_safe(text: str, max_retries=2) -> Optional[Product]:
    """模拟带验证和重试的提取。"""
    for attempt in range(max_retries + 1):
        # 模拟模型生成 JSON（实际中调用 LLM API）
        mock_json = json.dumps({
            "name": "Sony WH-1000XM5",
            "price": 348.00,
            "in_stock": True,
        })

        data = parse_json(mock_json)
        if not data and attempt < max_retries:
            continue

        product = validate_product(data)
        if product:
            return product

        if attempt < max_retries:
            # 带错误反馈重试
            pass
    return None


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("结构化输出演示\n")

    # 1. Schema 展示
    print(f"产品 Schema:\n{json.dumps(Product.schema(), indent=2)}\n")

    # 2. JSON 解析
    test_json = '{"name": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}'
    parsed = parse_json(test_json)
    print(f"解析结果: {parsed}")

    product = validate_product(parsed)
    if product:
        print(f"产品: {product.name}, 价格: {product.price}, 有库存: {product.in_stock}")

    # 3. Markdown 包裹的 JSON（常见问题）
    md_json = '```json\n{"name": "测试产品", "price": 100.00, "in_stock": false}\n```'
    parsed_md = parse_json(md_json)
    print(f"\nMarkdown 包裹解析: {parsed_md}")

    # 4. 安全提取
    result = extract_product_safe("某电商产品描述文本")
    if result:
        print(f"\n提取成功: {result.name} (${result.price})")
