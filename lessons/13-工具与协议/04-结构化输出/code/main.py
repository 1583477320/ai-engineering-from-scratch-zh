# 结构化输出：JSON Schema 验证 + 严格模式


def validate_json_schema(data, schema):
    """简化版 JSON Schema 验证。"""
    errors = []
    for field in schema.get("required", []):
        if field not in data:
            errors.append(f"缺少必填字段: {field}")

    for field, field_schema in schema.get("properties", {}).items():
        if field in data:
            expected = field_schema.get("type")
            if expected == "number" and not isinstance(data[field], (int, float)):
                errors.append(f"字段 {field}: 期望 number，实际 {type(data[field]).__name__}")
            elif expected == "string" and not isinstance(data[field], str):
                errors.append(f"字段 {field}: 期望 string，实际 {type(data[field]).__name__}")

    return errors


def strict_format(schema):
    """生成 OpenAI strict 格式的 response_format。"""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "output_schema",
            "strict": True,
            "schema": schema,
        }
    }


if __name__ == "__main__":
    print("结构化输出演示\n")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "产品名称"},
            "price": {"type": "number", "minimum": 0},
            "in_stock": {"type": "boolean"},
        },
        "required": ["name", "price", "in_stock"],
    }

    # 验证
    valid = {"name": "iPhone", "price": 999, "in_stock": True}
    invalid = {"name": "iPhone", "price": "999"}  # price 是字符串

    print("有效数据:", validate_json_schema(valid, schema))
    print("无效数据:", validate_json_schema(invalid, schema))
    print("strict 格式:", strict_format(schema)["type"])
