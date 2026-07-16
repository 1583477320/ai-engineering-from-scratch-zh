# 工具 Schema Linter——自动检查设计质量


def lint_tool_schema(name, description, schema):
    """检查工具 Schema 设计质量。"""
    issues = []

    # 检查命名
    if not name.replace("_", "").isalpha():
        issues.append(f"工具名 '{name}' 包含非字母字符")

    # 检查描述长度
    if len(description) > 1024:
        issues.append(f"描述过长 ({len(description)} > 1024)")

    # 检查描述内容
    if "当" not in description and "when" not in description.lower():
        issues.append("描述缺少 '当 X 时使用' 模式")

    # 检查必填字段
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    for field in required:
        if field not in properties:
            issues.append(f"必填字段 '{field}' 未在 properties 中定义")

    # 检查字段描述
    for field, field_schema in properties.items():
        if "description" not in field_schema:
            issues.append(f"字段 '{field}' 缺少 description")

    return issues


if __name__ == "__main__":
    print("工具 Schema Linter 演示\n")

    # 好的工具
    good_issues = lint_tool_schema(
        "get_weather",
        "当用户询问当前天气时使用此工具。返回温度和天气条件。",
        {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        }
    )
    print(f"好工具: {good_issues if good_issues else '无问题'}")

    # 差的工具
    bad_issues = lint_tool_schema(
        "helper123!",
        "获取数据",
        {
            "type": "object",
            "properties": {
                "data": {"type": "string"},
            },
        }
    )
    print(f"差工具: {bad_issues}")
