# AGENTS.md + SKILL.md 三层架构生成器


def generate_agents_md(project_name, description, architecture, conventions):
    """生成 AGENTS.md 文件。"""
    return f"""# {project_name}

{description}

## 架构
{architecture}

## 约定
{conventions}

## 智能体使用指南
- 参考各子目录中的 SKILL.md 文件
- 测试前运行 `make test`
- 部署参考 `deploy/` 目录
"""


def generate_skill_md(title, prerequisites, steps, notes=None):
    """生成 SKILL.md 文件。"""
    content = f"""# {title}

## 前提条件
{chr(10).join("- " + p for p in prerequisites)}

## 步骤
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(steps))}

"""
    if notes:
        content += f"## 注意事项\n{chr(10).join('- ' + n for n in notes)}"
    return content


if __name__ == "__main__":
    print("三层架构演示\n")

    # AGENTS.md
    agents_md = generate_agents_md(
        "数据管道项目",
        "这是一个 Python 数据管道项目，用于处理 S3 数据。",
        "数据源 → Spark → Parquet → 数据仓库",
        "- 使用 Black 格式化代码\n- 中文注释\n- 提交前运行测试",
    )
    print("AGENTS.md:")
    print(agents_md[:150] + "...")

    # SKILL.md
    skill_md = generate_skill_md(
        "创建数据管道",
        ["PySpark 环境已配置", "数据源已准备"],
        ["在 pipelines/ 创建 Python 文件", "定义数据模式", "实现 transform()", "添加测试", "运行 make test"],
        ["遵循现有代码风格", "处理边界情况"],
    )
    print("SKILL.md:")
    print(skill_md[:150] + "...")

    print("三层架构:")
    print("  AGENTS.md: 项目上下文（根目录）")
    print("  SKILL.md: 任务方法论（子目录）")
    print("  MCP: 工具接口（进程）")
