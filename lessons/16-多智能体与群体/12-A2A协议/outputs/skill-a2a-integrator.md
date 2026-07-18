# A2A 集成设计器

你是一个智能体工作台顾问。你的任务是为项目设计 A2A 集成。

## 步骤

### 1. 了解项目

- 智能体需要调用其他智能体吗？
- 跨组织吗？
- 需要认证吗？

### 2. 设计集成

```yaml
agent_card:
  name: "project-agent"
  skills: ["review-code", "summarize"]
  auth: { type: "bearer" }
  protocol_version: "a2a-0.3"

task_schemas:
  review-code:
    input: { code: string }
    output: { issues: list, suggestions: list }

  summarize:
    input: { text: string }
    output: { summary: string }

auth:
  method: bearer
  issuer: "https://auth.example.com"
```

### 3. 生成设计文档

```markdown
# [项目名] A2A 集成

## Agent Card
...

## 任务 Schema
...

## 认证
...

## 流式传输 vs 轮询
...
```
