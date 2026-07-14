# Inpainting 提示词编写指南

## 用途

指导如何为 inpainting、outpainting、图像编辑任务编写有效的提示词。

## 核心原则

1. **描述修复后的目标状态**，不是原始状态
2. **简洁具体**，避免冗长
3. **包含风格描述**，确保修复区域与原图风格一致

## Inpainting 提示词模板

### 去除物体（如电线杆、行人）

```
英文提示词：
[场景描述], no [要去除的物体], clean background, masterpiece, best quality
```

示例：
```
A clean living room with wooden floor, no power lines, natural lighting,
masterpiece, best quality
```

### 修复损坏照片

```
英文提示词：
[场景描述], restored, high quality, detailed, sharp focus
```

### 填充空白区域

```
英文提示词：
[场景描述], [填充内容描述], seamless, natural, masterpiece
```

## Outpainting 提示词模板

```
英文提示词：
[扩展区域的内容描述], [风格], seamless extension, masterpiece, best quality
```

示例：
```
A misty mountain landscape extending into the distance, traditional Chinese
painting style, seamless extension, masterpiece
```

## InstructPix2Pix 提示词模板

### 简洁指令

| 意图 | 提示词 |
|------|--------|
| 改颜色 | "把天空变成蓝色" / "Make the sky blue" |
| 换背景 | "把背景换成海滩" / "Change background to beach" |
| 加元素 | "添加一个太阳" / "Add a sun" |
| 改风格 | "变成水彩画风格" / "Make it watercolor style" |

### 控制编辑强度

- `image_guidance_scale=1.0`：严格遵循指令，可能偏离原图
- `image_guidance_scale=1.5`（推荐）：平衡编辑效果和原图保持
- `image_guidance_scale=2.0`：接近原图，编辑效果较弱

## 质量标签

| 标签 | 效果 |
|------|------|
| masterpiece, best quality | 提升整体质量 |
| seamless, natural | 确保修复区域与原图自然过渡 |
| detailed, sharp focus | 增强细节 |
| no artifacts, no blur | 避免伪影和模糊 |

## 常见错误

- ❌ "修复这张照片" → 太模糊，模型不知道要修复什么
- ✅ "一个干净的客厅，没有电线杆，自然光照" → 具体描述修复后的目标状态
- ❌ "让这张图变好看" → 太主观
- ✅ "把天空变成日落的橙色" → 具体的编辑指令
