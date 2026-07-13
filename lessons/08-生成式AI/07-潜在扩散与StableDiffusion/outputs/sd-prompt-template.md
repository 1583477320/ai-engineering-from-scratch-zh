# Stable Diffusion 提示词工程模板

## 用途

面向中文用户的 Stable Diffusion 提示词编写指南和模板。

## 核心原则

1. **英文提示词效果优于中文**——SD 的文本编码器在英文数据上训练
2. **结构：主体 + 细节 + 风格 + 质量**
3. **负面提示词至关重要**

## 提示词模板

### 基础模板

```
<prompt>
[主体描述], [细节描述], [艺术风格], [光照效果], [构图], [质量标签]
</prompt>

<negative_prompt>
blurry, low quality, deformed, distorted, disfigured, bad anatomy,
mutated, extra limbs, poorly drawn face, poorly drawn hands
</negative_prompt>
```

### 中文到英文翻译策略

使用 LLM 将中文提示词翻译为英文：

```
请将以下中文提示词翻译为 Stable Diffusion 优化的英文提示词。
保持描述的核心内容不变，但使用 SD 训练数据中常见的表达方式。

中文提示词：<用户输入>
```

### 常用质量标签

| 标签 | 效果 |
|------|------|
| masterpiece, best quality | 提升整体质量 |
| ultra detailed, sharp focus | 锐化和细节 |
| cinematic lighting | 电影级光照 |
| depth of field | 景深效果 |
| 8k uhd, HDR | 超高清 |

### 常用风格标签

| 风格 | 标签 |
|------|------|
| 写实摄影 | `photorealistic, DSLR photo, natural lighting` |
| 动漫 | `anime style, manga, cel shading` |
| 油画 | `oil painting, impasto, canvas texture` |
| 水彩 | `watercolor, wet-on-wet, paper texture` |
| 赛博朋克 | `cyberpunk, neon lights, futuristic city` |

## 推荐参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `num_inference_steps` | 20-50 | 步数越多质量越好，但超过 50 边际效益递减 |
| `guidance_scale` | 7.5 | 默认值，中文提示词可适当降低到 5-6 |
| `seed` | 随机 | 固定 seed 可复现相同生成结果 |

## 示例

### 示例 1：中国山水画

```
英文提示词：
Chinese ink wash painting, misty mountains, ancient pine trees, 
a lone fisherman on a river, traditional shan-shui style, 
monochrome with subtle green tones, masterpiece, best quality

负面提示词：
colorful, cartoon, 3d, realistic, western art style
```

### 示例 2：现代城市摄影

```
英文提示词：
Chengdu city skyline at dusk, Jinjiang River, neon lights reflecting on water, 
urban photography, wide angle, golden hour, cinematic lighting, 8k, 
masterpiece, best quality

负面提示词：
blurry, low resolution, bad composition, oversaturated
```
