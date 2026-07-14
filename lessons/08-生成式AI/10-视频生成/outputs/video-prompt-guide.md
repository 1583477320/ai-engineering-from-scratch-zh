# 视频生成提示词指南

## 用途

面向中文用户的视频生成提示词编写指南和模板。

## 核心原则

1. **描述运动轨迹**——视频的关键是运动，提示词必须包含运动描述
2. **指定镜头语言**——推拉摇移、固定/运动、特写/远景
3. **详细为主**——视频生成比图像生成需要更详细的描述

## 提示词模板

### 基础模板

```
英文提示词：
[主体], [运动描述], [场景/背景], [镜头语言], [光照], [质量标签]

中文提示词：
[主体], [运动描述], [场景/背景], [镜头语言], [光照], [质量标签]
```

### 示例：人物动作

```
英文：
A young woman walking down a rainy street at night, slow motion,
cinematic lighting, reflections on wet pavement, close-up to wide shot,
masterpiece, best quality, 8k

中文：
一位年轻女性在夜晚的雨街行走，慢动作，电影级光照，湿路面上的倒影，
从特写到远景的镜头切换，杰作，最佳画质，8k
```

### 示例：动物

```
英文：
An orange cat slowly walking across a wooden floor, sunlight streaming
through window, shallow depth of field, cinematic, natural lighting,
masterpiece

中文：
一只橘猫慢慢走过木地板，阳光从窗户照进来，浅景深，电影质感，自然光照
```

### 示例：风景/延时

```
A misty mountain landscape at sunrise, clouds flowing over peaks,
time-lapse effect, morning light, majestic, cinematic, 8k
```

### 示例：城市

```
A bustling Tokyo street at night, neon lights reflecting on wet ground,
people walking, cars passing, cyberpunk atmosphere, slow motion
```

## 镜头语言词汇

| 中文 | 英文 |
|------|------|
| 推镜头 | dolly in / zoom in |
| 拉镜头 | dolly out / zoom out |
| 从左到右平移 | pan from left to right |
| 从上到下摇 | tilt down |
| 慢动作 | slow motion |
| 延时摄影 | time-lapse |
| 固定镜头 | static camera / fixed shot |
| 手持镜头 | handheld camera shot |
| 特写 | close-up shot |
| 远景 | wide shot / long shot |
| 浅景深 | shallow depth of field |
| 跟随拍摄 | tracking shot / follow shot |

## 质量标签

| 标签 | 效果 |
|------|------|
| masterpiece, best quality | 最高质量 |
| cinematic lighting | 电影级光照 |
| 8k, high resolution | 超高分辨率 |
| slow motion | 慢动作（流畅画面） |
| smooth camera movement | 平滑镜头运动 |

## 中文 vs 英文提示词

视频生成模型（尤其是 SVD 和 Open-Sora）的文本编码器主要面向英文。推荐策略：

1. 先用大语言模型将中文提示词翻译为英文
2. 保留核心描述不变，但使用视频生成领域常用的表达方式
3. 避免使用过于文学化的描述

示例：

| 中文 | 英文（推荐） |
|------|-------------|
| "一只小鸟在枝头欢快地歌唱" | "A small bird on a branch singing, natural lighting, close-up" |
| "夕阳西下，余晖洒在海面上" | "Sunset over the ocean, golden light on water surface, wide shot" |

## 常见错误

- ❌ 只写一个名词（"猫"）→ 没有运动，生成质量差
- ✅ 描述运动（"一只猫慢慢走过地板"）
- ❌ 写矛盾的描述（"静止不动，同时快速移动"）
- ✅ 明确的运动方向（"从左向右走过画面"）
