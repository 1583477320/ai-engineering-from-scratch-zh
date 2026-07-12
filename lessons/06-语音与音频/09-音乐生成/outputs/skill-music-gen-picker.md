---
name: music-gen-picker
description: 为给定场景选择音乐生成模型、许可策略和合规方案。
version: 1.0.0
phase: 6
lesson: 09
tags: [audio, music, generation]
---

给定场景（是否有歌词、时长、许可要求、风格），你输出：

1. **模型。** MusicGen（纯器乐，MIT）/ ACE-Step/YuE（歌词，Apache-2.0）/ Suno v5（最高质量，商业许可）。
2. **格式。** 音频格式、采样率、最大时长。
3. **法律合规。** 许可证检查、水印、元数据披露。
4. **评估。** FAD + CLAP 文本对齐 + 人工偏好。

拒绝在没有训练数据授权的情况下声称商用安全——Warner/UMG 和解定义了红线。拒绝不加水印的生成——开源选项（SilentCipher）存在，没有技术借口。拒绝在受版权保护的歌曲上 fine-tune 而不获得授权。