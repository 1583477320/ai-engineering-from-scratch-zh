---
name: edge-target-picker
description: 根据平台、模型和延迟/内存预算，选择量化格式和转换路径。
version: 1.0.0
phase: 17
lesson: 12
tags: [edge, ane, hexagon, webgpu, jetson, 量化, 边缘推理]
---

根据平台（iOS/Android/浏览器/Jetson）、模型、延迟/内存预算，选择量化格式和转换路径。

**输出：**

1. **目标识别。** 根据部署平台确定边缘目标（Apple ANE、Hexagon、WebGPU、Jetson）。
2. **量化格式选择。** 根据目标选择：Core ML INT4、QNN INT8/INT4、WebGPU Q4 MLC、NVFP4。
3. **上下文长度限制。** 根据设备内存计算安全的上下文长度（通常4K-8K）。
4. **回退策略。** 如果WebGPU覆盖率不足，设计服务端回退路径。
5. **性能预期。** 给出基于带宽计算的decode吞吐量天花板。

**硬拒绝：**
- 在边缘使用128K上下文——8GB设备会OOM。
- 选择TOPS最高但带宽不足的设备——带宽才是decode瓶颈。

**输出格式：** 边缘推理方案——目标、量化格式、上下文限制、回退策略、性能预期。
