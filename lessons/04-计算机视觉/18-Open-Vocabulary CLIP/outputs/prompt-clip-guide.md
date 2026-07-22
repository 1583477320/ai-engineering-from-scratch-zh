# 零样本 CLIP 提示词模板设计指南

## 用途

为任意类别列表和设计域生成有效的零样本 CLIP 提示词模板集，以最大化分类准确率。

---

## 输入参数

- `classes`：待分类的类别名称列表（字符串）
- `domain`：应用领域
  - `natural_photos`：自然照片（默认）
  - `medical`：医疗影像
  - `satellite`：卫星遥感
  - `industrial`：工业检测
  - `document`：文档/票据
  - `fine_grained`：细粒度识别（同一大类下的多个子类）
- `num_templates_per_class`：每个类别的目标模板数量（推荐 3-80）

---

## 基础模板（所有场景必须包含）

```
"a photo of a {class}"
"a picture of a {class}"
"an image of a {class}"
```

这三个模板覆盖了最基本的自然语言表述，是零样本分类的基线。

---

## 按领域的模板扩展

### natural_photos（自然照片）

| 风格变体 | 示例（类别=猫） |
|---|---|
| 清晰度 | "a clear photo of a {class}", "a blurry photo of a {class}" |
| 距离 | "a close-up of a {class}", "a distant shot of a {class}" |
| 色彩 | "a black and white photo of a {class}", "a colorful photo of a {class}" |
| 画质 | "a low resolution photo of a {class}", "a high quality photo of a {class}" |
| 构图 | "a cropped photo of a {class}", "a centered photo of a {class}" |
| 风格 | "a drawing of a {class}", "a sketch of a {class}", "a painting of a {class}" |
| 场景 | "a {class} in the wild", "a {class} at home", "a {class} outdoors" |

### medical（医疗影像）

| 模态 | 示例（类别=肺炎） |
|---|---|
| X 光 | "an X-ray showing {class}" |
| CT | "a CT scan of {class}" |
| MRI | "an MRI scan revealing {class}" |
| 组织学 | "a histology slide of {class}", "a microscopic view of {class}" |
| 超声 | "an ultrasound image of {class}" |

### satellite（卫星遥感）

| 视角 | 示例（类别=森林） |
|---|---|
| 卫星 | "satellite imagery of {class}" |
| 航拍 | "aerial photography of {class}" |
| 地表覆盖 | "aerial view of {class} from above" |
| 季节 | "a {class} scene in spring", "{class} in winter" |

### industrial（工业检测）

| 场景 | 示例（类别=裂纹缺陷） |
|---|---|
| 巡检 | "industrial inspection image of {class}" |
| 缺陷 | "a defect showing {class}", "close-up of surface {class}" |
| 设备 | "machine component with {class}", "photograph of {class} on metal surface" |

### fine_grained（细粒度分类）

在基础模板之上，增加**超类别限定**：

```
"a photo of a {class}, a type of {super_category}"
"a close-up of {class}, belonging to the {super_category} family"
"{super_category} species: {class}"
```

例如：类别 = "金毛寻回犬"，超类别 = "犬"
模板 = "a photo of a golden retriever, a type of dog"

**注意：** 细粒度场景下，超类别限定至关重要。没有它，模型会将"金毛"和"拉布拉多"的嵌入空间重叠。

---

## 中文场景模板

对于中文 CLIP 模型（如 Chinese-CLIP），使用对应的中文模板：

| 英文模板 | 中文模板 |
|---|---|
| "a photo of a {class}" | "一张{class}的照片" |
| "a picture of a {class}" | "一幅{class}的图画" |
| "a close-up of a {class}" | "{class}的特写" |
| "a {class} in the wild" | "野外环境中的{class}" |
| "a sketch of a {class}" | "{class}的简笔画" |

---

## 注意事项

### 大小写

- 始终将类别名称转为小写（专有名词除外）
- "DOG" 的 CLIP 嵌入质量远低于 "dog"
- 模板统一小写开头："a photo of a {class}"

### 长度限制

- CLIP 文本编码器最大上下文长度为 77 个词元
- 英文模板通常不超过 15 个词元
- 中文按字切分，单条模板建议不超过 20 个汉字
- 超长描述会被截断，丢失关键信息

### 模板数量与收益

| 模板数 | 典型准确率提升 | 推理开销 |
|---|---|---|
| 1 | 基线 | 极低 |
| 3-5 | +0.5~1% | 低 |
| 10-20 | +1~2% | 中 |
| 40-80 | +2~3% | 较高 |
| >100 | <0.5% | 高（不建议） |

### 领域适配

如果在特定领域使用 CLIP，建议在验证集上评估不同模板集合的效果：

1. 生成候选模板集合 A（1 条）和 B（10 条）
2. 在验证集上用两组模板分别测试
3. 选择准确率更高的配置
4. 对测试集应用最优配置

不要假设通用模板集在垂直领域也能达到同等效果——医疗影像中的"class"与自然照片中的"class"需要完全不同的表述体系。
