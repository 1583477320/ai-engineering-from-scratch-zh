# Jupyter 笔记本问题诊断

| 问题 | 修复 |
|:-----|:-----|
| 单元格乱序执行报错 | Kernel → Restart & Run All |
| 变量未定义 | 重启内核，从头运行 |
| 内存占用过高 | del + gc.collect() 或重启 |
| 输出不显示 | 检查 %matplotlib inline |
| 图表不显示 | 添加 plt.show() 或 %matplotlib inline |
