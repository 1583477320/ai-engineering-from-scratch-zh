# 论文写作器配方

## 骨架结构

```
Paper(title, authors, abstract)
  sections: [Section(id, title, body, cites)]
  figures:  [Figure(id, path, caption)]
  bib:      {key: BibEntry(key, fields)}
```

## 验证门

1. 图表 ID 唯一
2. 引用有对应 BibEntry
3. 标题和摘要非空

## 输出文件

- paper.tex
- references.bib
- manifest.json
