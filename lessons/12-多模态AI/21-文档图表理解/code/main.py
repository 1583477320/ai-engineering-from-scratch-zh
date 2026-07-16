# 文档理解管道：OCR + VLM 问答


def extract_table_structure(image):
    """简化版表格提取。"""
    return [{"bbox": [0, 0, 100, 100], "rows": 3, "cols": 2}]


def document_qa(vlm, document_image, question):
    """VLM 文档问答。"""
    prompt = f"根据以下文档，回答：{question}"
    return f"基于文档内容回答：{question[:30]}..."


if __name__ == "__main__":
    print("文档理解管道演示\n")
    tables = extract_table_structure(None)
    print(f"检测到 {len(tables)} 个表格")
    answer = document_qa(None, None, "收入增长率是多少？")
    print(f"问答: {answer}")
