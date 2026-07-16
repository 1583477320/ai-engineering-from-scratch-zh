# MCP App 资源生成器

import json
import hashlib


class MCPAppEmitter:
    """MCP App 资源生成器。"""
    @staticmethod
    def create_dashboard(data, title="仪表板"):
        """生成仪表板 HTML。"""
        data_html = "<ul>" + "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in data.items()) + "</ul>"
        html = f"""<html><head><title>{title}</title><style>body{{font-family:sans-serif;padding:20px;}}</style></head>
<body><h2>{title}</h2>{data_html}
<script>window.parent.postMessage({{type:'mcp-app-ready'}}, '*');</script></body></html>"""
        return {"uri": f"ui://app-{hash(title) % 10000}", "mimeType": "text/html;profile=mcp-app", "text": html, "metadata": {"title": title}}

    @staticmethod
    def create_form(fields, title="表单"):
        """生成交互式表单 HTML。"""
        fields_html = "".join(
            f'<div><label>{f["label"]}</label><input name="{f["name"]}" type="{f.get("type","text")}" /></div>'
            for f in fields
        )
        html = f"""<html><head><title>{title}</title></head>
<body><h2>{title}</h2><form>{fields_html}
<button onclick="window.parent.postMessage({{type:'mcp-app-submit',data:'submitted'}}, '*')">提交</button>
</form></body></html>"""
        return {"uri": f"ui://form-{hash(title) % 10000}", "mimeType": "text/html;profile=mcp-app", "text": html}


if __name__ == "__main__":
    print("MCP App UI 资源演示\n")

    emitter = MCPAppEmitter()

    dashboard = emitter.create_dashboard({"用户数": 1234, "请求量": 5678, "错误率": "0.5%"})
    print(f"仪表板: uri={dashboard['uri']}, mime={dashboard['mimeType']}")
    print(f"  HTML 长度: {len(dashboard['text'])} 字符")

    form = emitter.create_form([
        {"name": "title", "label": "笔记标题", "type": "text"},
        {"name": "content", "label": "笔记内容", "type": "text"},
    ], "新建笔记")
    print(f"\n表单: uri={form['uri']}, mime={form['mimeType']}")
    print(f"  HTML 长度: {len(form['text'])} 字符")
