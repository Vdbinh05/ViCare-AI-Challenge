"""Local Gradio interface for the ViCare academic demo."""
import argparse
import errno
import logging
import os
import socket
import sys
from datetime import datetime
from html import escape
from urllib.parse import urlsplit

from src.config import Settings
from src.generation import Assistant
from src.safety import DISCLAIMER, PRIVACY, validate_url

TITLE = "ViCare AI – Vietnamese Medical Knowledge Assistant"
EXAMPLES = (
    "Những yếu tố nào liên quan đến huyết áp cao?",
    "Thủ đô của nước Úc là gì?",
    "Thuốc giả định ZYX-9000 có chữa khỏi mọi bệnh mạn tính không?",
    "Tôi nên uống bao nhiêu viên thuốc này mỗi ngày?",
    "Người nhà bất tỉnh, tôi phải làm gì?",
)
CSS = """
:root { color-scheme: light; }
.gradio-container { max-width: 1160px !important; margin: auto; }
#intro { padding-top: .7rem; }
#intro h1 { color: #105766; margin-bottom: .35rem; }
#notice { padding: .7rem .9rem; background: #eaf6f6; border-left: 4px solid #127b82; border-radius: .4rem; color: #163943; }
#notice p, #notice strong { color: #163943 !important; }
#model-meta { font-size: .92rem; color: #2b4c57; margin-top: .2rem; }
#question { min-width: min(100%, 260px); }
#source-list .reference-card { padding: .65rem .8rem; border: 1px solid #d2e5e9; border-radius: .55rem; margin: .5rem 0; overflow-wrap: anywhere; background: #ffffff; }
#source-list .reference-card.cited { border-color: #127b82; background: #f2fbfb; box-shadow: inset 3px 0 0 #127b82; }
#source-list .cited-badge { display: inline-block; margin-left: .45rem; padding: .08rem .45rem; border-radius: 999px; background: #127b82; color: #ffffff; font-size: .78rem; font-weight: 600; }
#source-list details { margin-top: .4rem; }
#source-list summary { cursor: pointer; color: #155569; }
#source-list .source-copy { white-space: pre-wrap; }
#source-list a { color: #12667a; text-decoration: underline; }
#source-list .source-id { color: #3d5564; }
#footer { margin-top: .7rem; font-size: .88rem; color: #46606b; text-align: center; }
@media (max-width: 640px) { #chat { height: 192px !important; min-height: 192px !important; } }
"""


def render_sources(hits, method=None, cited_ids=None):
    if not hits:
        return "<p>Chưa có tài liệu cho câu hỏi này.</p>"
    cited_set = {str(cid) for cid in (cited_ids or ())}
    cards = []
    for i, hit in enumerate(hits, 1):
        chunk = hit.chunk
        url = validate_url(chunk.url)
        doc_id = escape(chunk.doc_id)
        if url:
            domain = escape(urlsplit(url).hostname)
            origin = (f'<a href="{escape(url, quote=True)}" target="_blank" '
                      f'rel="noopener noreferrer">{domain}</a> <span class="source-id">· {doc_id}</span>')
        else:
            origin = f'{doc_id} <span class="source-id">· Không có liên kết hợp lệ</span>'
        is_cited = str(i) in cited_set
        card_class = "reference-card cited" if is_cited else "reference-card"
        badge = '<span class="cited-badge">Được trích dẫn</span>' if is_cited else ""
        cards.append(
            f'<div class="{card_class}"><strong>[{i}]</strong> {origin}{badge}'
            f'<details><summary>Xem đoạn trích</summary><p class="source-copy">'
            f'{escape(chunk.text)}</p></details></div>'
        )
    label = f"<p>Phương pháp <strong>{escape(method)}</strong></p>" if method else ""
    return label + "".join(cards)


def _format_latency(latency_ms):
    if latency_ms is None:
        return ""
    if latency_ms >= 1000:
        return f"{latency_ms / 1000:.2f}s"
    return f"{latency_ms:.0f} ms"


def format_status(reply):
    code = getattr(reply, "reason_code", "")
    latency = _format_latency(getattr(reply, "latency_ms", None))
    latency_part = f" · Thời gian: {latency}" if latency else ""
    model_part = f" · Mô hình: `{reply.model_name}`" if getattr(reply, "model_name", "") else ""
    retry_part = " · Đã kiểm tra và tạo lại 1 lần" if getattr(reply, "retried", False) else ""

    if code == "SUPPORTED" or reply.status == "answered":
        ordered_cids = sorted(
            getattr(reply, "citation_ids", ()),
            key=lambda cid: (0, int(cid)) if str(cid).isdigit() else (1, str(cid)),
        )
        cids = ", ".join(f"[{cid}]" for cid in ordered_cids)
        cite_part = f" · Nguồn trích dẫn: {cids}" if cids else ""
        return f"**Trạng thái: Có cơ sở tham khảo** · Truy xuất: **{reply.method}**{cite_part}{model_part}{latency_part}{retry_part}"
    if code == "EMERGENCY" or reply.status == "emergency":
        return f"**Cảnh báo khẩn cấp:** Tình huống cần hỗ trợ y tế khẩn cấp (ưu tiên gọi 115).{latency_part}"
    if code == "MISSING_CLINICAL_CONTEXT":
        return f"**Từ chối (Thiếu thông tin lâm sàng):** Không hỗ trợ chẩn đoán cá nhân, kê đơn hoặc liều dùng.{latency_part}"
    if code == "OUT_OF_DOMAIN":
        return f"**Từ chối (Ngoài phạm vi):** Câu hỏi nằm ngoài phạm vi tài liệu sức khỏe.{latency_part}"
    if code == "CONFLICTING_EVIDENCE":
        return f"**Từ chối (Thông tin chưa nhất quán):** Các tài liệu truy xuất chưa thống nhất.{latency_part}"
    if code == "RETRIEVAL_FAILURE":
        return f"**Lỗi truy xuất:** Không thể truy xuất tài liệu tham khảo vào lúc này.{latency_part}"
    if code == "GENERATION_FAILURE" or reply.status == "unavailable":
        note = " Bạn vẫn có thể xem các đoạn tài liệu đã truy xuất bên phải." if reply.hits else ""
        return f"**Dịch vụ chưa sẵn sàng:** Không thể tạo câu trả lời vào lúc này.{note}{latency_part}"
    if code == "INSUFFICIENT_EVIDENCE" or reply.status == "refused":
        method_part = f" · Truy xuất: **{reply.method}**" if reply.hits else ""
        return f"**Từ chối (Thiếu bằng chứng):** Tài liệu tham khảo chưa đủ cơ sở để trả lời đáng tin cậy.{method_part}{latency_part}{retry_part}"
    return reply.answer if reply.status == "invalid" else ""


def build_app(settings=None, assistant=None, startup_status=""):
    import gradio as gr

    settings = settings or Settings.from_env()
    assistant = assistant or Assistant(settings)
    theme = gr.themes.Soft(
        primary_hue="teal", secondary_hue="cyan", neutral_hue="slate",
        font=["system-ui", "Arial", "sans-serif"],
    )
    palette = theme.to_dict()["theme"]
    theme.set(**{name: palette[name.removesuffix("_dark")] for name in palette
                 if name.endswith("_dark") and name.removesuffix("_dark") in palette})
    with gr.Blocks(title=TITLE, theme=theme, css=CSS, analytics_enabled=False) as demo:
        gr.Markdown("# ViCare AI\nTrợ lý tra cứu kiến thức sức khỏe tiếng Việt.", elem_id="intro")
        gr.Markdown(
            f"**{DISCLAIMER}**\n\n{PRIVACY}\n\n"
            f"Mô hình cục bộ: `{settings.llm_model}` · "
            "Hệ thống chỉ tổng hợp từ đoạn trích ViHealthQA và có thể từ chối khi thiếu bằng chứng; "
            "ngưỡng lọc cosine chưa được thẩm định lâm sàng.",
            elem_id="notice",
        )
        method = gr.Radio(
            choices=["Dense", "Hybrid"], value="Hybrid", label="Phương pháp truy xuất", elem_id="method",
            info="Dense tìm theo ngữ nghĩa. Hybrid kết hợp thêm đối chiếu từ khóa BM25 (RRF).",
        )
        with gr.Row():
            with gr.Column(scale=7, min_width=280):
                chat = gr.Chatbot(
                    label="Cuộc trò chuyện", type="messages", height=390, render_markdown=False,
                    placeholder="Nhập một câu hỏi về kiến thức sức khỏe.",
                    elem_id="chat",
                )
                with gr.Row():
                    question = gr.Textbox(
                        label="Câu hỏi", placeholder="Huyết áp cao liên quan đến yếu tố nào?",
                        lines=2, max_lines=4, scale=5, elem_id="question",
                    )
                    send = gr.Button("Gửi", variant="primary", scale=1, elem_id="send")
                status = gr.Markdown(startup_status, elem_id="status")
                clear = gr.Button("Xóa cuộc trò chuyện", size="sm", elem_id="clear")
                gr.Markdown("**Câu hỏi thử**")
                for example in EXAMPLES:
                    gr.Button(example, size="sm").click(
                        lambda value=example: value, outputs=question, queue=False, api_name=False,
                    )
            with gr.Column(scale=4, min_width=280):
                gr.Markdown(
                    "### Tài liệu tham khảo đã truy xuất\n"
                    "Đoạn trích từ kho ViHealthQA, chưa được thẩm định y khoa."
                )
                sources = gr.HTML(render_sources([]), elem_id="source-list")
        gr.Markdown(
            "Bản demo học thuật, không thay thế tư vấn của nhân viên y tế. "
            "Không nhập thông tin định danh cá nhân.",
            elem_id="footer",
        )

        def submit(text, selected, history):
            yield (
                text, history, gr.update(), "Đang xử lý câu hỏi…",
                gr.update(interactive=False), gr.update(interactive=False),
            )
            reply = assistant.ask(text, selected)
            if reply.status == "invalid":
                yield (
                    text, history, gr.update(), reply.answer,
                    gr.update(interactive=True), gr.update(interactive=True),
                )
                return
            updated = (history or []) + [
                {"role": "user", "content": text.strip()},
                {"role": "assistant", "content": reply.answer},
            ]
            yield (
                "", updated, render_sources(reply.hits, reply.method, reply.citation_ids),
                format_status(reply),
                gr.update(interactive=True), gr.update(interactive=True),
            )

        submission = gr.on(
            triggers=[send.click, question.submit], fn=submit, inputs=[question, method, chat],
            outputs=[question, chat, sources, status, send, clear], show_progress="hidden",
            trigger_mode="once", api_name=False,
        )
        clear.click(
            lambda: ("", [], render_sources([]), "", gr.update(interactive=True), gr.update(interactive=True)),
            outputs=[question, chat, sources, status, send, clear], cancels=[submission],
            queue=False, api_name=False,
        )
    return demo.queue(default_concurrency_limit=1)


def port_in_use(port):
    address = ("127.0.0.1", port)
    with socket.socket() as probe:
        try:
            probe.bind(address)
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                return True
            # Windows may report access denied for an exclusive listener.
            if exc.errno == errno.EACCES:
                with socket.socket() as client:
                    client.settimeout(0.3)
                    if client.connect_ex(address) == 0:
                        return True
            raise
    return False


def launch_app(*, port=None, prevent_thread_lock=False):
    from src.ollama_runtime import ModelUnavailable

    if port is not None and not 1 <= port <= 65535:
        raise ValueError("Cổng phải từ 1 đến 65535.")
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    settings = Settings.from_env()
    demo = None
    for candidate in ([port] if port is not None else range(7860, 7870)):
        if port_in_use(candidate):
            continue
        if demo is None:
            assistant = Assistant(settings)
            try:
                status = assistant.generate.prepare()
            except ModelUnavailable as exc:
                status = str(exc)
            print(status, flush=True)
            demo = build_app(settings, assistant, startup_status=status)
        try:
            result = demo.launch(
                server_name="127.0.0.1", server_port=candidate, share=False,
                show_error=False, show_api=False, prevent_thread_lock=True,
                inbrowser=False, inline=False,
            )
        except OSError as exc:
            # Gradio wraps bind failures. Confirm a race before trying another port.
            if str(exc).startswith("Cannot find empty port in range:") and port_in_use(candidate):
                continue
            raise
        print(f"ViCare PID={os.getpid()} · {started} · {result[1]}", flush=True)
        if not prevent_thread_lock:
            demo.block_thread()
        return result
    target = str(port) if port is not None else "7860–7869"
    raise OSError(f"Cổng {target} trên 127.0.0.1 đang được sử dụng.")


def main():
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("src.ollama_runtime").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    try:
        launch_app(port=args.port)
    except Exception as exc:
        logging.exception("Không mở được ViCare (%s)", type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    if os.name == "nt":
        for output in (sys.stdout, sys.stderr):
            output.reconfigure(encoding="utf-8")
    raise SystemExit(main())
