"""Optional Gradio adapter for EvolutionApplicationService."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .service import EvolutionApplicationService


def preview_callback(
    service: EvolutionApplicationService,
    prompt: str,
    sequence_text: str,
) -> tuple[str, dict[str, Any], str]:
    """Pure callback kept separate from Gradio for fast contract tests."""

    try:
        preview = service.preview(prompt, sequence_text=sequence_text or None)
    except Exception as error:  # noqa: BLE001 - UI boundary returns a public failure state.
        return f"预检失败：{error}", {}, ""
    status = "可确认运行" if preview.ready_for_confirmation else "需要修正输入"
    message = f"{status}。{preview.confirmation_summary}"
    if preview.blockers:
        message += "\n阻断项：" + "；".join(preview.blockers)
    if preview.warnings:
        message += "\n提示：" + "；".join(preview.warnings)
    return message, preview.model_dump(mode="json"), preview.preview_id


def run_callback(
    service: EvolutionApplicationService,
    preview_id: str,
    confirmed: bool,
) -> tuple[str, dict[str, Any], list[str]]:
    """Run one confirmed preview and expose only allow-listed public artifacts."""

    try:
        result = service.run(preview_id, confirmed=confirmed)
    except Exception as error:  # noqa: BLE001 - UI boundary returns a public failure state.
        return f"未启动或运行失败：{error}", {}, []
    return (
        result.public_message,
        result.summary,
        list(result.artifact_paths),
    )


def build_app(config_path: str | Path):
    """Create the local UI without importing Gradio in the core package path."""

    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError(
            "交互界面是可选组件；请安装 `pip install -e .[ui]` 后重试。"
        ) from error

    service = EvolutionApplicationService(config_path)
    with gr.Blocks(title="Fitness Agents 开放序列定向进化") as app:
        gr.Markdown(
            "# 开放序列定向进化\n"
            "用自然语言描述目标；系统先生成结构化预览，确认后才会启动计算。"
        )
        preview_state = gr.State("")
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="对话式需求",
                    lines=6,
                    placeholder=(
                        "例如：希望对该序列进行定向进化，提高结合能力，开放全部位置，输出 8 条。"
                    ),
                )
                sequence = gr.Textbox(
                    label="完整参考序列或 FASTA（可留空使用可信配置）",
                    lines=7,
                )
                preview_button = gr.Button("解析并预览", variant="primary")
                confirmation = gr.Checkbox(
                    label="我已核对 reference hash、允许位置、预算和模型能力"
                )
                run_button = gr.Button("确认并运行")
            with gr.Column(scale=2):
                status = gr.Textbox(label="状态", lines=5, interactive=False)
                preview_json = gr.JSON(label="结构化任务卡")
                result_json = gr.JSON(label="运行摘要")
                downloads = gr.Files(label="已批准的候选与审计产物")

        preview_button.click(
            fn=lambda text, fasta: preview_callback(service, text, fasta),
            inputs=[prompt, sequence],
            outputs=[status, preview_json, preview_state],
            concurrency_limit=1,
            concurrency_id="open-design",
        )
        run_button.click(
            fn=lambda preview_id, checked: run_callback(service, preview_id, checked),
            inputs=[preview_state, confirmation],
            outputs=[status, result_json, downloads],
            concurrency_limit=1,
            concurrency_id="open-design",
        )
    return app


def launch_app(
    config_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
) -> None:
    app = build_app(config_path)
    app.queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        share=False,
        show_error=False,
    )
