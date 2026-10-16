"""Gradio UI for Poppy Chat."""

from collections.abc import Iterator
from typing import TypedDict

from .lm_studio import (
    AUTO_REASONING,
    DEFAULT_API_HOST,
    ChatStreamEvent,
    LMStudioService,
    LoadedModel,
)


class ConversationState(TypedDict):
    response_id: str
    api_host: str
    model_id: str


def _reasoning_choices(
    models: list[LoadedModel], selected_model: str | None
) -> list[tuple[str, str]]:
    selected = next(
        (model for model in models if model.identifier == selected_model), None
    )
    options = [("自動（LM Studioの既定値）", AUTO_REASONING)]
    if selected:
        options.extend((option, option) for option in selected.reasoning_options)
    return options


def refresh_models(
    api_host: str,
    selected_model: str | None,
    *,
    service: LMStudioService | None = None,
) -> tuple[
    list[tuple[str, str]],
    str | None,
    list[tuple[str, str]],
    str,
    str,
]:
    """Load available models and reasoning choices with a readable status."""
    service = service or LMStudioService()
    try:
        models = service.list_loaded_models(api_host)
    except Exception as error:
        return (
            [],
            None,
            _reasoning_choices([], None),
            AUTO_REASONING,
            f"LM Studioに接続できません: {error}",
        )

    if not models:
        return (
            [],
            None,
            _reasoning_choices([], None),
            AUTO_REASONING,
            "接続しましたが、ロード済みのLLMがありません。LM Studioでモデルをロードしてください。",
        )

    choices = [(model.display_name, model.identifier) for model in models]
    selected = (
        selected_model
        if any(model.identifier == selected_model for model in models)
        else models[0].identifier
    )
    return (
        choices,
        selected,
        _reasoning_choices(models, selected),
        AUTO_REASONING,
        f"接続しました（ロード済みモデル{len(models)}件）。",
    )


def chat_response(
    message: str,
    history: list[dict[str, str]],
    api_host: str,
    model_id: str | None,
    reasoning: str,
    conversation_state: ConversationState | None,
    *,
    service: LMStudioService | None = None,
) -> Iterator[tuple[str, ConversationState | None]]:
    """Stream cumulative assistant text and update the native chat response ID."""
    service = service or LMStudioService()
    if not model_id:
        yield (
            "モデルを選択してください。上の「モデル一覧を更新」から接続してください。",
            None,
        )
        return

    matches_current_chat = bool(
        history
        and conversation_state
        and conversation_state["api_host"] == api_host
        and conversation_state["model_id"] == model_id
    )
    previous_response_id = (
        conversation_state["response_id"]
        if matches_current_chat and conversation_state
        else None
    )
    current_state = conversation_state if matches_current_chat else None

    response_text = ""
    try:
        for event in service.stream_reply(
            api_host=api_host,
            model_id=model_id,
            message=message,
            previous_response_id=previous_response_id,
            reasoning=reasoning,
        ):
            if event.content:
                response_text += event.content
                yield response_text, current_state
            if event.response_id:
                current_state = {
                    "response_id": event.response_id,
                    "api_host": api_host,
                    "model_id": model_id,
                }
                yield response_text, current_state
    except Exception as error:
        detail = f"\n\nLM Studioに接続できません: {error}"
        yield response_text + detail, current_state


def build_app(service: LMStudioService | None = None):
    """Create the Gradio app."""
    import gradio as gr

    service = service or LMStudioService()
    auto_reasoning_choices = _reasoning_choices([], None)

    def update_model_choices(api_host: str, selected_model: str | None):
        choices, selected, reasoning_choices, reasoning, status = refresh_models(
            api_host, selected_model, service=service
        )
        return (
            gr.update(choices=choices, value=selected),
            gr.update(choices=reasoning_choices, value=reasoning),
            status,
        )

    def update_reasoning_for_model(api_host: str, selected_model: str | None):
        try:
            models = service.list_loaded_models(api_host)
        except Exception:
            models = []
        return (
            gr.update(
                choices=_reasoning_choices(models, selected_model),
                value=AUTO_REASONING,
            ),
            [],
            [],
            [],
            None,
        )

    def reset_chat():
        return [], [], [], None

    def respond(
        message: str,
        history: list[dict[str, str]],
        api_host: str,
        model_id: str | None,
        reasoning: str,
        conversation_state: ConversationState | None,
    ):
        yield from chat_response(
            message,
            history,
            api_host,
            model_id,
            reasoning,
            conversation_state,
            service=service,
        )

    with gr.Blocks(title="Poppy Chat", analytics_enabled=False) as app:
        gr.Markdown("# Poppy Chat")
        gr.Markdown("LM Studioでロードしたローカルモデルと会話します。")

        with gr.Row():
            api_host = gr.Textbox(
                label="LM Studio APIホスト (host:port)",
                value=DEFAULT_API_HOST,
                placeholder="localhost:1234",
            )
            model_id = gr.Dropdown(
                label="ロード済みモデル",
                choices=[],
                value=None,
                interactive=True,
            )
            reasoning = gr.Dropdown(
                label="Reasoning",
                choices=auto_reasoning_choices,
                value=AUTO_REASONING,
                interactive=True,
            )
            refresh_button = gr.Button("モデル一覧を更新", variant="secondary")

        status = gr.Markdown("LM Studioを起動し、モデルをロードしてから一覧を更新してください。")
        refresh_button.click(
            fn=update_model_choices,
            inputs=[api_host, model_id],
            outputs=[model_id, reasoning, status],
        )

        conversation_state = gr.State(value=None)
        chat = gr.ChatInterface(
            fn=respond,
            additional_inputs=[api_host, model_id, reasoning, conversation_state],
            additional_outputs=[conversation_state],
            title="チャット",
            analytics_enabled=False,
        )
        chat.chatbot.clear(
            fn=lambda: None,
            outputs=conversation_state,
            api_visibility="undocumented",
            queue=False,
        )
        model_id.change(
            fn=update_reasoning_for_model,
            inputs=[api_host, model_id],
            outputs=[
                reasoning,
                chat.chatbot,
                chat.chatbot_state,
                chat.chatbot_value,
                conversation_state,
            ],
            api_visibility="undocumented",
        )
        api_host.change(
            fn=reset_chat,
            outputs=[
                chat.chatbot,
                chat.chatbot_state,
                chat.chatbot_value,
                conversation_state,
            ],
            api_visibility="undocumented",
        )

    return app


def main() -> None:
    build_app().launch(server_name="127.0.0.1", inbrowser=True, share=False)
