"""Gradio UI for Poppy Chat."""

from collections.abc import Iterator
from typing import TypedDict

from .lm_studio import (
    AUTO_REASONING,
    DEFAULT_API_HOST,
    DEFAULT_REASONING,
    ChatStreamEvent,
    LMStudioService,
    LoadedModel,
)
from .irodori_tts import (
    CHECKPOINT_CHOICES,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_CHECKPOINT,
    NO_VOICE_OPTION,
    OUTPUT_FORMATS,
    IrodoriTTSService,
)
from .system_prompts import SystemPromptPresetStore


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


def _default_reasoning(
    models: list[LoadedModel], selected_model: str | None
) -> str:
    available = {value for _, value in _reasoning_choices(models, selected_model)}
    return DEFAULT_REASONING if DEFAULT_REASONING in available else AUTO_REASONING


def _refresh_model_data(
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
    list[LoadedModel],
]:
    """Load models once and prepare all model-dependent UI values together."""
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
            [],
        )

    if not models:
        return (
            [],
            None,
            _reasoning_choices([], None),
            AUTO_REASONING,
            "接続しましたが、ロード済みのLLMがありません。LM Studioでモデルをロードしてください。",
            [],
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
        _default_reasoning(models, selected),
        f"接続しました（ロード済みモデル{len(models)}件）。",
        models,
    )


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
    return _refresh_model_data(api_host, selected_model, service=service)[:5]


def chat_response(
    message: str,
    history: list[dict[str, str]],
    api_host: str,
    model_id: str | None,
    reasoning: str,
    conversation_state: ConversationState | None,
    system_prompt: str = "",
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
            system_prompt=system_prompt,
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


def _optional_int(value: int | float | str | None, label: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}には整数を入力してください。") from error
    if not number.is_integer():
        raise ValueError(f"{label}には整数を入力してください。")
    return int(number)


def build_app(
    service: LMStudioService | None = None,
    preset_store: SystemPromptPresetStore | None = None,
    tts_service: IrodoriTTSService | None = None,
):
    """Create the Gradio app."""
    import gradio as gr

    service = service or LMStudioService()
    preset_store = preset_store or SystemPromptPresetStore()
    tts_service = tts_service or IrodoriTTSService()
    auto_reasoning_choices = [
        *_reasoning_choices([], None),
        (DEFAULT_REASONING, DEFAULT_REASONING),
    ]
    try:
        initial_presets = preset_store.load()
        initial_preset_status = ""
    except ValueError as error:
        initial_presets = {}
        initial_preset_status = str(error)

    def update_model_choices(api_host: str, selected_model: str | None):
        choices, selected, reasoning_choices, reasoning, status, models = (
            _refresh_model_data(api_host, selected_model, service=service)
        )
        return (
            gr.update(choices=choices, value=selected),
            gr.update(choices=reasoning_choices, value=reasoning),
            status,
            models,
        )

    def update_reasoning_for_model(
        selected_model: str | None, models: list[LoadedModel]
    ):
        return (
            gr.update(
                choices=_reasoning_choices(models, selected_model),
                value=_default_reasoning(models, selected_model),
            ),
            [],
            [],
            [],
            None,
            [],
            None,
            "",
            [],
        )

    def reset_chat():
        return [], [], [], None, [], None, "", []

    def update_voice_choices(selected_voice: str | None):
        choices = tts_service.voice_choices()
        value = selected_voice if selected_voice in choices else NO_VOICE_OPTION
        return gr.update(choices=choices, value=value), f"参照音声 {len(choices) - 1}件"

    def preload_tts_model(auto_read_enabled: bool, checkpoint: str):
        if not auto_read_enabled:
            return ""
        try:
            tts_service.load_model(checkpoint)
        except Exception as error:
            return f"起動時にIrodori-TTSモデルをロードできませんでした: {error}"
        return f"Irodori-TTSモデルをロードしました: `{checkpoint}`"

    def respond(
        message: str,
        history: list[dict[str, str]],
        api_host: str,
        model_id: str | None,
        reasoning: str,
        conversation_state: ConversationState | None,
        system_prompt: str,
        auto_read: bool,
    ):
        current_state = conversation_state
        response_text = ""
        for response_text, current_state in chat_response(
            message,
            history,
            api_host,
            model_id,
            reasoning,
            conversation_state,
            system_prompt,
            service=service,
        ):
            yield response_text, current_state, None

        has_new_response = bool(
            current_state
            and current_state.get("response_id")
            and response_text.strip()
            and "LM Studioに接続できません:" not in response_text
            and (
                not conversation_state
                or current_state["response_id"] != conversation_state["response_id"]
            )
        )
        tts_request = None
        if auto_read and has_new_response:
            tts_request = {
                "response_id": current_state["response_id"],
                "text": response_text,
            }
        yield response_text, current_state, tts_request

    def synthesize_response(
        request: dict[str, str] | None,
        checkpoint: str,
        voice_name: str,
        num_steps: int | float | None,
        cfg_scale_text: float,
        cfg_scale_caption: float,
        cfg_scale_speaker: float,
        caption: str,
        output_format: str,
        seed: int | float | None,
        audio_history: list[dict[str, str]] | None,
    ):
        if not request:
            return

        saved_audio = list(audio_history or [])
        yield (
            saved_audio,
            gr.update(),
            "音声を生成しています…",
            [item["path"] for item in saved_audio],
        )
        try:
            requested_steps = _optional_int(num_steps, "ステップ数")
            generated_audio = tts_service.synthesize(
                request["text"],
                checkpoint=checkpoint,
                voice_name=voice_name,
                num_steps=requested_steps,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_caption=cfg_scale_caption,
                cfg_scale_speaker=cfg_scale_speaker,
                caption=caption,
                seed=_optional_int(seed, "seed"),
                output_format=output_format,
            )
        except Exception as error:
            yield (
                saved_audio,
                gr.update(),
                f"読み上げを生成できませんでした: {error}",
                [item["path"] for item in saved_audio],
            )
            return

        saved_audio.append({"text": request["text"], "path": generated_audio})
        yield (
            saved_audio,
            generated_audio,
            "読み上げを生成しました。",
            [item["path"] for item in saved_audio],
        )

    def load_system_prompt_preset(name: str | None):
        if not name:
            return "", "", ""
        try:
            return name, preset_store.load().get(name, ""), ""
        except ValueError as error:
            return name, "", f"プリセットを読み込めません: {error}"

    def save_system_prompt_preset(name: str, prompt: str):
        try:
            presets = preset_store.save(name, prompt)
        except (ValueError, OSError) as error:
            return gr.update(), name, f"保存できません: {error}"
        normalized_name = name.strip()
        return (
            gr.update(choices=list(presets), value=normalized_name),
            normalized_name,
            f"「{normalized_name}」を保存しました。",
        )

    def delete_system_prompt_preset(name: str | None):
        if not name:
            return gr.update(), "", "", "削除するプリセットを選択してください。"
        try:
            deleted = preset_store.delete(name)
            presets = preset_store.load()
        except (ValueError, OSError) as error:
            return gr.update(), name, "", f"削除できません: {error}"
        message = f"「{name}」を削除しました。" if deleted else "プリセットが見つかりません。"
        return gr.update(choices=list(presets), value=None), "", "", message

    def refresh_system_prompt_presets():
        try:
            presets = preset_store.load()
            return gr.update(choices=list(presets), value=None), ""
        except ValueError as error:
            return gr.update(choices=[], value=None), f"プリセットを読み込めません: {error}"

    with gr.Blocks(title="Poppy Chat", analytics_enabled=False) as app:
        api_host = gr.Textbox(
            label="LM Studio APIホスト (host:port)",
            value=DEFAULT_API_HOST,
            placeholder="localhost:1234",
            render=False,
        )
        model_id = gr.Dropdown(
            label="ロード済みモデル",
            choices=[],
            value=None,
            interactive=True,
            render=False,
        )
        loaded_models_state = gr.State(value=[], render=False)
        reasoning = gr.Dropdown(
            label="Reasoning",
            choices=auto_reasoning_choices,
            value=DEFAULT_REASONING,
            interactive=True,
            render=False,
        )
        system_prompt = gr.Textbox(
            label="システムプロンプト",
            placeholder="モデルへの役割や応答方法を入力します。",
            lines=5,
            render=False,
        )
        conversation_state = gr.State(value=None)
        audio_history_state = gr.State(value=[])
        tts_request_state = gr.State(value=None)
        with gr.Accordion("Irodori-TTS 読み上げ設定", open=False):
            auto_read = gr.Checkbox(label="回答を自動で読み上げる", value=True)
            checkpoint = gr.Dropdown(
                label="モデル",
                choices=CHECKPOINT_CHOICES,
                value=DEFAULT_CHECKPOINT,
            )
            with gr.Row():
                voice_name = gr.Dropdown(
                    label="参照音声",
                    choices=tts_service.voice_choices(),
                    value=NO_VOICE_OPTION,
                    interactive=True,
                )
                refresh_voices_button = gr.Button("参照音声一覧を更新", variant="secondary")
            voice_status = gr.Markdown("voicesディレクトリの音声ファイルを使用できます。")
            caption = gr.Textbox(
                label="Voice Design キャプション（任意）",
                placeholder="例: 落ち着いた自然な声で、やわらかく話す",
                lines=2,
            )
            output_format = gr.Radio(
                label="出力形式",
                choices=OUTPUT_FORMATS,
                value=DEFAULT_OUTPUT_FORMAT,
            )
            with gr.Row():
                num_steps = gr.Number(
                    label="ステップ数（空欄でモデル既定値）",
                    value=20,
                    minimum=1,
                    maximum=100,
                    precision=0,
                )
                seed = gr.Number(label="Seed（空欄でランダム）", value=None, precision=0)
            with gr.Row():
                cfg_scale_text = gr.Slider(
                    label="Text CFG Scale", minimum=0, maximum=10, value=3, step=0.25
                )
                cfg_scale_caption = gr.Slider(
                    label="Caption CFG Scale", minimum=0, maximum=10, value=3, step=0.25
                )
                cfg_scale_speaker = gr.Slider(
                    label="Speaker CFG Scale", minimum=0, maximum=10, value=3, step=0.25
                )
        latest_audio = gr.Audio(
            label="最新の読み上げ",
            value=None,
            type="filepath",
            interactive=False,
            autoplay=True,
        )
        tts_status = gr.Markdown()
        audio_history_files = gr.Files(label="読み上げ履歴", file_count="multiple")
        with gr.Column(elem_classes="poppy-chatinterface"):
            chat = gr.ChatInterface(
                fn=respond,
                additional_inputs=[
                    api_host,
                    model_id,
                    reasoning,
                    conversation_state,
                    system_prompt,
                    auto_read,
                ],
                additional_outputs=[
                    conversation_state,
                    tts_request_state,
                ],
                title="Poppy Chat",
                analytics_enabled=False,
                concurrency_limit=1,
            )
        tts_request_state.change(
            fn=synthesize_response,
            inputs=[
                tts_request_state,
                checkpoint,
                voice_name,
                num_steps,
                cfg_scale_text,
                cfg_scale_caption,
                cfg_scale_speaker,
                caption,
                output_format,
                seed,
                audio_history_state,
            ],
            outputs=[
                audio_history_state,
                latest_audio,
                tts_status,
                audio_history_files,
            ],
            concurrency_limit=1,
            api_visibility="undocumented",
        )

        with gr.Row():
            refresh_button = gr.Button("モデル一覧を更新", variant="secondary")
            status = gr.Markdown(
                "LM Studioを起動し、モデルをロードしてから一覧を更新してください。"
            )
        refresh_button.click(
            fn=update_model_choices,
            inputs=[api_host, model_id],
            outputs=[model_id, reasoning, status, loaded_models_state],
        )
        refresh_voices_button.click(
            fn=update_voice_choices,
            inputs=voice_name,
            outputs=[voice_name, voice_status],
            api_visibility="undocumented",
        )
        app.load(
            fn=update_model_choices,
            inputs=[api_host, model_id],
            outputs=[model_id, reasoning, status, loaded_models_state],
            api_visibility="undocumented",
        )
        app.load(
            fn=preload_tts_model,
            inputs=[auto_read, checkpoint],
            outputs=tts_status,
            api_visibility="undocumented",
        )
        auto_read.change(
            fn=preload_tts_model,
            inputs=[auto_read, checkpoint],
            outputs=tts_status,
            api_visibility="undocumented",
        )
        checkpoint.change(
            fn=preload_tts_model,
            inputs=[auto_read, checkpoint],
            outputs=tts_status,
            api_visibility="undocumented",
        )

        with gr.Accordion("システムプロンプトのプリセット", open=False):
            with gr.Row():
                system_prompt_preset = gr.Dropdown(
                    label="プリセット",
                    choices=list(initial_presets),
                    value=None,
                    interactive=True,
                )
                system_prompt_name = gr.Textbox(
                    label="プリセット名",
                    placeholder="例: 翻訳アシスタント",
                )
            with gr.Row():
                save_preset_button = gr.Button("プリセットを保存", variant="secondary")
                delete_preset_button = gr.Button("選択中を削除", variant="secondary")
            preset_status = gr.Markdown(initial_preset_status)
            app.load(
                fn=refresh_system_prompt_presets,
                outputs=[system_prompt_preset, preset_status],
                api_visibility="undocumented",
            )
            system_prompt_preset.change(
                fn=load_system_prompt_preset,
                inputs=system_prompt_preset,
                outputs=[system_prompt_name, system_prompt, preset_status],
                api_visibility="undocumented",
            )
            save_preset_button.click(
                fn=save_system_prompt_preset,
                inputs=[system_prompt_name, system_prompt],
                outputs=[system_prompt_preset, system_prompt_name, preset_status],
                api_visibility="undocumented",
            )
            delete_preset_button.click(
                fn=delete_system_prompt_preset,
                inputs=system_prompt_preset,
                outputs=[
                    system_prompt_preset,
                    system_prompt_name,
                    system_prompt,
                    preset_status,
                ],
                api_visibility="undocumented",
            )
        chat.chatbot.clear(
            fn=lambda: (None, [], None, "", []),
            outputs=[
                conversation_state,
                audio_history_state,
                latest_audio,
                tts_status,
                audio_history_files,
            ],
            api_visibility="undocumented",
            queue=False,
        )
        model_id.input(
            fn=update_reasoning_for_model,
            inputs=[model_id, loaded_models_state],
            outputs=[
                reasoning,
                chat.chatbot,
                chat.chatbot_state,
                chat.chatbot_value,
                conversation_state,
                audio_history_state,
                latest_audio,
                tts_status,
                audio_history_files,
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
                audio_history_state,
                latest_audio,
                tts_status,
                audio_history_files,
            ],
            api_visibility="undocumented",
        )

    return app


def main() -> None:
    tts_service = IrodoriTTSService()
    tts_service.ensure_directories()
    build_app(tts_service=tts_service).launch(
        server_name="127.0.0.1",
        inbrowser=True,
        share=False,
        allowed_paths=[str(tts_service.output_dir), str(tts_service.voices_dir)],
        css="""
        .poppy-chatinterface,
        .poppy-chatinterface * {
            font-family: sans-serif !important;
        }
        """,
    )
