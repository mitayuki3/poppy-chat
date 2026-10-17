from collections.abc import Iterator
import inspect

from poppy_chat.app import build_app, chat_response, refresh_models
from poppy_chat.irodori_tts import DEFAULT_CHECKPOINT, NO_VOICE_OPTION
from poppy_chat.lm_studio import ChatStreamEvent, LoadedModel


class FakeService:
    def __init__(self) -> None:
        self.model_list_calls = 0

    def list_loaded_models(self, api_host: str) -> list[LoadedModel]:
        self.model_list_calls += 1
        if api_host == "offline:1234":
            raise ConnectionError("server not found")
        return [
            LoadedModel("qwen-7b", "Qwen 7B", ("off", "low"), "low"),
            LoadedModel("llama-8b", "Llama 8B", (), None),
        ]

    def stream_reply(
        self,
        *,
        api_host: str,
        model_id: str,
        message: str,
        previous_response_id: str | None = None,
        reasoning: str = "auto",
        system_prompt: str = "",
    ) -> Iterator[ChatStreamEvent]:
        if api_host == "offline:1234":
            raise ConnectionError("server not found")
        self.last_previous_response_id = previous_response_id
        self.last_reasoning = reasoning
        self.last_system_prompt = system_prompt
        yield ChatStreamEvent(content="Hi")
        yield ChatStreamEvent(content=" there")
        yield ChatStreamEvent(response_id="resp-next")


class FakeTTSService:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.error = error

    def voice_choices(self) -> list[str]:
        return [NO_VOICE_OPTION, "sample.wav"]

    def synthesize(self, text: str, **kwargs) -> str:
        self.calls.append((text, kwargs))
        if self.error:
            raise self.error
        return "C:/tts-output/reply.wav"


def test_refresh_models_returns_choices_and_reasoning_options() -> None:
    choices, selected, reasoning_choices, selected_reasoning, status = refresh_models(
        "localhost:1234", "llama-8b", service=FakeService()
    )

    assert choices == [("Qwen 7B", "qwen-7b"), ("Llama 8B", "llama-8b")]
    assert selected == "llama-8b"
    assert reasoning_choices == [("自動（LM Studioの既定値）", "auto")]
    assert selected_reasoning == "auto"
    assert "2件" in status


def test_refresh_models_preserves_reasoning_options_for_selected_model() -> None:
    _, selected, reasoning_choices, selected_reasoning, _ = refresh_models(
        "localhost:1234", "qwen-7b", service=FakeService()
    )

    assert selected == "qwen-7b"
    assert reasoning_choices == [
        ("自動（LM Studioの既定値）", "auto"),
        ("off", "off"),
        ("low", "low"),
    ]
    assert selected_reasoning == "off"


def test_refresh_models_reports_connection_error() -> None:
    choices, selected, reasoning, selected_reasoning, status = refresh_models(
        "offline:1234", None, service=FakeService()
    )

    assert choices == []
    assert selected is None
    assert reasoning == [("自動（LM Studioの既定値）", "auto")]
    assert selected_reasoning == "auto"
    assert "接続できません" in status
    assert "server not found" in status


def test_refresh_models_reports_when_no_llm_is_loaded() -> None:
    class EmptyService(FakeService):
        def list_loaded_models(self, api_host: str) -> list[LoadedModel]:
            return []

    choices, selected, reasoning, selected_reasoning, status = refresh_models(
        "localhost:1234", None, service=EmptyService()
    )

    assert choices == []
    assert selected is None
    assert reasoning == [("自動（LM Studioの既定値）", "auto")]
    assert selected_reasoning == "auto"
    assert "ロード済みのLLMがありません" in status


def test_chat_response_streams_and_updates_stateful_conversation() -> None:
    service = FakeService()
    previous_state = {
        "response_id": "resp-old",
        "api_host": "localhost:1234",
        "model_id": "qwen-7b",
    }

    result = list(
        chat_response(
            "hello",
            [{"role": "user", "content": "previous"}],
            "localhost:1234",
            "qwen-7b",
            "off",
            previous_state,
            "Use short answers.",
            service=service,
        )
    )

    expected_state = {
        "response_id": "resp-next",
        "api_host": "localhost:1234",
        "model_id": "qwen-7b",
    }
    assert [text for text, _ in result] == ["Hi", "Hi there", "Hi there"]
    assert result[-1][1] == expected_state
    assert service.last_previous_response_id == "resp-old"
    assert service.last_reasoning == "off"
    assert service.last_system_prompt == "Use short answers."


def test_chat_response_starts_new_thread_when_model_changes() -> None:
    service = FakeService()
    previous_state = {
        "response_id": "resp-old",
        "api_host": "localhost:1234",
        "model_id": "qwen-7b",
    }

    list(
        chat_response(
            "hello",
            [{"role": "user", "content": "previous"}],
            "localhost:1234",
            "llama-8b",
            "auto",
            previous_state,
            service=service,
        )
    )

    assert service.last_previous_response_id is None


def test_chat_response_shows_connection_error_and_keeps_state() -> None:
    previous_state = {
        "response_id": "resp-old",
        "api_host": "offline:1234",
        "model_id": "qwen-7b",
    }

    result = list(
        chat_response(
            "hello",
            [],
            "offline:1234",
            "qwen-7b",
            "auto",
            previous_state,
            service=FakeService(),
        )
    )

    assert len(result) == 1
    assert "LM Studioに接続できません" in result[0][0]
    assert "server not found" in result[0][0]
    assert result[0][1] is None


def test_chat_response_prompts_for_model_selection() -> None:
    result = list(
        chat_response(
            "hello", [], "localhost:1234", None, "auto", None, service=FakeService()
        )
    )

    assert result == [("モデルを選択してください。上の「モデル一覧を更新」から接続してください。", None)]


def test_build_app_returns_gradio_blocks_with_generator_callback() -> None:
    app = build_app(service=FakeService(), tts_service=FakeTTSService())

    assert app.__class__.__name__ == "Blocks"
    respond = next(
        inspect.unwrap(block_fn.fn)
        for block_fn in app.fns.values()
        if getattr(inspect.unwrap(block_fn.fn), "__name__", "") == "respond"
    )
    assert inspect.isgeneratorfunction(respond)


def test_build_app_refreshes_models_when_browser_loads() -> None:
    app = build_app(service=FakeService(), tts_service=FakeTTSService())
    function_indexes = {
        index
        for index, block_fn in app.fns.items()
        if getattr(inspect.unwrap(block_fn.fn), "__name__", "")
        == "update_model_choices"
    }

    assert any(
        dependency["id"] in function_indexes
        and any(event == "load" for _, event in dependency["targets"])
        for dependency in app.config["dependencies"]
    )


def test_model_and_reasoning_choices_use_the_same_loaded_model_data() -> None:
    service = FakeService()
    app = build_app(service=service, tts_service=FakeTTSService())
    update_choices = next(
        inspect.unwrap(block_fn.fn)
        for block_fn in app.fns.values()
        if getattr(inspect.unwrap(block_fn.fn), "__name__", "")
        == "update_model_choices"
    )

    model_update, reasoning_update, status, loaded_models = update_choices(
        "localhost:1234", None
    )

    assert model_update["choices"] == [
        ("Qwen 7B", "qwen-7b"),
        ("Llama 8B", "llama-8b"),
    ]
    assert reasoning_update["choices"] == [
        ("自動（LM Studioの既定値）", "auto"),
        ("off", "off"),
        ("low", "low"),
    ]
    assert reasoning_update["value"] == "off"
    assert len(loaded_models) == 2
    assert "2件" in status
    assert service.model_list_calls == 1


def test_reasoning_ui_starts_with_off_selected() -> None:
    app = build_app(service=FakeService(), tts_service=FakeTTSService())
    reasoning = next(
        component
        for component in app.blocks.values()
        if getattr(component, "label", None) == "Reasoning"
    )

    assert reasoning.value == "off"
    assert ("off", "off") in reasoning.choices


def test_model_selection_uses_cached_models_instead_of_refetching() -> None:
    service = FakeService()
    app = build_app(service=service, tts_service=FakeTTSService())
    update_choices = next(
        inspect.unwrap(block_fn.fn)
        for block_fn in app.fns.values()
        if getattr(inspect.unwrap(block_fn.fn), "__name__", "")
        == "update_model_choices"
    )
    _, _, _, loaded_models = update_choices("localhost:1234", None)
    update_reasoning = next(
        inspect.unwrap(block_fn.fn)
        for block_fn in app.fns.values()
        if getattr(inspect.unwrap(block_fn.fn), "__name__", "")
        == "update_reasoning_for_model"
    )

    reasoning_update, *_ = update_reasoning("qwen-7b", loaded_models)

    assert reasoning_update["choices"] == [
        ("自動（LM Studioの既定値）", "auto"),
        ("off", "off"),
        ("low", "low"),
    ]
    assert service.model_list_calls == 1
    update_reasoning_indexes = {
        index
        for index, block_fn in app.fns.items()
        if getattr(inspect.unwrap(block_fn.fn), "__name__", "")
        == "update_reasoning_for_model"
    }
    assert any(
        dependency["id"] in update_reasoning_indexes
        and any(event == "input" for _, event in dependency["targets"])
        for dependency in app.config["dependencies"]
    )


def test_build_app_refreshes_presets_when_browser_loads() -> None:
    class MutablePresetStore:
        presets: dict[str, str] = {}

        def load(self) -> dict[str, str]:
            return self.presets.copy()

    store = MutablePresetStore()
    app = build_app(
        service=FakeService(), preset_store=store, tts_service=FakeTTSService()
    )
    refresh = next(
        inspect.unwrap(block_fn.fn)
        for block_fn in app.fns.values()
        if getattr(inspect.unwrap(block_fn.fn), "__name__", "")
        == "refresh_system_prompt_presets"
    )
    store.presets["Saved preset"] = "Use concise answers."

    choices_update, status = refresh()

    assert choices_update["choices"] == ["Saved preset"]
    assert choices_update["value"] is None
    assert status == ""


def test_respond_completes_before_queuing_audio_generation() -> None:
    tts_service = FakeTTSService()
    app = build_app(service=FakeService(), tts_service=tts_service)
    respond = next(
        inspect.unwrap(block_fn.fn)
        for block_fn in app.fns.values()
        if getattr(inspect.unwrap(block_fn.fn), "__name__", "") == "respond"
    )

    result = list(
        respond(
            "hello",
            [],
            "localhost:1234",
            "qwen-7b",
            "off",
            None,
            "",
            True,
        )
    )

    assert result[-1] == (
        "Hi there",
        {
            "response_id": "resp-next",
            "api_host": "localhost:1234",
            "model_id": "qwen-7b",
        },
        {"response_id": "resp-next", "text": "Hi there"},
    )
    assert tts_service.calls == []

    synthesize_response = next(
        inspect.unwrap(block_fn.fn)
        for block_fn in app.fns.values()
        if getattr(inspect.unwrap(block_fn.fn), "__name__", "")
        == "synthesize_response"
    )
    audio_result = list(
        synthesize_response(
            result[-1][2], DEFAULT_CHECKPOINT, NO_VOICE_OPTION, None, 3, 3, 5,
            "", "ogg", None, [],
        )
    )

    assert audio_result[0][2] == "音声を生成しています…"
    assert audio_result[-1] == (
        [{"text": "Hi there", "path": "C:/tts-output/reply.wav"}],
        "C:/tts-output/reply.wav",
        "読み上げを生成しました。",
        ["C:/tts-output/reply.wav"],
    )
    assert tts_service.calls == [
        (
            "Hi there",
            {
                "checkpoint": DEFAULT_CHECKPOINT,
                "voice_name": NO_VOICE_OPTION,
                "num_steps": None,
                "cfg_scale_text": 3,
                "cfg_scale_caption": 3,
                "cfg_scale_speaker": 5,
                "caption": "",
                "seed": None,
                "output_format": "ogg",
            },
        )
    ]


def test_audio_generation_error_keeps_completed_answer() -> None:
    tts_service = FakeTTSService(error=RuntimeError("model unavailable"))
    app = build_app(service=FakeService(), tts_service=tts_service)
    synthesize_response = next(
        inspect.unwrap(block_fn.fn)
        for block_fn in app.fns.values()
        if getattr(inspect.unwrap(block_fn.fn), "__name__", "")
        == "synthesize_response"
    )

    result = list(
        synthesize_response(
            {"response_id": "resp-next", "text": "Hi there"},
            DEFAULT_CHECKPOINT, NO_VOICE_OPTION, None, 3, 3, 5,
            "", "ogg", None, [],
        )
    )

    assert "model unavailable" in result[-1][2]
    assert result[-1][0] == []


def test_respond_skips_tts_when_auto_read_is_disabled() -> None:
    tts_service = FakeTTSService()
    app = build_app(service=FakeService(), tts_service=tts_service)
    respond = next(
        inspect.unwrap(block_fn.fn)
        for block_fn in app.fns.values()
        if getattr(inspect.unwrap(block_fn.fn), "__name__", "") == "respond"
    )

    result = list(
        respond(
            "hello", [], "localhost:1234", "qwen-7b", "off", None, "", False,
        )
    )

    assert result[-1][0] == "Hi there"
    assert result[-1][2] is None
    assert tts_service.calls == []
