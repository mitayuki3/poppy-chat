from collections.abc import Iterator
import inspect

from poppy_chat.app import build_app, chat_response, refresh_models
from poppy_chat.lm_studio import ChatStreamEvent, LoadedModel


class FakeService:
    def list_loaded_models(self, api_host: str) -> list[LoadedModel]:
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
    ) -> Iterator[ChatStreamEvent]:
        if api_host == "offline:1234":
            raise ConnectionError("server not found")
        self.last_previous_response_id = previous_response_id
        self.last_reasoning = reasoning
        yield ChatStreamEvent(content="Hi")
        yield ChatStreamEvent(content=" there")
        yield ChatStreamEvent(response_id="resp-next")


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
    assert selected_reasoning == "auto"


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
    app = build_app(service=FakeService())

    assert app.__class__.__name__ == "Blocks"
    respond = next(
        inspect.unwrap(block_fn.fn)
        for block_fn in app.fns.values()
        if getattr(inspect.unwrap(block_fn.fn), "__name__", "") == "respond"
    )
    assert inspect.isgeneratorfunction(respond)
