import json
from contextlib import AbstractContextManager
from typing import Any

import pytest

from poppy_chat.lm_studio import (
    ChatStreamEvent,
    LMStudioService,
    normalize_api_host,
)


class FakeResponse:
    def __init__(self, payload: dict[str, Any] | None = None, lines: list[str] | None = None):
        self.payload = payload or {}
        self.lines = lines or []

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload

    def iter_lines(self):
        yield from self.lines


class FakeHttpClient(AbstractContextManager["FakeHttpClient"]):
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.get_url: str | None = None
        self.post_url: str | None = None
        self.post_json: dict[str, Any] | None = None
        self.get_response = FakeResponse()
        self.stream_response = FakeResponse()

    def __enter__(self) -> "FakeHttpClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True

    def get(self, url: str) -> FakeResponse:
        self.get_url = url
        return self.get_response

    def stream(self, method: str, url: str, *, json: dict[str, Any]):
        assert method == "POST"
        self.post_url = url
        self.post_json = json
        return self.stream_response


def make_service(
    *, models: dict[str, Any] | None = None, lines: list[str] | None = None
) -> tuple[LMStudioService, FakeHttpClient]:
    client = FakeHttpClient()
    client.get_response = FakeResponse(models)
    client.stream_response = FakeResponse(lines=lines)

    def client_factory(**kwargs: Any) -> FakeHttpClient:
        client.kwargs.update(kwargs)
        return client

    service = LMStudioService(client_factory=client_factory)
    return service, client


def sse(event: str, data: dict[str, Any]) -> list[str]:
    return [f"event: {event}", f"data: {json.dumps(data)}", ""]


def test_normalize_api_host_accepts_host_and_port() -> None:
    assert normalize_api_host("localhost:1234") == "localhost:1234"


@pytest.mark.parametrize("value", ["", "http://localhost:1234", "localhost:1234/v1"])
def test_normalize_api_host_rejects_non_host_port_values(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_api_host(value)


def test_list_loaded_models_returns_loaded_llms_with_reasoning_options() -> None:
    service, client = make_service(
        models={
            "models": [
                {
                    "type": "llm",
                    "key": "publisher/model",
                    "display_name": "Model",
                    "capabilities": {
                        "reasoning": {
                            "allowed_options": ["off", "low"],
                            "default": "low",
                        }
                    },
                    "loaded_instances": [{"id": "publisher/model"}],
                },
                {
                    "type": "llm",
                    "key": "unloaded/model",
                    "loaded_instances": [],
                },
                {
                    "type": "embedding",
                    "key": "embedding/model",
                    "loaded_instances": [{"id": "embedding/model"}],
                },
            ]
        }
    )

    models = service.list_loaded_models("127.0.0.1:1234")

    assert [(model.identifier, model.display_name) for model in models] == [
        ("publisher/model", "Model")
    ]
    assert models[0].reasoning_options == ("off", "low")
    assert models[0].reasoning_default == "low"
    assert client.kwargs["base_url"] == "http://127.0.0.1:1234"
    assert client.get_url == "/api/v1/models"
    assert client.closed


def test_stream_reply_sends_reasoning_and_previous_response_id() -> None:
    lines = (
        sse("reasoning.delta", {"content": "hidden"})
        + sse("message.delta", {"content": "Hello"})
        + sse("message.delta", {"content": "!"})
        + sse("chat.end", {"result": {"response_id": "resp-next"}})
    )
    service, client = make_service(lines=lines)

    updates = list(
        service.stream_reply(
            api_host="localhost:1234",
            model_id="qwen-7b",
            message="How are you?",
            previous_response_id="resp-previous",
            reasoning="off",
            system_prompt="Answer concisely.",
        )
    )

    assert updates == [
        ChatStreamEvent(content="Hello"),
        ChatStreamEvent(content="!"),
        ChatStreamEvent(response_id="resp-next"),
    ]
    assert client.post_url == "/api/v1/chat"
    assert client.post_json == {
        "model": "qwen-7b",
        "input": "How are you?",
        "stream": True,
        "reasoning": "off",
        "previous_response_id": "resp-previous",
        "system_prompt": "Answer concisely.",
    }
    assert client.closed


def test_stream_reply_omits_reasoning_for_auto_mode() -> None:
    service, client = make_service(lines=sse("chat.end", {"result": {}}))

    list(
        service.stream_reply(
            api_host="localhost:1234",
            model_id="qwen-7b",
            message="Hello",
            reasoning="auto",
        )
    )

    assert client.post_json is not None
    assert "reasoning" not in client.post_json


def test_stream_reply_omits_empty_system_prompt() -> None:
    service, client = make_service(lines=sse("chat.end", {"result": {}}))

    list(
        service.stream_reply(
            api_host="localhost:1234",
            model_id="qwen-7b",
            message="Hello",
            system_prompt="  ",
        )
    )

    assert client.post_json is not None
    assert "system_prompt" not in client.post_json


def test_stream_reply_raises_error_events() -> None:
    service, _ = make_service(
        lines=sse(
            "error",
            {"error": {"message": "reasoning setting unsupported"}},
        )
    )

    with pytest.raises(RuntimeError, match="reasoning setting unsupported"):
        list(
            service.stream_reply(
                api_host="localhost:1234",
                model_id="qwen-7b",
                message="Hello",
            )
        )


def test_stream_reply_reports_no_model_when_model_id_is_empty() -> None:
    service, _ = make_service()

    with pytest.raises(ValueError, match="モデルを選択"):
        list(
            service.stream_reply(
                api_host="localhost:1234",
                model_id="",
                message="Hello",
            )
        )
