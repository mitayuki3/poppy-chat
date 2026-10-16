"""LM Studio integration through its native REST API."""

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any


DEFAULT_API_HOST = "localhost:1234"
AUTO_REASONING = "auto"
REASONING_OPTIONS = {"off", "on", "low", "medium", "high"}


@dataclass(frozen=True)
class LoadedModel:
    """A loaded LLM instance and its advertised reasoning settings."""

    identifier: str
    display_name: str
    reasoning_options: tuple[str, ...]
    reasoning_default: str | None


@dataclass(frozen=True)
class ChatStreamEvent:
    """A user-visible text fragment or the ID of the completed chat turn."""

    content: str = ""
    response_id: str | None = None


def normalize_api_host(value: str) -> str:
    """Validate the host:port value used to build the native API URL."""
    host = value.strip()
    if not host or "/" in host or "://" in host:
        raise ValueError("接続先は host:port 形式で入力してください（例: localhost:1234）。")

    host_part, separator, port = host.rpartition(":")
    if not separator or not host_part or not port.isdecimal():
        raise ValueError("接続先は host:port 形式で入力してください（例: localhost:1234）。")

    return host


class LMStudioService:
    """Small injectable adapter around LM Studio's native REST endpoints."""

    def __init__(self, client_factory: Callable[..., Any] | None = None) -> None:
        self._client_factory = client_factory

    def _new_client(self, api_host: str) -> Any:
        host = normalize_api_host(api_host)
        if self._client_factory is None:
            import httpx

            client_factory = httpx.Client
        else:
            client_factory = self._client_factory
        return client_factory(base_url=f"http://{host}", timeout=60.0)

    def list_loaded_models(self, api_host: str) -> list[LoadedModel]:
        """Return loaded LLM instances and their advertised reasoning options."""
        with self._new_client(api_host) as client:
            response = client.get("/api/v1/models")
            response.raise_for_status()
            payload = response.json()

        loaded_models: list[LoadedModel] = []
        for model in payload.get("models", []):
            if model.get("type") != "llm":
                continue
            capabilities = model.get("capabilities") or {}
            reasoning = capabilities.get("reasoning") or {}
            options = tuple(
                option
                for option in reasoning.get("allowed_options", [])
                if option in REASONING_OPTIONS
            )
            for instance in model.get("loaded_instances", []):
                identifier = instance.get("id")
                if not identifier:
                    continue
                loaded_models.append(
                    LoadedModel(
                        identifier=identifier,
                        display_name=model.get("display_name") or identifier,
                        reasoning_options=options,
                        reasoning_default=reasoning.get("default"),
                    )
                )
        return loaded_models

    def stream_reply(
        self,
        *,
        api_host: str,
        model_id: str,
        message: str,
        previous_response_id: str | None = None,
        reasoning: str = AUTO_REASONING,
    ) -> Iterator[ChatStreamEvent]:
        """Stream visible message deltas and return the final response ID."""
        if not model_id:
            raise ValueError("モデルを選択してください。")

        request: dict[str, Any] = {
            "model": model_id,
            "input": message,
            "stream": True,
        }
        if reasoning != AUTO_REASONING:
            request["reasoning"] = reasoning
        if previous_response_id:
            request["previous_response_id"] = previous_response_id

        with self._new_client(api_host) as client:
            with client.stream("POST", "/api/v1/chat", json=request) as response:
                response.raise_for_status()
                event_name: str | None = None
                data_lines: list[str] = []

                for line in response.iter_lines():
                    if not line:
                        update = self._decode_sse_event(event_name, data_lines)
                        if update is not None:
                            yield update
                        event_name = None
                        data_lines = []
                    elif line.startswith("event:"):
                        event_name = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].strip())

                update = self._decode_sse_event(event_name, data_lines)
                if update is not None:
                    yield update

    @staticmethod
    def _decode_sse_event(
        event_name: str | None, data_lines: list[str]
    ) -> ChatStreamEvent | None:
        if not event_name or not data_lines:
            return None
        try:
            data = json.loads("\n".join(data_lines))
        except json.JSONDecodeError as error:
            raise RuntimeError("LM Studioから不正なストリーム応答を受け取りました。") from error

        if event_name == "message.delta":
            content = data.get("content")
            return ChatStreamEvent(content=content) if isinstance(content, str) else None
        if event_name == "error":
            detail = (data.get("error") or {}).get("message", "不明なエラー")
            raise RuntimeError(detail)
        if event_name == "chat.end":
            response_id = (data.get("result") or {}).get("response_id")
            return ChatStreamEvent(response_id=response_id) if response_id else None
        return None
