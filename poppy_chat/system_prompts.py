"""Persistent storage for named system prompt presets."""

import json
import os
from pathlib import Path


def default_preset_path() -> Path:
    """Return the per-user location used for system prompt presets."""
    app_data = os.environ.get("APPDATA")
    if app_data:
        root = Path(app_data)
    else:
        root = Path.home() / ".config"
    return root / "PoppyChat" / "system_prompts.json"


class SystemPromptPresetStore:
    """Read and update a JSON mapping of preset names to prompt text."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_preset_path()

    def load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"プリセットを読み込めません: {error}") from error
        if not isinstance(data, dict) or not all(
            isinstance(name, str) and isinstance(prompt, str)
            for name, prompt in data.items()
        ):
            raise ValueError("プリセットファイルの形式が正しくありません。")
        return data

    def save(self, name: str, prompt: str) -> dict[str, str]:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("プリセットの名前を入力してください。")
        if not prompt.strip():
            raise ValueError("システムプロンプトを入力してください。")

        presets = self.load()
        presets[normalized_name] = prompt
        self._write(presets)
        return presets

    def delete(self, name: str) -> bool:
        presets = self.load()
        if name not in presets:
            return False
        del presets[name]
        self._write(presets)
        return True

    def _write(self, presets: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(presets, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.path)
