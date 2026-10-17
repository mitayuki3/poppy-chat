import json

import pytest

from poppy_chat.system_prompts import SystemPromptPresetStore, default_preset_path


def test_store_saves_loads_updates_and_deletes_presets(tmp_path) -> None:
    store = SystemPromptPresetStore(tmp_path / "system_prompts.json")

    store.save("Translator", "Translate to Japanese.")
    store.save("Coder", "Return Python code.")
    store.save("Translator", "Translate to English.")

    assert store.load() == {
        "Translator": "Translate to English.",
        "Coder": "Return Python code.",
    }
    assert store.delete("Translator") is True
    assert store.delete("missing") is False
    assert store.load() == {"Coder": "Return Python code."}


def test_store_loads_empty_when_file_does_not_exist(tmp_path) -> None:
    store = SystemPromptPresetStore(tmp_path / "missing.json")

    assert store.load() == {}


@pytest.mark.parametrize("name", ["", "  "])
def test_store_rejects_empty_preset_name(tmp_path, name: str) -> None:
    store = SystemPromptPresetStore(tmp_path / "system_prompts.json")

    with pytest.raises(ValueError, match="名前"):
        store.save(name, "prompt")


def test_store_rejects_corrupt_file_without_overwriting_it(tmp_path) -> None:
    path = tmp_path / "system_prompts.json"
    path.write_text("not json", encoding="utf-8")
    store = SystemPromptPresetStore(path)

    with pytest.raises(ValueError, match="読み込めません"):
        store.load()

    assert path.read_text(encoding="utf-8") == "not json"


def test_default_preset_path_uses_windows_user_settings(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path))

    assert default_preset_path() == tmp_path / "PoppyChat" / "system_prompts.json"
