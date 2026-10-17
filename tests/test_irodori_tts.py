from pathlib import Path
import sys
from types import ModuleType

import pytest

from poppy_chat.irodori_tts import IrodoriTTSService, NO_VOICE_OPTION


def test_dotenv_configures_directories_and_system_environment_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("POPPY_CHAT_TTS_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("POPPY_CHAT_TTS_VOICES_DIR", raising=False)
    (tmp_path / ".env").write_text(
        'POPPY_CHAT_TTS_OUTPUT_DIR="generated audio"\n'
        "POPPY_CHAT_TTS_VOICES_DIR=local-voices\n",
        encoding="utf-8",
    )

    service = IrodoriTTSService()

    assert service.output_dir == (tmp_path / "generated audio").resolve()
    assert service.voices_dir == (tmp_path / "local-voices").resolve()

    monkeypatch.setenv("POPPY_CHAT_TTS_OUTPUT_DIR", "environment-output")
    override_service = IrodoriTTSService()
    assert override_service.output_dir == (tmp_path / "environment-output").resolve()


def test_voice_choices_are_relative_and_ignore_unsupported_files(tmp_path: Path) -> None:
    voices_dir = tmp_path / "voices"
    (voices_dir / "nested").mkdir(parents=True)
    (voices_dir / "nested" / "voice.wav").write_bytes(b"wav")
    (voices_dir / "notes.txt").write_text("not audio", encoding="utf-8")
    service = IrodoriTTSService(voices_dir=voices_dir)

    assert service.voice_choices() == [NO_VOICE_OPTION, "nested/voice.wav"]


def test_voice_resolver_rejects_paths_outside_voice_directory(tmp_path: Path) -> None:
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (tmp_path / "outside.wav").write_bytes(b"wav")
    service = IrodoriTTSService(voices_dir=voices_dir)

    with pytest.raises(ValueError, match="外"):
        service._resolve_voice("../outside.wav")


def test_synthesize_uses_cached_runtime_and_saves_compressed_ogg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POPPY_CHAT_TTS_AUDIO_BITRATE", "192000")
    runtime_calls: list[object] = []
    save_calls: list[tuple[object, object, int, int]] = []

    class AudioTensor:
        def cpu(self):
            return "cpu-audio-tensor"

    class RuntimeKey:
        def __init__(self, **kwargs) -> None:
            self.values = kwargs

    class SamplingRequest:
        def __init__(self, **kwargs) -> None:
            self.values = kwargs

    class Runtime:
        def synthesize(self, request, *, log_fn):
            runtime_calls.append(request)
            return type("Result", (), {"audio": AudioTensor(), "sample_rate": 48000})()

    runtime_module = ModuleType("irodori_tts.inference_runtime")
    runtime_module.RuntimeKey = RuntimeKey
    runtime_module.SamplingRequest = SamplingRequest
    runtime_module.default_runtime_device = lambda: "cuda"
    runtime_module.download_hf_checkpoint = lambda checkpoint: "C:/models/model.safetensors"
    runtime_module.get_cached_runtime = lambda key: (runtime_calls.append(key) or Runtime(), True)
    torchaudio_module = ModuleType("torchaudio")
    torchaudio_module.save = lambda path, audio, rate, *, compression: save_calls.append(
        (path, audio, rate, compression)
    )
    package_module = ModuleType("irodori_tts")
    package_module.__path__ = []
    monkeypatch.setitem(sys.modules, "irodori_tts", package_module)
    monkeypatch.setitem(sys.modules, "irodori_tts.inference_runtime", runtime_module)
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio_module)

    service = IrodoriTTSService(output_dir=Path.cwd())
    saved_path = service.synthesize(
        "こんにちは",
        checkpoint="Aratako/model",
        voice_name=NO_VOICE_OPTION,
        num_steps=8,
        cfg_scale_text=2.5,
        cfg_scale_caption=3.5,
        cfg_scale_speaker=4.5,
        caption="やさしい声",
        seed=12,
    )

    runtime_key, request = runtime_calls
    assert runtime_key.values["checkpoint"] == "C:/models/model.safetensors"
    assert runtime_key.values["model_device"] == "cuda"
    assert runtime_key.values["model_precision"] == "bf16"
    assert request.values == {
        "text": "こんにちは",
        "caption": "やさしい声",
        "ref_wav": None,
        "no_ref": True,
        "num_steps": 8,
        "cfg_scale_text": 2.5,
        "cfg_scale_caption": 3.5,
        "cfg_scale_speaker": 4.5,
        "seed": 12,
    }
    assert save_calls[0][1:] == ("cpu-audio-tensor", 48000, 192000)
    assert Path(saved_path).parent.name == "no_voice"
    assert Path(saved_path).name.endswith(".ogg")
    assert len(Path(saved_path).stem) == len("20260101T123456123456")
