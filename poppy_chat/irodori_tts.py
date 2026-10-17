"""Local Irodori-TTS integration used by the chat UI."""

from __future__ import annotations

import os
import re
import threading
from datetime import datetime
from pathlib import Path


DEFAULT_CHECKPOINT = "Aratako/Irodori-TTS-v4.1-Small"
CHECKPOINT_CHOICES = [
    ("4.1 Small", DEFAULT_CHECKPOINT),
    ("4 Large", "Aratako/Irodori-TTS-v4-Large"),
]
NO_VOICE_OPTION = "無指定"
VOICE_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
OUTPUT_FORMATS = ("ogg", "wav", "mp3", "flac")
DEFAULT_OUTPUT_FORMAT = "ogg"
_DOTENV_LOCK = threading.Lock()


def _load_dotenv() -> None:
    """Load simple KEY=VALUE settings from the workspace .env, without overriding env."""
    with _DOTENV_LOCK:
        env_path = Path.cwd() / ".env"
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8-sig").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("export "):
                    stripped = stripped[7:].lstrip()
                key, separator, value = stripped.partition("=")
                key = key.strip()
                if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                    continue
                if key in os.environ:
                    continue
                value = value.strip()
                if value and value[0] in "\"'":
                    quote = value[0]
                    closing_quote = value.find(quote, 1)
                    if closing_quote >= 0:
                        value = value[1:closing_quote]
                else:
                    value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
                os.environ[key] = value


def _configured_directory(variable: str, default: str) -> Path:
    _load_dotenv()
    value = os.environ.get(variable, default).strip()
    if not value:
        value = default
    return Path(value).expanduser().resolve()


class IrodoriTTSService:
    """Lazily load Irodori-TTS and synthesize chat responses to WAV files."""

    def __init__(
        self,
        output_dir: Path | str | None = None,
        voices_dir: Path | str | None = None,
    ) -> None:
        self.output_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else _configured_directory("POPPY_CHAT_TTS_OUTPUT_DIR", "outputs")
        )
        self.voices_dir = (
            Path(voices_dir).expanduser().resolve()
            if voices_dir is not None
            else _configured_directory("POPPY_CHAT_TTS_VOICES_DIR", "voices")
        )
        _load_dotenv()
        try:
            self.audio_bitrate = int(os.environ.get("POPPY_CHAT_TTS_AUDIO_BITRATE", "192000"))
        except ValueError as error:
            raise ValueError("POPPY_CHAT_TTS_AUDIO_BITRATEには正の整数を指定してください。") from error
        if self.audio_bitrate <= 0:
            raise ValueError("POPPY_CHAT_TTS_AUDIO_BITRATEには正の整数を指定してください。")
        self._generation_lock = threading.Lock()

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    def voice_choices(self) -> list[str]:
        if not self.voices_dir.exists():
            return [NO_VOICE_OPTION]
        voices = sorted(
            path.relative_to(self.voices_dir).as_posix()
            for path in self.voices_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VOICE_EXTENSIONS
        )
        return [NO_VOICE_OPTION, *voices]

    def _resolve_voice(self, voice_name: str | None) -> str | None:
        if not voice_name or voice_name == NO_VOICE_OPTION:
            return None

        candidate = (self.voices_dir / voice_name).resolve()
        try:
            candidate.relative_to(self.voices_dir)
        except ValueError as error:
            raise ValueError("参照音声のパスがvoicesディレクトリ外です。") from error
        if not candidate.is_file() or candidate.suffix.lower() not in VOICE_EXTENSIONS:
            raise ValueError("選択した参照音声が見つかりません。一覧を更新してください。")
        return str(candidate)

    @staticmethod
    def _resolve_checkpoint(checkpoint: str) -> str:
        value = checkpoint.strip()
        if not value:
            raise ValueError("Irodori-TTSのチェックポイントを入力してください。")
        path = Path(value).expanduser()
        if path.suffix.lower() in {".pt", ".safetensors"}:
            resolved = path.resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"チェックポイントが見つかりません: {resolved}")
            return str(resolved)

        from irodori_tts.inference_runtime import download_hf_checkpoint

        return str(download_hf_checkpoint(value))

    def _load_runtime(self, checkpoint: str):
        from irodori_tts.inference_runtime import (
            RuntimeKey,
            default_runtime_device,
            get_cached_runtime,
        )

        device = default_runtime_device()
        precision = "bf16" if device in {"cuda", "xpu"} else "fp32"
        checkpoint_path = self._resolve_checkpoint(checkpoint)
        runtime_key = RuntimeKey(
            checkpoint=checkpoint_path,
            model_device=device,
            model_precision=precision,
            codec_device=device,
            codec_precision=precision,
        )
        return get_cached_runtime(runtime_key)

    def load_model(self, checkpoint: str = DEFAULT_CHECKPOINT) -> None:
        """Download and load a checkpoint into the shared runtime cache."""
        with self._generation_lock:
            self._load_runtime(checkpoint)

    def synthesize(
        self,
        text: str,
        *,
        checkpoint: str = DEFAULT_CHECKPOINT,
        voice_name: str = NO_VOICE_OPTION,
        num_steps: int | None = None,
        cfg_scale_text: float = 3.0,
        cfg_scale_caption: float = 3.0,
        cfg_scale_speaker: float = 5.0,
        caption: str = "",
        seed: int | None = None,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
    ) -> str:
        spoken_text = text.strip()
        if not spoken_text:
            raise ValueError("読み上げるテキストがありません。")

        if output_format not in OUTPUT_FORMATS:
            raise ValueError(f"未対応の音声形式です: {output_format}")

        import torchaudio
        from irodori_tts.inference_runtime import (
            SamplingRequest,
        )

        ref_wav = self._resolve_voice(voice_name)

        with self._generation_lock:
            runtime, _ = self._load_runtime(checkpoint)
            result = runtime.synthesize(
                SamplingRequest(
                    text=spoken_text,
                    caption=caption.strip() or None,
                    ref_wav=ref_wav,
                    no_ref=ref_wav is None,
                    num_steps=num_steps,
                    cfg_scale_text=float(cfg_scale_text),
                    cfg_scale_caption=float(cfg_scale_caption),
                    cfg_scale_speaker=float(cfg_scale_speaker),
                    seed=seed,
                ),
                log_fn=lambda message: print(message, flush=True),
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
            output_subdir = (
                Path("no_voice")
                if voice_name == NO_VOICE_OPTION
                else Path(voice_name).with_suffix("")
            )
            output_dir = self.output_dir / output_subdir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{timestamp}.{output_format}"
            torchaudio.save(
                str(output_path),
                result.audio.cpu(),
                int(result.sample_rate),
                compression=self.audio_bitrate,
            )
            return str(output_path)
