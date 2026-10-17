# Poppy Chat

LM Studioで動作するローカルLLMと会話する、日本語のローカルWebアプリです。UIはGradio、接続にはLM StudioのネイティブREST APIを使います。

## 起動

1. LM Studioでモデルをロードし、Local Serverを起動します（標準ポートは`1234`）。
2. アプリを起動します。

```powershell
uv run python main.py
```

画面を開くとモデル一覧が自動更新されます。接続先は`host:port`形式で変更でき、必要なら「モデル一覧を更新」から再取得できます。モデル一覧にはLM Studioでロード済みのLLMだけが表示されます。Reasoningは対応モデルでは既定で`off`になり、`off`に対応しないモデルでは「自動」（LM Studioの既定値）になります。

「システムプロンプト」を開くと、入力した指示をチャットに適用できます。プリセット名と内容を入力して保存すると、次回以降も選択できます。プリセットはWindowsではユーザーの`%APPDATA%\PoppyChat\system_prompts.json`に保存されます。プリセットを切り替えても会話は続き、次の送信から新しい指示が適用されます。

## Irodori-TTS 読み上げ

アシスタントの回答は既定でIrodori-TTSにより読み上げられます。自動読み上げが有効な場合は画面起動時に選択中のチェックポイントを取得・ロードします。モデルは「Irodori-TTS 読み上げ設定」から `4.1 Small` または `4 Large` を選択できます。参照音声はvoicesディレクトリに置いて、同じ設定欄の一覧から選択します。生成ファイルは参照音声ごとのサブフォルダに、`YYYYMMDDTHHMMSSffffff` のファイル名で保存します。出力形式はOGG、WAV、MP3、FLACから選択でき、既定はOGGです。圧縮ビットレートの既定値は192000 bit/sです。

保存場所はプロジェクト直下の`.env`で変更できます。`.env.example`をコピーして必要なパスを設定してください。`.env`はGit管理対象外です。未設定の場合はアプリを起動したディレクトリ以下を使います。

| 環境変数 | 既定値 | 用途 |
| --- | --- | --- |
| `POPPY_CHAT_TTS_OUTPUT_DIR` | `outputs` | 生成音声（参照音声別のサブフォルダ） |
| `POPPY_CHAT_TTS_VOICES_DIR` | `voices` | 参照音声（WAV、MP3、FLAC、OGG） |
| `POPPY_CHAT_TTS_AUDIO_BITRATE` | `192000` | 音声エンコードの圧縮設定 |

`.env`の例:

```dotenv
POPPY_CHAT_TTS_OUTPUT_DIR="D:/PoppyChat/audio"
POPPY_CHAT_TTS_VOICES_DIR="D:/PoppyChat/voices"
POPPY_CHAT_TTS_AUDIO_BITRATE=192000
```

## テスト

```powershell
uv run pytest
```

テストはLM Studioへの通信をモックするため、LM Studioを起動せずに実行できます。
