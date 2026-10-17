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

## テスト

```powershell
uv run pytest
```

テストはLM Studioへの通信をモックするため、LM Studioを起動せずに実行できます。
