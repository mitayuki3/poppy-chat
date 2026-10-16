# Poppy Chat

LM Studioで動作するローカルLLMと会話する、日本語のローカルWebアプリです。UIはGradio、接続にはLM StudioのネイティブREST APIを使います。

## 起動

1. LM Studioでモデルをロードし、Local Serverを起動します（標準ポートは`1234`）。
2. アプリを起動します。

```powershell
uv run python main.py
```

ブラウザで開いた画面で「モデル一覧を更新」を押し、モデルを選んで会話を始めます。接続先は`host:port`形式で変更できます。モデル一覧にはLM Studioでロード済みのLLMだけが表示されます。Reasoning欄では選択したモデルが対応する設定を選べます。「自動」はLM Studioの既定値を使います。

## テスト

```powershell
uv run pytest
```

テストはLM Studioへの通信をモックするため、LM Studioを起動せずに実行できます。
