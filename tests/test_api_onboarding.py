from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.research_digest import call_generation_with_retry, extract_generation_text
from app.transcription import (
    load_api_key,
    load_deepseek_api_key,
    save_api_key,
    save_deepseek_api_key,
)


def test_deepseek_generation_uses_current_chat_endpoint_and_normalizes_output() -> None:
    response = SimpleNamespace(
        status_code=200,
        reason="OK",
        json=lambda: {
            "choices": [
                {
                    "message": {"content": "中文译文"},
                    "finish_reason": "stop",
                }
            ]
        },
    )
    with patch("app.research_digest.requests.post", return_value=response) as request:
        result = call_generation_with_retry(
            provider="deepseek",
            api_key="ds-test-key",
            model="qwen-plus",
            messages=[{"role": "user", "content": "Translate this"}],
            max_attempts=1,
        )

    assert extract_generation_text(result) == "中文译文"
    url = request.call_args.args[0]
    options = request.call_args.kwargs
    assert url == "https://api.deepseek.com/chat/completions"
    assert options["headers"]["Authorization"] == "Bearer ds-test-key"
    assert options["json"]["model"] == "deepseek-v4-flash"
    assert options["json"]["thinking"] == {"type": "disabled"}
    assert options["json"]["stream"] is False


def test_dashscope_and_deepseek_keys_share_one_local_env_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    dotenv_path = tmp_path / ".env"

    save_api_key(dotenv_path, "aliyun-key")
    save_deepseek_api_key(dotenv_path, "deepseek-key")

    assert load_api_key(dotenv_path) == "aliyun-key"
    assert load_deepseek_api_key(dotenv_path) == "deepseek-key"
    assert dotenv_path.read_text(encoding="utf-8").splitlines() == [
        "DASHSCOPE_API_KEY=aliyun-key",
        "DEEPSEEK_API_KEY=deepseek-key",
    ]


def test_main_contains_first_run_gate_and_both_provider_choices() -> None:
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    assert "self.root.after(260, self.maybe_show_first_run_api_setup)" in source
    assert "def show_api_setup_dialog(" in source
    assert '"\u963f\u91cc\u4e91\u767e\u70bc": "dashscope"' in source
    assert '"DeepSeek": "deepseek"' in source
    assert "self.settings_canvas" in source
    assert 'reason="生成节目中文详情需要先选择翻译服务并填写自己的 API Key。"' in source
    assert "def ui_rounded_entry(" in source
    assert "card = RoundedPanel(" in source
    assert "api_form = RoundedPanel(" in source
    assert "self.after(40, lambda: update_height(remaining_passes - 1))" in source
