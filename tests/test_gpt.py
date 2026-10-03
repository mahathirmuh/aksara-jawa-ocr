"""Test wadah API GPT (ablasi koreksi pasca-OCR): kunci tidak bocor, cache mencegah bayar ulang."""

from types import SimpleNamespace

import pytest

from src.gpt import GPTClient, MissingAPIKey, check, lookup_key, mask

SECRET = "sk-test-RAHASIA-9876"


class FakeOpenAI:
    """Pengganti klien openai: mencatat panggilan, tanpa jaringan."""

    def __init__(self):
        self.calls = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="ꦲꦤ", model="gpt-server-name",
                               usage=SimpleNamespace(input_tokens=3, output_tokens=1))


def make_client(tmp_path, fake):
    return GPTClient(api_key=SECRET, cache_dir=tmp_path, client=fake)


def test_environment_variable_overrides_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=dari-berkas\n", encoding="utf-8")
    assert lookup_key(env, environ={"OPENAI_API_KEY": "dari-env"}) == ("dari-env", "environment variable")
    # Environment variable kosong tidak menghapus nilai di .env.
    assert lookup_key(env, environ={"OPENAI_API_KEY": ""}) == ("dari-berkas", ".env")


def test_empty_template_value_counts_as_missing(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=\n", encoding="utf-8")
    assert lookup_key(env, environ={}) == (None, None)
    with pytest.raises(MissingAPIKey, match="OPENAI_API_KEY"):
        GPTClient(cache_dir=tmp_path, client=FakeOpenAI(), env_path=env, environ={})


def test_mask_shows_only_last_four():
    shown = mask(SECRET)
    assert "9876" in shown and "RAHASIA" not in shown and "sk-test" not in shown
    assert mask("") == "(kosong)"
    assert "1234" not in mask("abcd1234")


def test_identical_call_is_served_from_cache_and_key_is_never_stored(tmp_path):
    fake = FakeOpenAI()
    client = make_client(tmp_path, fake)
    first = client.generate("gpt-x", "koreksi: ꦲꦤ", reasoning={"effort": "none"})
    second = client.generate("gpt-x", "koreksi: ꦲꦤ", reasoning={"effort": "none"})
    assert len(fake.calls) == 1
    assert fake.calls[0] == {"model": "gpt-x", "input": "koreksi: ꦲꦤ", "reasoning": {"effort": "none"}}
    assert not first.cached and second.cached
    assert second.text == "ꦲꦤ" and second.model == "gpt-server-name"
    stored = first.cache_path.read_text(encoding="utf-8")
    assert "RAHASIA" not in stored
    assert "RAHASIA" not in repr(client)


def test_different_params_or_models_are_cached_separately(tmp_path):
    fake = FakeOpenAI()
    client = make_client(tmp_path, fake)
    client.generate("gpt-x", "sama", reasoning={"effort": "none"})
    client.generate("gpt-x", "sama", reasoning={"effort": "low"})
    client.generate("gpt-y", "sama", reasoning={"effort": "none"})
    assert len(fake.calls) == 3


def test_use_cache_false_always_calls_and_writes_nothing(tmp_path):
    fake = FakeOpenAI()
    client = make_client(tmp_path, fake)
    for _ in range(2):
        assert client.generate("gpt-x", "ping", use_cache=False).cache_path is None
    assert len(fake.calls) == 2
    assert not any(tmp_path.rglob("*.json"))


def test_check_report_masks_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text(f"OPENAI_API_KEY={SECRET}\n", encoding="utf-8")
    report = "\n".join(check(env, environ={}))
    assert "RAHASIA" not in report and "9876" in report and "status: siap" in report
