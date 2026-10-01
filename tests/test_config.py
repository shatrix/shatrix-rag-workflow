"""Configuration resolution and the embedding capability map."""

from __future__ import annotations

import pytest

from app.config import (
    CHARS_PER_TOKEN,
    ConfigError,
    PROJECT_ROOT,
    config_warnings,
    get_settings,
    redact,
    reload_settings,
)


class TestPathResolution:
    def test_relative_paths_resolve_against_the_repo_root(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("RAG_DATA_DIR", "some/relative/dir")
        settings = reload_settings()
        assert settings.data_dir.is_absolute()
        assert str(settings.data_dir).startswith(str(PROJECT_ROOT))

    def test_absolute_paths_are_respected(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("RAG_DOCS_DIR", str(tmp_path / "elsewhere"))
        settings = reload_settings()
        assert settings.docs_dir == tmp_path / "elsewhere"

    def test_no_path_points_outside_the_project_by_default(self) -> None:
        settings = get_settings()
        for path in (settings.data_dir, settings.docs_dir, settings.chroma_dir):
            assert path.is_absolute()
            assert path.exists()

    def test_ensure_dirs_is_idempotent(self) -> None:
        settings = get_settings()
        settings.ensure_dirs()
        settings.ensure_dirs()
        assert settings.docs_dir.is_dir()
        assert settings.chroma_dir.is_dir()


class TestEnvironmentPrecedence:
    def test_real_env_var_overrides_dotenv(self, monkeypatch) -> None:
        # The autouse fixture sets this, proving env beats .env.
        assert get_settings().openrouter_api_key.startswith("sk-or-v1-test")

    def test_placeholder_key_is_rejected(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "your-key-here")
        settings = reload_settings()
        with pytest.raises(ConfigError, match="placeholder"):
            settings.require_api_key()

    def test_missing_key_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
            reload_settings().require_api_key()

    def test_valid_key_passes(self) -> None:
        assert get_settings().require_api_key()


class TestValidation:
    def test_overlap_must_be_smaller_than_chunk(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("RAG_CHUNK_CHARS", "1000")
        monkeypatch.setenv("RAG_CHUNK_OVERLAP_CHARS", "1000")
        with pytest.raises(ConfigError, match="must be smaller"):
            reload_settings()

    def test_non_numeric_value_is_reported_clearly(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("RAG_TOP_K", "lots")
        with pytest.raises(ConfigError, match="RAG_TOP_K"):
            reload_settings()

    def test_bad_boolean_is_reported_clearly(self, monkeypatch) -> None:
        monkeypatch.setenv("RAG_SHOW_SOURCES", "maybe")
        with pytest.raises(ConfigError, match="boolean"):
            reload_settings()

    def test_negative_value_is_rejected(self, monkeypatch) -> None:
        monkeypatch.setenv("RAG_TOP_K", "0")
        with pytest.raises(ConfigError, match="at least"):
            reload_settings()


class TestEmbeddingCapabilities:
    def test_known_model_reports_dims(self, monkeypatch) -> None:
        monkeypatch.setenv(
            "RAG_EMBED_MODEL", "nvidia/nemotron-3-embed-1b:free"
        )
        settings = reload_settings()
        assert settings.embed_dims == 2048
        assert settings.embed_max_input_tokens == 32768

    def test_tiny_context_model_forces_a_clamp(self, monkeypatch) -> None:
        # liquid's embedder only accepts 512 tokens per input, so a 1800-char
        # chunk must be clamped to roughly 2048 chars or it gets truncated.
        monkeypatch.setenv(
            "RAG_EMBED_MODEL", "liquid/lfm-2.5-embedding-350m:free"
        )
        monkeypatch.setenv("RAG_CHUNK_CHARS", "1800")
        settings = reload_settings()
        # 1800 chars is under the 2048-char ceiling, so nothing is clamped.
        assert settings.max_chunk_chars == 1800
        assert settings.chunk_truncated is False

        monkeypatch.setenv("RAG_CHUNK_CHARS", "9000")
        settings = reload_settings()
        assert settings.chunk_truncated is True
        assert settings.max_chunk_chars == 512 * CHARS_PER_TOKEN

    def test_unknown_model_is_conservative(self, monkeypatch) -> None:
        monkeypatch.setenv("RAG_EMBED_MODEL", "some/new-model:v9")
        settings = reload_settings()
        assert settings.embed_dims == 0
        assert settings.embed_max_input_tokens == 512
        # The fallback ceiling is 512 * 4 = 2048 chars, so the default 1800
        # fits, but anything larger is clamped rather than silently truncated
        # by the provider.
        assert settings.max_chunk_chars == min(1800, 512 * CHARS_PER_TOKEN)
        assert settings.chunk_truncated is False
        assert any(
            "not in the known-models table" in w for w in config_warnings()
        )


class TestWarnings:
    def test_no_warnings_for_the_default_configuration(self) -> None:
        assert config_warnings() == []

    def test_excessive_overlap_is_flagged(self, monkeypatch) -> None:
        monkeypatch.setenv("RAG_CHUNK_CHARS", "600")
        monkeypatch.setenv("RAG_CHUNK_OVERLAP_CHARS", "400")
        reload_settings()
        warnings = " ".join(config_warnings())
        assert "overlap" in warnings.lower()

    def test_public_bind_address_is_flagged(self, monkeypatch) -> None:
        monkeypatch.setenv("APP_BIND_ADDRESS", "0.0.0.0")
        reload_settings()
        warnings = " ".join(config_warnings())
        assert "no authentication" in warnings or "APP_BIND_ADDRESS" in warnings

    def test_missing_key_is_flagged(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        reload_settings()
        assert any("OPENROUTER_API_KEY" in w for w in config_warnings())


class TestRedaction:
    def test_key_is_never_shown_in_full(self) -> None:
        secret = "sk-or-v1-abcdefghijklmnopqrstuvwxyz0123456789"
        shown = redact(secret)
        assert secret not in shown
        assert shown.startswith("sk-o")
        assert str(len(secret)) in shown

    @pytest.mark.parametrize("empty", ["", None])
    def test_empty_key(self, empty) -> None:
        assert redact(empty) == "<empty>"

    def test_short_key_is_fully_masked(self) -> None:
        assert redact("abcd") == "****"