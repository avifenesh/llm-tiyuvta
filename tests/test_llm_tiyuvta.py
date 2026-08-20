import json

import httpx
import llm_tiyuvta


CATALOG = {
    "data": [
        {
            "id": "qwen/qwen3.8-27b",
            "context_length": 262144,
            "input_modalities": ["text", "image", "video"],
            "capabilities": {
                "streaming": True,
                "tools": True,
                "structured_output": True,
                "reasoning": True,
                "prompt_caching": True,
            },
        },
        {
            "id": "google/gemma-4-31b-it",
            "context_length": 262144,
            "input_modalities": ["text"],
            "capabilities": {"streaming": True, "tools": True, "structured_output": True},
        },
    ]
}


def test_fetch_cached_json_uses_cache_on_network_failure(tmp_path, httpx_mock):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(CATALOG))
    httpx_mock.add_exception(httpx.ConnectError("boom"))
    # cache_timeout=0 forces a fetch attempt; the stale file must win over the error
    data = llm_tiyuvta.fetch_cached_json("https://api.tiyuvta.ai/v1/models", path, cache_timeout=0)
    assert len(data["data"]) == 2


def test_corrupt_cache_is_a_miss_not_a_crash(tmp_path, httpx_mock):
    # a half-written cache used to traceback every llm command until deleted
    path = tmp_path / "models.json"
    path.write_text('{"data": [truncated')
    httpx_mock.add_response(json=CATALOG)
    data = llm_tiyuvta.fetch_cached_json("https://api.tiyuvta.ai/v1/models", path, cache_timeout=3600)
    assert len(data["data"]) == 2
    # and the refetch must repair the file atomically
    assert json.loads(path.read_text()) == CATALOG


def test_html_200_falls_back_to_stale_cache(tmp_path, httpx_mock):
    # a proxy interstitial answering 200 text/html is not a catalog
    path = tmp_path / "models.json"
    path.write_text(json.dumps(CATALOG))
    httpx_mock.add_response(text="<html>challenge</html>")
    data = llm_tiyuvta.fetch_cached_json("https://api.tiyuvta.ai/v1/models", path, cache_timeout=0)
    assert len(data["data"]) == 2


def test_capability_mapping():
    qwen, gemma = CATALOG["data"]
    assert llm_tiyuvta.supports_images(qwen) is True
    assert llm_tiyuvta.supports_images(gemma) is False
    assert llm_tiyuvta.capability(qwen, "tools") is True
    assert llm_tiyuvta.capability(gemma, "reasoning") is False


def test_models_construct_on_this_llm_version():
    # the bug this pins: kwargs Chat.__init__ does not accept on the installed
    # llm version must be filtered, not passed (a floor mismatch here used to
    # TypeError every llm command)
    qwen, gemma = CATALOG["data"]
    chat = llm_tiyuvta.TiyuvtaChat(**llm_tiyuvta._chat_kwargs(qwen))
    async_chat = llm_tiyuvta.TiyuvtaAsyncChat(**llm_tiyuvta._chat_kwargs(qwen))
    assert chat.model_id == "tiyuvta/qwen/qwen3.8-27b"
    assert async_chat.needs_key == "tiyuvta"
    # the endpoint refuses OpenAI "file" parts — PDF must not be advertised
    assert "application/pdf" not in chat.attachment_types
    assert "image/png" in chat.attachment_types
    text_only = llm_tiyuvta.TiyuvtaChat(**llm_tiyuvta._chat_kwargs(gemma))
    assert text_only.attachment_types == set()
