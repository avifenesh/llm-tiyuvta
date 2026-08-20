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


def test_capability_mapping():
    qwen, gemma = CATALOG["data"]
    assert llm_tiyuvta.supports_images(qwen) is True
    assert llm_tiyuvta.supports_images(gemma) is False
    assert llm_tiyuvta.capability(qwen, "tools") is True
    assert llm_tiyuvta.capability(gemma, "reasoning") is False
