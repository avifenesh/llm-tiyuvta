"""LLM plugin for tiyuvta — prepaid inference at https://api.tiyuvta.ai/v1.

Every model on the endpoint speaks the identical OpenAI-compatible API
(streaming, tools, structured output, vision where the model has eyes), so
this plugin is a thin registration layer: it reads the live /v1/models
catalog and registers each model with the capabilities the endpoint itself
declares. A new model on the endpoint appears here with no plugin update.
"""

import json
import time
from pathlib import Path

import click
import httpx
import llm
from llm.default_plugins.openai_models import AsyncChat, Chat

API_BASE = "https://api.tiyuvta.ai/v1"
KEY_NAME = "tiyuvta"
KEY_ENV_VAR = "TIYUVTA_KEY"


def get_tiyuvta_models(skip_cache=False):
    return fetch_cached_json(
        url=f"{API_BASE}/models",
        path=llm.user_dir() / "tiyuvta_models.json",
        cache_timeout=0 if skip_cache else 3600,
    )["data"]


def supports_images(model_definition):
    return "image" in (model_definition.get("input_modalities") or [])


def capability(model_definition, name):
    return bool((model_definition.get("capabilities") or {}).get(name))


class TiyuvtaChat(Chat):
    needs_key = KEY_NAME
    key_env_var = KEY_ENV_VAR

    def __str__(self):
        return "tiyuvta: {}".format(self.model_id)


class TiyuvtaAsyncChat(AsyncChat):
    needs_key = KEY_NAME
    key_env_var = KEY_ENV_VAR

    def __str__(self):
        return "tiyuvta: {}".format(self.model_id)


@llm.hookimpl
def register_models(register):
    # Only register when a key is configured, so `llm models` stays clean for
    # people who installed this plugin but have not signed up yet.
    key = llm.get_key("", KEY_NAME, KEY_ENV_VAR)
    if not key:
        return
    try:
        models = get_tiyuvta_models()
    except DownloadError:
        return
    for model_definition in models:
        model_id = model_definition["id"]
        kwargs = dict(
            model_id="tiyuvta/{}".format(model_id),
            model_name=model_id,
            api_base=API_BASE,
            vision=supports_images(model_definition),
            supports_tools=capability(model_definition, "tools"),
            supports_schema=capability(model_definition, "structured_output"),
            can_stream=capability(model_definition, "streaming"),
        )
        # `llm -m qwen3.8-27b` beats `llm -m tiyuvta/qwen/qwen3.8-27b`
        alias = model_id.split("/")[-1]
        register(
            TiyuvtaChat(**kwargs),
            TiyuvtaAsyncChat(**kwargs),
            aliases=(alias,),
        )


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def tiyuvta():
        "Commands relating to the llm-tiyuvta plugin"

    @tiyuvta.command()
    @click.option("json_", "--json", is_flag=True, help="Output as JSON")
    def models(json_):
        "List models served by tiyuvta"
        all_models = get_tiyuvta_models(skip_cache=True)
        if json_:
            click.echo(json.dumps(all_models, indent=2))
            return
        for model in all_models:
            bits = [
                "- id: {}".format(model["id"]),
                "  context: {}".format(model.get("context_length", "?")),
                "  modalities: {}".format(",".join(model.get("input_modalities") or ["text"])),
            ]
            caps = model.get("capabilities") or {}
            enabled = [name for name, on in caps.items() if on]
            if enabled:
                bits.append("  capabilities: {}".format(",".join(sorted(enabled))))
            click.echo("\n".join(bits))

    @tiyuvta.command()
    def refresh():
        "Refresh the cached model catalog"
        before = {m["id"] for m in get_tiyuvta_models()}
        after = {m["id"] for m in get_tiyuvta_models(skip_cache=True)}
        for added in sorted(after - before):
            click.echo("Added: {}".format(added), err=True)
        for removed in sorted(before - after):
            click.echo("Removed: {}".format(removed), err=True)
        click.echo("{} models".format(len(after)), err=True)


class DownloadError(Exception):
    pass


def fetch_cached_json(url, path, cache_timeout):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        mod_time = path.stat().st_mtime
        if time.time() - mod_time < cache_timeout:
            with open(path, "r") as file:
                return json.load(file)

    try:
        response = httpx.get(url, follow_redirects=True, timeout=15.0)
        response.raise_for_status()
        with open(path, "w") as file:
            json.dump(response.json(), file)
        return response.json()
    except httpx.HTTPError:
        # stale cache beats no models when the endpoint is briefly unreachable
        if path.is_file():
            with open(path, "r") as file:
                return json.load(file)
        raise DownloadError("Failed to download data and no cache is available.")
