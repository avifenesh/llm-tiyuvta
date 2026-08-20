"""LLM plugin for tiyuvta — prepaid inference at https://api.tiyuvta.ai/v1.

Every model on the endpoint speaks the identical OpenAI-compatible API
(streaming, tools, structured output, vision where the model has eyes), so
this plugin is a thin registration layer: it reads the live /v1/models
catalog and registers each model with the capabilities the endpoint itself
declares. A new model on the endpoint appears here with no plugin update.
"""

import inspect
import json
import os
import time
from collections import Counter
from pathlib import Path

import click
import httpx
import llm
from llm.default_plugins.openai_models import AsyncChat, Chat

API_BASE = "https://api.tiyuvta.ai/v1"
KEY_NAME = "tiyuvta"
KEY_ENV_VAR = "TIYUVTA_KEY"


def get_tiyuvta_models(skip_cache=False):
    payload = fetch_cached_json(
        url=f"{API_BASE}/models",
        path=llm.user_dir() / "tiyuvta_models.json",
        cache_timeout=0 if skip_cache else 3600,
    )
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise DownloadError("Unexpected /v1/models payload shape")
    return models


def supports_images(model_definition):
    return "image" in (model_definition.get("input_modalities") or [])


def capability(model_definition, name):
    return bool((model_definition.get("capabilities") or {}).get(name))


class _TiyuvtaMixin:
    needs_key = KEY_NAME
    key_env_var = KEY_ENV_VAR

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the endpoint takes images but refuses OpenAI's "file" content part
        # (probed live: 400 'unsupported content part type "file"'), so the
        # PDF type that vision=True auto-advertises must not be promised
        self.attachment_types.discard("application/pdf")

    def __str__(self):
        return "tiyuvta: {}".format(self.model_id)


class TiyuvtaChat(_TiyuvtaMixin, Chat):
    pass


class TiyuvtaAsyncChat(_TiyuvtaMixin, AsyncChat):
    pass


# Chat's kwarg surface grows across llm versions (reasoning arrived after
# tools did); passing only what this installation accepts keeps one plugin
# working across the declared floor instead of TypeError-ing every command.
_CHAT_PARAMS = frozenset(inspect.signature(Chat.__init__).parameters)


def _chat_kwargs(model_definition):
    model_id = model_definition["id"]
    kwargs = dict(
        model_id="tiyuvta/{}".format(model_id),
        model_name=model_id,
        api_base=API_BASE,
        vision=supports_images(model_definition),
        supports_tools=capability(model_definition, "tools"),
        supports_schema=capability(model_definition, "structured_output"),
        can_stream=capability(model_definition, "streaming"),
        reasoning=capability(model_definition, "reasoning"),
    )
    return {name: value for name, value in kwargs.items() if name in _CHAT_PARAMS}


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
    # `llm -m qwen3.8-27b` beats `llm -m tiyuvta/qwen/qwen3.8-27b` — but only
    # while the short name is unambiguous within the catalog
    basenames = Counter(m["id"].split("/")[-1] for m in models)
    for model_definition in models:
        kwargs = _chat_kwargs(model_definition)
        alias = model_definition["id"].split("/")[-1]
        register(
            TiyuvtaChat(**kwargs),
            TiyuvtaAsyncChat(**kwargs),
            aliases=(alias,) if basenames[alias] == 1 else None,
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
        try:
            all_models = get_tiyuvta_models(skip_cache=True)
        except DownloadError as ex:
            raise click.ClickException(str(ex))
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
        try:
            before = {m["id"] for m in get_tiyuvta_models()}
            after = {m["id"] for m in get_tiyuvta_models(skip_cache=True)}
        except DownloadError as ex:
            raise click.ClickException(str(ex))
        for added in sorted(after - before):
            click.echo("Added: {}".format(added), err=True)
        for removed in sorted(before - after):
            click.echo("Removed: {}".format(removed), err=True)
        click.echo("{} models".format(len(after)), err=True)


class DownloadError(Exception):
    pass


def _read_cache(path):
    """A cache that fails to parse is a cache miss, never a crash — a corrupt
    file here used to traceback EVERY llm command until manually deleted."""
    try:
        with open(path, "r") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return None


def fetch_cached_json(url, path, cache_timeout):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file() and time.time() - path.stat().st_mtime < cache_timeout:
        cached = _read_cache(path)
        if cached is not None:
            return cached

    try:
        response = httpx.get(url, follow_redirects=True, timeout=15.0)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, json.JSONDecodeError):
        # stale cache beats no models when the endpoint is briefly unreachable
        # or answering non-JSON (proxy interstitials return 200 text/html)
        cached = _read_cache(path)
        if cached is not None:
            return cached
        raise DownloadError("Failed to download data and no cache is available.")

    # atomic write: a process killed mid-dump must not leave a truncated file
    # with a fresh mtime (that state used to brick the CLI)
    tmp = path.with_suffix(".tmp{}".format(os.getpid()))
    with open(tmp, "w") as file:
        json.dump(payload, file)
    os.replace(tmp, path)
    return payload
