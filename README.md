# llm-tiyuvta

[![PyPI](https://img.shields.io/pypi/v/llm-tiyuvta.svg)](https://pypi.org/project/llm-tiyuvta/)
[![Changelog](https://img.shields.io/github/v/release/avifenesh/llm-tiyuvta?include_prereleases&label=changelog)](https://github.com/avifenesh/llm-tiyuvta/releases)
[![Tests](https://github.com/avifenesh/llm-tiyuvta/actions/workflows/test.yml/badge.svg)](https://github.com/avifenesh/llm-tiyuvta/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/avifenesh/llm-tiyuvta/blob/main/LICENSE)

[LLM](https://llm.datasette.io/) plugin for the [tiyuvta](https://tiyuvta.ai/) trial endpoint.

tiyuvta deploys open-source frontier models on hardware you can afford; the trial endpoint
serves one of them itself, DeepSeek-V4.1-Flash, so you can put a few dollars in and test your
own workload before we talk about your deployment. The endpoint is OpenAI-compatible: streaming,
tool calling, structured output, and vision where the model has eyes. The plugin reads the live
model catalog, so whatever the endpoint serves appears without a plugin update.

## Installation

Install this plugin in the same environment as LLM:

```bash
llm install llm-tiyuvta
```

## Configuration

Get a trial key at [tiyuvta.ai/services/try](https://tiyuvta.ai/services/try/), then:

```bash
llm keys set tiyuvta
# Paste key here
```

You can also set it as the `TIYUVTA_KEY` environment variable — note that a key stored
with `llm keys set` takes precedence over the environment variable.

Models appear in `llm models` once a key is configured.

## Usage

Run a prompt (short aliases work — `deepseek-v4.1-flash` for `tiyuvta/deepseek-ai/deepseek-v4.1-flash`):

```bash
llm -m deepseek-v4.1-flash "Three reasons the sky looks blue"
```

Chat interactively:

```bash
llm chat -m deepseek-v4.1-flash
```

Vision — attach an image:

```bash
llm -m deepseek-v4.1-flash "describe this" -a photo.jpg
```

Tools and schemas work the way they do for any OpenAI-compatible LLM model:

```bash
llm -m deepseek-v4.1-flash --schema 'name, bullet_points: three key points' "summarize: ..."
```

List the models the endpoint currently serves:

```bash
llm tiyuvta models
```

Refresh the cached catalog after a new model launches:

```bash
llm tiyuvta refresh
```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
cd llm-tiyuvta
python -m venv venv
source venv/bin/activate
```

Now install the dependencies and test dependencies:

```bash
llm install -e '.[test]'
```

To run the tests:

```bash
pytest
```
