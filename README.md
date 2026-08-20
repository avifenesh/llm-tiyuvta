# llm-tiyuvta

[LLM](https://llm.datasette.io/) plugin for [tiyuvta](https://inference.tiyuvta.ai/) — prepaid
open-model inference with one price per model, no subscription.

Every model on the endpoint speaks the identical OpenAI-compatible API: streaming, tool
calling, structured output, and vision where the model has eyes. The plugin reads the live
model catalog, so new models appear without a plugin update.

## Installation

Install this plugin in the same environment as LLM:

```bash
llm install llm-tiyuvta
```

## Configuration

Get an API key from [inference.tiyuvta.ai](https://inference.tiyuvta.ai/?ref=llm-plugin), then:

```bash
llm keys set tiyuvta
# Paste key here
```

You can also set it as the `TIYUVTA_KEY` environment variable.

## Usage

Run a prompt (short aliases work — `qwen3.8-27b` for `tiyuvta/qwen/qwen3.8-27b`):

```bash
llm -m qwen3.8-27b "Three reasons the sky looks blue"
```

Chat interactively:

```bash
llm chat -m qwen3.8-27b
```

Vision — attach an image:

```bash
llm -m qwen3.8-27b "describe this" -a photo.jpg
```

Tools and schemas work the way they do for any OpenAI-compatible LLM model:

```bash
llm -m qwen3.8-27b --schema 'name, bullet_points: three key points' "summarize: ..."
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
