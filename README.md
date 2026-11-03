# mini-llm-agent

A small Transformer language model built from scratch with PyTorch, trained on my own machine, and plugged into a local agent that can read files and inspect Git repositories.

This is a learning project. The model is tiny (around 1 to 10 million parameters), so it will not compete with anything you can download. The point is to understand every piece of a language model by writing it myself, instead of calling an API or running someone else's weights.

## Why I'm building this

I'm a second-year computer science student and I wanted to know what actually happens inside an LLM. Reading about attention is one thing, getting a loss curve to go down because of code you wrote is another. I'm also using the project to get better at Git and GitHub: branches, pull requests, tests and CI are part of the work, not an afterthought.

## What it will be

```
User
  |
  v
Local agent
  |
  +--> MiniLLM (my Transformer)
  |
  +--> Tools: files, Git (read-only first), terminal (with confirmation)
```

The model predicts the next token. The agent wraps it and decides which tool to call. Anything that could change or delete data will ask for confirmation, and the agent is restricted to a folder I choose.

## Status

| Part | State |
|---|---|
| Project setup (venv, PyTorch CPU, pytest) | Done |
| Character-level tokenizer | Done, on `feature/tokenizer` |
| Dataset (input/target windows, train/val split) | Not started |
| First simple model (bigram) | Not started |
| Attention and Transformer blocks | Not started |
| Training loop and checkpoints | Not started |
| Text generation (greedy, temperature, top-k) | Not started |
| Evaluation and experiments | Not started |
| Code specialization (MiniCodeLM) | Not started |
| Local agent and tools | Not started |
| Interface | Not started |
| CI with GitHub Actions | Not started |

## Roadmap

1. Tokenizer, dataset and a bigram baseline, to check the training loop before the Transformer exists.
2. Embeddings, positional information, causal self-attention, multi-head attention, feed-forward layers, residual connections and layer norm.
3. Training on a small text dataset, then on Python code, and comparing configurations in `experiments/`.
4. Sampling strategies and a few simple evaluation metrics (validation loss, perplexity).
5. A local agent with read-only file and Git tools, then controlled commands behind a confirmation step.
6. A small interface, tests, and CI.

## Hardware and scale

The project targets a regular desktop: a Ryzen 5 5600X, 16 GB of RAM, and CPU training. A 1 to 10 million parameter model with a context of 128 to 256 characters trains in a reasonable time on that. I'd rather have a small model that actually trains than a big one that doesn't fit.

## Limitations

- A model this small does not know facts and does not reason. At best it learns the shape of text and of simple code.
- A raw language model only continues text. Answering questions would need extra training on question/answer pairs, and I plan to try that on a narrow topic (Git and basic Python).
- The agent will not trust the model to pick tools from free-form text. The agent code keeps control of what can run.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/ElictFishy/mini-llm-agent.git
cd mini-llm-agent
python -m venv .venv
```

Activate the environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Install PyTorch (CPU build), then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Usage

For now the only usable piece is the tokenizer:

```python
from model.tokenizer import CharTokenizer

tok = CharTokenizer.from_text("hello world")
ids = tok.encode("hello")
print(ids)              # list of token ids
print(tok.decode(ids))  # "hello"
```

Characters that are not in the vocabulary are mapped to `<unk>`. Run the tests with:

```bash
python -m pytest
```

## Project structure

```
mini-llm-agent/
  model/
    tokenizer/      character-level tokenizer
  tests/            pytest tests
  pyproject.toml    pytest configuration
  requirements.txt
```

The folders for the dataset, architecture, training, generation, evaluation, agent and app will be added as each part gets built.

## Git workflow

Work happens on short-lived `feature/*` branches and goes into `main` through pull requests. Commit messages follow the Conventional Commits style (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`). Model weights, datasets and `.env` files are never committed.
