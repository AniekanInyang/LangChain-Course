# LangChain Course

Practical LangChain examples using Azure OpenAI, including prompts, chains, memory, routing, retrieval, and agents.

## Files

- `utils.py`: Azure OpenAI + LangChain client helpers
- `prompt_template.py`: prompt templating and output parsing examples
- `chain.py`: LCEL chain and router examples
- `memory.py`: conversational memory example
- `qna.py`: retrieval-augmented QA example
- `agent.py`: tools + agent examples (Wikipedia, Python REPL, custom tools)
- `.env.example`: environment variable template

## Requirements

- Python 3.10+
- Azure OpenAI resource + deployment names

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

Create `.env` from `.env.example` and set values:

```env
AZURE_OPENAI_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_MODEL_NAME=...
AZURE_OPENAI_VERSION=...
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=...
```

## Run Examples

```bash
python prompt_template.py
python chain.py
python memory.py
python qna.py
python agent.py
```

## Notes

- `.env`, `.venv`, and `__pycache__/` are ignored in `.gitignore`.
- This repo is tutorial-oriented and intentionally keeps examples in standalone scripts.
