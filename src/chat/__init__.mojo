"""
Chat templating for LLM conversation formats.

Supports common chat formats:
- ChatML (GPT-4, Claude)
- Llama / Llama 2 / Llama 3
- Mistral / Mixtral
- Alpaca
- Vicuna
"""

from .template import ChatTemplate, ChatMessage, apply_chat_template
from .formats import (
    chatml_template,
    llama2_template,
    llama3_template,
    mistral_template,
    alpaca_template,
    vicuna_template,
)
