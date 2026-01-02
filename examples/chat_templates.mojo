"""
Chat template usage examples for mojo-tokenizer.

This example demonstrates:
1. Creating chat messages
2. Applying different chat templates
3. Custom template creation
"""

from mojo_tokenizer.chat import (
    ChatMessage,
    ChatTemplate,
    apply_chat_template,
    TemplateBuilder,
)
from mojo_tokenizer.chat.formats import (
    chatml_template,
    llama2_template,
    llama3_template,
    mistral_template,
    alpaca_template,
)


fn main() raises:
    print("=== mojo-tokenizer Chat Templates ===\n")

    # Create a conversation
    var messages = List[ChatMessage]()
    messages.append(ChatMessage.system("You are a helpful AI assistant."))
    messages.append(ChatMessage.user("What is the capital of France?"))
    messages.append(ChatMessage.assistant("The capital of France is Paris."))
    messages.append(ChatMessage.user("What about Germany?"))

    # Apply different templates
    print("1. ChatML Format (GPT-4, Claude, Qwen):")
    print("=" * 50)
    var chatml = apply_chat_template(messages, chatml_template())
    print(chatml)
    print()

    print("2. Llama 3 Format:")
    print("=" * 50)
    var llama3 = apply_chat_template(messages, llama3_template())
    print(llama3)
    print()

    print("3. Mistral Format:")
    print("=" * 50)
    var mistral = apply_chat_template(messages, mistral_template())
    print(mistral)
    print()

    print("4. Alpaca Format:")
    print("=" * 50)
    var alpaca = apply_chat_template(messages, alpaca_template())
    print(alpaca)
    print()

    # Custom template using builder
    print("5. Custom Template (using TemplateBuilder):")
    print("=" * 50)
    var custom = TemplateBuilder()
        .bos("<|START|>")
        .eos("<|END|>")
        .system("System: ", "\n")
        .user("Human: ", "\n")
        .assistant("AI: ", "\n")
        .generation_prompt("AI: ")
        .build()

    var custom_formatted = apply_chat_template(messages, custom)
    print(custom_formatted)
    print()

    print("=== Done! ===")
