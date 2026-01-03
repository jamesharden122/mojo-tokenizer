"""
Pre-defined chat template formats.

Contains templates for common LLM chat formats:
- ChatML: GPT-4, Claude, Qwen
- Llama 2: Llama 2 Chat
- Llama 3: Llama 3 Instruct
- Mistral: Mistral Instruct
- Alpaca: Stanford Alpaca
- Vicuna: Vicuna
"""

from .template import ChatTemplate, TemplateBuilder


fn chatml_template() -> ChatTemplate:
    """
    ChatML format used by OpenAI GPT-4 and others.

    Format:
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        {assistant_message}<|im_end|>
    """
    var template = ChatTemplate()
    template.bos_token = ""
    template.eos_token = ""
    template.system_prefix = "<|im_start|>system\n"
    template.system_suffix = "<|im_end|>\n"
    template.user_prefix = "<|im_start|>user\n"
    template.user_suffix = "<|im_end|>\n"
    template.assistant_prefix = "<|im_start|>assistant\n"
    template.assistant_suffix = "<|im_end|>\n"
    template.sep = ""
    template.generation_prompt = "<|im_start|>assistant\n"
    template.add_generation_prompt = True
    return template^


fn llama2_template() -> ChatTemplate:
    """
    Llama 2 Chat format.

    Format:
        <s>[INST] <<SYS>>
        {system_message}
        <</SYS>>

        {user_message} [/INST] {assistant_message} </s><s>[INST] {user_message} [/INST]
    """
    var template = ChatTemplate()
    template.bos_token = "<s>"
    template.eos_token = "</s>"
    template.system_prefix = "[INST] <<SYS>>\n"
    template.system_suffix = "\n<</SYS>>\n\n"
    template.user_prefix = ""
    template.user_suffix = " [/INST] "
    template.assistant_prefix = ""
    template.assistant_suffix = " </s><s>[INST] "
    template.sep = ""
    template.generation_prompt = ""
    template.add_generation_prompt = False
    return template^


fn llama3_template() -> ChatTemplate:
    """
    Llama 3 Instruct format.

    Format:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

        {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        {assistant_message}<|eot_id|>
    """
    var template = ChatTemplate()
    template.bos_token = "<|begin_of_text|>"
    template.eos_token = "<|eot_id|>"
    template.system_prefix = "<|start_header_id|>system<|end_header_id|>\n\n"
    template.system_suffix = "<|eot_id|>"
    template.user_prefix = "<|start_header_id|>user<|end_header_id|>\n\n"
    template.user_suffix = "<|eot_id|>"
    template.assistant_prefix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    template.assistant_suffix = "<|eot_id|>"
    template.sep = ""
    template.generation_prompt = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    template.add_generation_prompt = True
    return template^


fn mistral_template() -> ChatTemplate:
    """
    Mistral Instruct format.

    Format:
        <s>[INST] {user_message} [/INST]{assistant_message}</s>[INST] {user_message} [/INST]
    """
    var template = ChatTemplate()
    template.bos_token = "<s>"
    template.eos_token = "</s>"
    template.system_prefix = ""  # Mistral includes system in first user message
    template.system_suffix = "\n\n"
    template.user_prefix = "[INST] "
    template.user_suffix = " [/INST]"
    template.assistant_prefix = ""
    template.assistant_suffix = "</s>"
    template.sep = ""
    template.generation_prompt = ""
    template.add_generation_prompt = False
    return template^


fn alpaca_template() -> ChatTemplate:
    """
    Stanford Alpaca format.

    Format:
        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        {response}
    """
    var template = ChatTemplate()
    template.bos_token = ""
    template.eos_token = ""
    template.system_prefix = ""
    template.system_suffix = "\n\n"
    template.user_prefix = "### Instruction:\n"
    template.user_suffix = "\n\n"
    template.assistant_prefix = "### Response:\n"
    template.assistant_suffix = "\n\n"
    template.sep = ""
    template.generation_prompt = "### Response:\n"
    template.add_generation_prompt = True
    return template^


fn vicuna_template() -> ChatTemplate:
    """
    Vicuna format.

    Format:
        USER: {user_message}
        ASSISTANT: {assistant_message}
    """
    var template = ChatTemplate()
    template.bos_token = ""
    template.eos_token = "</s>"
    template.system_prefix = ""
    template.system_suffix = "\n\n"
    template.user_prefix = "USER: "
    template.user_suffix = "\n"
    template.assistant_prefix = "ASSISTANT: "
    template.assistant_suffix = "</s>\n"
    template.sep = ""
    template.generation_prompt = "ASSISTANT: "
    template.add_generation_prompt = True
    return template^


fn phi3_template() -> ChatTemplate:
    """
    Phi-3 format.

    Format:
        <|system|>
        {system_message}<|end|>
        <|user|>
        {user_message}<|end|>
        <|assistant|>
        {assistant_message}<|end|>
    """
    var template = ChatTemplate()
    template.bos_token = ""
    template.eos_token = ""
    template.system_prefix = "<|system|>\n"
    template.system_suffix = "<|end|>\n"
    template.user_prefix = "<|user|>\n"
    template.user_suffix = "<|end|>\n"
    template.assistant_prefix = "<|assistant|>\n"
    template.assistant_suffix = "<|end|>\n"
    template.sep = ""
    template.generation_prompt = "<|assistant|>\n"
    template.add_generation_prompt = True
    return template^


fn gemma_template() -> ChatTemplate:
    """
    Google Gemma format.

    Format:
        <start_of_turn>user
        {user_message}<end_of_turn>
        <start_of_turn>model
        {assistant_message}<end_of_turn>
    """
    var template = ChatTemplate()
    template.bos_token = "<bos>"
    template.eos_token = "<eos>"
    template.system_prefix = ""  # Gemma uses system in first user turn
    template.system_suffix = "\n\n"
    template.user_prefix = "<start_of_turn>user\n"
    template.user_suffix = "<end_of_turn>\n"
    template.assistant_prefix = "<start_of_turn>model\n"
    template.assistant_suffix = "<end_of_turn>\n"
    template.sep = ""
    template.generation_prompt = "<start_of_turn>model\n"
    template.add_generation_prompt = True
    return template^


fn qwen_template() -> ChatTemplate:
    """
    Qwen/Qwen2 format (uses ChatML).

    Same as ChatML format with <|im_start|> and <|im_end|> tags.
    """
    return chatml_template()


fn zephyr_template() -> ChatTemplate:
    """
    Zephyr format (HuggingFace).

    Format:
        <|system|>
        {system_message}</s>
        <|user|>
        {user_message}</s>
        <|assistant|>
        {assistant_message}</s>
    """
    var template = ChatTemplate()
    template.bos_token = ""
    template.eos_token = "</s>"
    template.system_prefix = "<|system|>\n"
    template.system_suffix = "</s>\n"
    template.user_prefix = "<|user|>\n"
    template.user_suffix = "</s>\n"
    template.assistant_prefix = "<|assistant|>\n"
    template.assistant_suffix = "</s>\n"
    template.sep = ""
    template.generation_prompt = "<|assistant|>\n"
    template.add_generation_prompt = True
    return template^
