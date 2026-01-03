"""
Chat template core types and rendering.

Chat templates define how multi-turn conversations are formatted
for different LLM models. Each model family has its own format.
"""


struct ChatMessage(Copyable, Movable):
    """A single message in a conversation."""

    var role: String
    """Message role: 'system', 'user', 'assistant', 'tool'."""

    var content: String
    """Message content."""

    var name: String
    """Optional name for the message sender."""

    fn __init__(out self, role: String, content: String, name: String = ""):
        """Create a chat message."""
        self.role = role
        self.content = content
        self.name = name

    fn __copyinit__(out self, existing: Self):
        """Copy constructor."""
        self.role = existing.role
        self.content = existing.content
        self.name = existing.name

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self.role = existing.role^
        self.content = existing.content^
        self.name = existing.name^

    @staticmethod
    fn system(content: String) -> ChatMessage:
        """Create a system message."""
        return ChatMessage("system", content)

    @staticmethod
    fn user(content: String) -> ChatMessage:
        """Create a user message."""
        return ChatMessage("user", content)

    @staticmethod
    fn assistant(content: String) -> ChatMessage:
        """Create an assistant message."""
        return ChatMessage("assistant", content)

    @staticmethod
    fn tool(content: String, name: String = "") -> ChatMessage:
        """Create a tool response message."""
        return ChatMessage("tool", content, name)


struct ChatTemplate(Copyable, Movable):
    """
    Chat template for formatting conversations.

    Templates define the format for each message type and how
    messages are joined together.
    """

    var bos_token: String
    """Beginning of sequence token."""

    var eos_token: String
    """End of sequence token."""

    var system_prefix: String
    """Prefix before system message."""

    var system_suffix: String
    """Suffix after system message."""

    var user_prefix: String
    """Prefix before user message."""

    var user_suffix: String
    """Suffix after user message."""

    var assistant_prefix: String
    """Prefix before assistant message."""

    var assistant_suffix: String
    """Suffix after assistant message."""

    var sep: String
    """Separator between messages."""

    var add_generation_prompt: Bool
    """Whether to add prompt for generation at end."""

    var generation_prompt: String
    """The prompt to add for generation (usually assistant prefix)."""

    fn __init__(out self):
        """Create a default (empty) template."""
        self.bos_token = ""
        self.eos_token = ""
        self.system_prefix = ""
        self.system_suffix = ""
        self.user_prefix = ""
        self.user_suffix = ""
        self.assistant_prefix = ""
        self.assistant_suffix = ""
        self.sep = ""
        self.add_generation_prompt = True
        self.generation_prompt = ""

    fn __copyinit__(out self, existing: Self):
        """Copy constructor."""
        self.bos_token = existing.bos_token
        self.eos_token = existing.eos_token
        self.system_prefix = existing.system_prefix
        self.system_suffix = existing.system_suffix
        self.user_prefix = existing.user_prefix
        self.user_suffix = existing.user_suffix
        self.assistant_prefix = existing.assistant_prefix
        self.assistant_suffix = existing.assistant_suffix
        self.sep = existing.sep
        self.add_generation_prompt = existing.add_generation_prompt
        self.generation_prompt = existing.generation_prompt

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self.bos_token = existing.bos_token^
        self.eos_token = existing.eos_token^
        self.system_prefix = existing.system_prefix^
        self.system_suffix = existing.system_suffix^
        self.user_prefix = existing.user_prefix^
        self.user_suffix = existing.user_suffix^
        self.assistant_prefix = existing.assistant_prefix^
        self.assistant_suffix = existing.assistant_suffix^
        self.sep = existing.sep^
        self.add_generation_prompt = existing.add_generation_prompt
        self.generation_prompt = existing.generation_prompt^

    fn apply(self, messages: List[ChatMessage]) -> String:
        """
        Apply template to a list of messages.

        Args:
            messages: List of chat messages.

        Returns:
            Formatted string ready for tokenization.
        """
        var result = self.bos_token

        for i in range(len(messages)):
            var msg = messages[i].copy()

            if i > 0:
                result += self.sep

            if msg.role == "system":
                result += self.system_prefix + msg.content + self.system_suffix
            elif msg.role == "user":
                result += self.user_prefix + msg.content + self.user_suffix
            elif msg.role == "assistant":
                result += self.assistant_prefix + msg.content + self.assistant_suffix
            elif msg.role == "tool":
                # Tool messages typically use user format
                result += self.user_prefix + msg.content + self.user_suffix

        # Add generation prompt if last message is not from assistant
        if self.add_generation_prompt and len(messages) > 0:
            var last = messages[len(messages) - 1].copy()
            if last.role != "assistant":
                result += self.generation_prompt

        return result

    fn format_message(self, msg: ChatMessage) -> String:
        """Format a single message."""
        if msg.role == "system":
            return self.system_prefix + msg.content + self.system_suffix
        elif msg.role == "user":
            return self.user_prefix + msg.content + self.user_suffix
        elif msg.role == "assistant":
            return self.assistant_prefix + msg.content + self.assistant_suffix
        else:
            return self.user_prefix + msg.content + self.user_suffix


fn apply_chat_template(
    messages: List[ChatMessage],
    template: ChatTemplate,
    add_generation_prompt: Bool = True
) -> String:
    """
    Apply a chat template to a list of messages.

    This is a convenience function that creates a copy of the template
    with the specified generation prompt setting.

    Args:
        messages: List of chat messages.
        template: The chat template to use.
        add_generation_prompt: Whether to add assistant prompt at end.

    Returns:
        Formatted string ready for tokenization.

    Example:
        var messages = List[ChatMessage]()
        messages.append(ChatMessage.system("You are a helpful assistant."))
        messages.append(ChatMessage.user("Hello!"))

        var formatted = apply_chat_template(messages, llama3_template())
    """
    var t = template.copy()
    t.add_generation_prompt = add_generation_prompt
    return t.apply(messages)


struct TemplateBuilder(Movable):
    """Builder pattern for creating custom chat templates."""

    var _template: ChatTemplate

    fn __init__(out self):
        """Create a new template builder."""
        self._template = ChatTemplate()

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor."""
        self._template = existing._template^

    fn bos(var self, token: String) -> Self:
        """Set BOS token."""
        self._template.bos_token = token
        return self^

    fn eos(var self, token: String) -> Self:
        """Set EOS token."""
        self._template.eos_token = token
        return self^

    fn system(var self, prefix: String, suffix: String) -> Self:
        """Set system message format."""
        self._template.system_prefix = prefix
        self._template.system_suffix = suffix
        return self^

    fn user(var self, prefix: String, suffix: String) -> Self:
        """Set user message format."""
        self._template.user_prefix = prefix
        self._template.user_suffix = suffix
        return self^

    fn assistant(var self, prefix: String, suffix: String) -> Self:
        """Set assistant message format."""
        self._template.assistant_prefix = prefix
        self._template.assistant_suffix = suffix
        return self^

    fn sep(var self, separator: String) -> Self:
        """Set message separator."""
        self._template.sep = separator
        return self^

    fn generation_prompt(var self, prompt: String) -> Self:
        """Set generation prompt."""
        self._template.generation_prompt = prompt
        self._template.add_generation_prompt = True
        return self^

    fn build(self) -> ChatTemplate:
        """Build the template."""
        return self._template.copy()
