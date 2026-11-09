"""Errors whose messages are written by PromptSieve itself.

The CLI hides messages from other exceptions because backend errors can embed
prompts, commands or credentials. UsageError messages must use application-owned
text, validated enum values or numeric positions, never caller-supplied text.
"""


class UsageError(ValueError):
    """Invalid configuration or arguments; the message is safe to display."""


def exception_summary(context: str, error: Exception) -> str:
    """Use a fixed caller-owned context and exception class, never its contents."""
    return f"{context} ({type(error).__name__}); details withheld"
