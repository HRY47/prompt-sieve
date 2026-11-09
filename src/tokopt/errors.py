"""Errors whose messages are written by tokopt itself.

The CLI hides messages from other exceptions because backend errors can embed
prompts, commands or credentials. UsageError messages never contain input text,
so the CLI prints them to tell the user what to fix.
"""


class UsageError(ValueError):
    """Invalid configuration or arguments; the message is safe to display."""
