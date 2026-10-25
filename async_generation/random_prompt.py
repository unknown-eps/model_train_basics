"""Generate small, reproducible prompts without network access."""

import random


def generate_random_prompts(seed: int, num_prompts: int) -> list[tuple[int, str]]:
    """Return (ID, prompt) pairs with consecutive IDs starting at zero."""
    if num_prompts < 0:
        raise ValueError("num_prompts must be nonnegative")

    rng = random.Random(seed)
    actions = ["Explain", "Describe", "Write a short story about", "Compare"]
    topics = ["robots", "oceans", "space travel", "forests", "computers", "music"]
    styles = ["for a beginner", "with examples", "in simple words", "with humor"]
    prompts = []

    for prompt_id in range(num_prompts):
        action = rng.choice(actions)
        topic = rng.choice(topics)
        style = rng.choice(styles)
        prompt = f"{action} {topic} {style}."
        prompts.append((prompt_id, prompt))

    return prompts
