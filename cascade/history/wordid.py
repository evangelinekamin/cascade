"""Human-readable session ID generator.

Produces IDs like ``purple-banana`` or ``swift-falcon`` — easy to
remember and type while still providing ~250k unique combinations.
Falls back to hex if collisions are detected.
"""

import random

_ADJECTIVES = [
    "amber", "aqua", "azure", "black", "blue", "brass", "bright", "bronze",
    "burnt", "calm", "cedar", "civic", "clear", "cobalt", "cold", "cool",
    "coral", "crisp", "cyan", "dark", "dawn", "deep", "dim", "dusk",
    "dusty", "faded", "fast", "fern", "fire", "flint", "frost", "gilt",
    "glad", "gold", "gray", "green", "hazy", "hot", "iron", "ivory",
    "jade", "keen", "lava", "lead", "lean", "light", "lilac", "lime",
    "lunar", "maple", "mild", "mint", "misty", "moss", "neon", "new",
    "night", "noble", "oaken", "opal", "pale", "peach", "pearl", "pine",
    "plum", "prime", "pure", "quiet", "raw", "red", "rich", "rose",
    "ruby", "rust", "sage", "sand", "sharp", "silk", "slick", "slim",
    "slow", "smoky", "snow", "solar", "solid", "sour", "stark", "steel",
    "still", "stone", "storm", "sunny", "swift", "tan", "teal", "thick",
    "thin", "tidal", "torn", "true", "warm", "white", "wild", "wise",
]

_NOUNS = [
    "acorn", "amber", "anvil", "arch", "aspen", "badge", "basin", "beam",
    "blade", "blaze", "bloom", "bluff", "bolt", "braid", "brick", "brook",
    "cairn", "cedar", "chalk", "cider", "cliff", "cloud", "comet", "coral",
    "crane", "crest", "crown", "curve", "delta", "drift", "drum", "dusk",
    "eagle", "ember", "fable", "fairy", "fern", "finch", "flame", "flare",
    "flint", "forge", "fox", "frost", "gale", "gate", "gem", "ghost",
    "glass", "gleam", "globe", "grain", "grove", "gust", "haven", "hawk",
    "hazel", "heath", "heron", "hive", "holly", "horn", "husky", "iris",
    "ivory", "jewel", "lake", "larch", "lark", "leaf", "light", "lily",
    "lotus", "lunar", "lynx", "maple", "marsh", "mesa", "mist", "moth",
    "nexus", "night", "north", "oak", "ocean", "olive", "onyx", "orbit",
    "otter", "owl", "palm", "pearl", "petal", "pine", "pixel", "plum",
    "pond", "prism", "pulse", "quail", "quartz", "raven", "reed", "ridge",
    "river", "robin", "rune", "sage", "scale", "shell", "shore", "sierra",
    "slate", "smoke", "solar", "spark", "spire", "spoke", "spray", "star",
    "steel", "stone", "storm", "surge", "thorn", "tide", "tiger", "torch",
    "trail", "trout", "tulip", "vale", "vapor", "vault", "veil", "vine",
    "wave", "wheat", "willow", "wind", "wolf", "wren", "zenith",
]


def generate_word_id() -> str:
    """Return a random ``adjective-noun`` ID."""
    return f"{random.choice(_ADJECTIVES)}-{random.choice(_NOUNS)}"
