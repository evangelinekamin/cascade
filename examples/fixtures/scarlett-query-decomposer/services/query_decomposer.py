"""Query decomposition for retailer-friendly search terms.

Verbose queries like "long black silk duster kimono robe outerwear" return
garbage from most retailer search APIs. This module breaks them down into a
colour-led PRIMARY query that keeps the full descriptor stack (colour +
silhouette/length + garment anchor), plus a few shorter recall fallbacks --
queries that retailers actually index well.

Applied as a preprocessing step in LiveSearchService to ALL incoming queries
-- both from users typing in the Shop page and from Scarlett's chat tool calls.
No LLM involved; pure rule-based anchor extraction.
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Anchor terms -- the nouns retailers actually index on.
# NOT a complete fashion taxonomy -- just terms that work as search keywords.
# Multi-word anchors are checked first (longest match wins).
# ---------------------------------------------------------------------------

GARMENT_ANCHORS: set[str] = {
    # Tops
    "tee", "t-shirt", "tank", "camisole", "cami", "blouse", "shirt",
    "crop top", "bodysuit", "tube top", "halter", "henley", "polo",
    "sweater", "pullover", "cardigan", "hoodie", "sweatshirt", "turtleneck",
    # Bottoms
    "jeans", "pants", "trousers", "shorts", "skirt", "leggings",
    "joggers", "culottes", "palazzo", "chinos",
    # Dresses / jumpsuits
    "dress", "gown", "romper", "jumpsuit", "playsuit",
    # Outerwear
    "jacket", "coat", "blazer", "vest", "puffer", "parka", "trench",
    "duster", "kimono", "cape", "poncho", "shacket", "windbreaker",
    "robe",
    # Shoes
    "sneakers", "boots", "heels", "sandals", "flats", "loafers",
    "mules", "slides", "platforms", "pumps", "oxfords", "clogs",
    "stilettos", "wedges", "espadrilles",
    # Bags
    "bag", "tote", "clutch", "backpack", "crossbody", "satchel",
    "purse", "wallet", "fanny pack",
    # Accessories
    "hat", "scarf", "belt", "sunglasses", "necklace", "earrings",
    "bracelet", "ring", "watch", "choker",
    # Swimwear / intimates
    "bikini", "swimsuit", "lingerie", "bralette", "corset", "bustier",
}

# Multi-word anchors sorted by length (longest first) for greedy matching
_MULTI_WORD_ANCHORS = sorted(
    [a for a in GARMENT_ANCHORS if " " in a],
    key=len,
    reverse=True,
)
_SINGLE_WORD_ANCHORS = {a for a in GARMENT_ANCHORS if " " not in a}

# ---------------------------------------------------------------------------
# Modifiers worth keeping when paired with an anchor
# ---------------------------------------------------------------------------

LENGTH_MODIFIERS: set[str] = {
    "mini", "midi", "maxi", "cropped", "longline", "oversized",
    "fitted", "relaxed", "wide-leg", "skinny", "straight", "flared",
    "high-waisted", "low-rise", "floor-length", "knee-length",
    "ankle", "tall", "petite", "plus-size",
}

STYLE_MODIFIERS: set[str] = {
    # Fabric / texture
    "silk", "satin", "velvet", "leather", "faux-leather", "denim",
    "knit", "mesh", "lace", "sequin", "ribbed", "corduroy", "tweed",
    "cashmere", "linen", "chiffon", "tulle",
    # Construction / detail
    "pleated", "ruched", "wrap", "button-down", "open-front", "zip-up",
    "distressed", "raw-hem", "paperbag", "peplum", "smocked", "tiered",
    # Cut / silhouette — the words that actually define a piece's shape
    "cargo", "parachute", "baggy", "bootcut", "flare", "tapered", "slip",
    "tube", "strapless", "sleeveless", "off-shoulder", "one-shoulder",
    "puff-sleeve", "bodycon", "a-line", "bandeau", "asymmetric",
    # Pattern / print
    "floral", "striped", "plaid", "checkered", "gingham", "polka-dot",
    "leopard", "snakeskin", "paisley", "tie-dye", "camo", "houndstooth",
    "tartan", "animal-print",
}

# Colour is the single most discriminating attribute for a stylist's search,
# yet a bare anchor+modifier decomposition drops it. Kept as its own set so the
# primary query can lead with colour. Single words only — "navy blue" is caught
# by "navy", "hot pink" by "pink". Also imported by the live-search scorer.
COLOR_MODIFIERS: set[str] = {
    "black", "white", "ivory", "cream", "beige", "tan", "camel", "brown",
    "chocolate", "grey", "gray", "charcoal", "silver", "navy", "blue",
    "cobalt", "teal", "turquoise", "green", "olive", "sage", "mint", "emerald",
    "red", "crimson", "burgundy", "maroon", "wine", "pink", "blush", "rose",
    "fuchsia", "magenta", "coral", "peach", "purple", "lavender", "lilac",
    "plum", "violet", "yellow", "mustard", "gold", "orange", "rust", "khaki",
    "metallic", "neon", "nude", "taupe",
}

# Words to strip -- never useful as search terms
_STOP_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "with", "in", "by", "for", "of",
    "at", "to", "from", "on", "is", "it", "my", "this", "that",
    "some", "like", "very", "really", "super", "pretty", "cute",
    "nice", "good", "great", "perfect", "beautiful", "amazing",
    "looking", "something", "style", "styled", "piece", "item",
    "clothing", "garment", "outfit", "wear", "wearing", "wardrobe",
    "going", "out",
    # Gender prefixes handled separately
    "women", "womens", "women's", "men", "mens", "men's",
    "ladies", "girls", "guys",
}


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens, preserving hyphenated words."""
    return re.findall(r"[a-z]+(?:-[a-z]+)*", text.lower())


def _find_anchors(tokens: list[str], raw_lower: str) -> list[str]:


    """Find garment anchor terms in the token list.

    Checks multi-word anchors against the raw string first,
    then single-word anchors against individual tokens.
    Returns anchors in the order they appear.
    """
    found: list[str] = []
    consumed_positions: set[int] = set()

    # Multi-word anchors (check against raw lowered string)
    for anchor in _MULTI_WORD_ANCHORS:
        if anchor in raw_lower:
            found.append(anchor)
            # Mark consumed token positions
            anchor_tokens = anchor.split()
            for i in range(len(tokens) - len(anchor_tokens) + 1):
                if tokens[i:i + len(anchor_tokens)] == anchor_tokens:
                    for j in range(i, i + len(anchor_tokens)):
                        consumed_positions.add(j)

    # Single-word anchors
    for i, token in enumerate(tokens):
        if i in consumed_positions:
            continue
        if token in _SINGLE_WORD_ANCHORS:
            found.append(token)
            consumed_positions.add(i)

    return found


def _find_modifiers(tokens: list[str], modifier_set: set[str], raw_lower: str) -> list[str]:


    """Find modifier terms in the token list.

    Handles both hyphenated ("wide-leg") and space-separated ("wide leg")
    forms by checking the raw string for multi-word modifiers.
    """
    found: list[str] = []
    consumed: set[int] = set()

    # Multi-word modifiers: check raw string for "high-waisted" and "high waisted"
    multi_mods = sorted(
        [m for m in modifier_set if "-" in m],
        key=len,
        reverse=True,
    )
    for mod in multi_mods:
        # Check both hyphenated and space-separated forms
        space_form = mod.replace("-", " ")
        if mod in raw_lower or space_form in raw_lower:
            found.append(mod)
            # Mark consumed token positions
            mod_tokens = space_form.split()
            for i in range(len(tokens) - len(mod_tokens) + 1):
                if tokens[i:i + len(mod_tokens)] == mod_tokens:
                    for j in range(i, i + len(mod_tokens)):
                        consumed.add(j)

    # Single-word modifiers
    for i, token in enumerate(tokens):
        if i in consumed:
            continue
        if token in modifier_set:
            found.append(token)

    return found


def _cap_words(query: str, max_words: int = 3) -> Optional[str]:
    """Cap a query to max_words. Returns None if empty after capping."""
    words = query.split()
    if not words:
        return None
    return " ".join(words[:max_words])


def decompose_query(raw_query: str) -> list[str]:


    """Turn a verbose query into a few short retailer-friendly queries.

    The FIRST result is the high-recall PRIMARY term: it keeps the full
    descriptor stack a stylist search depends on -- colour first, then every
    recognised style/pattern/silhouette and length modifier, then the garment
    anchor -- so "black pleated midi skirt" stays whole instead of collapsing to
    "midi skirt". The rest are shorter recall fallbacks.

    Strategy:
    1. If the query is already <=3 words, return it unchanged as the only result.
    2. Tokenize; drop stop words and gender prefixes (_STOP_WORDS).
    3. Find the garment anchor(s) (_find_anchors), the colour (COLOR_MODIFIERS),
       and the style/length modifiers (_find_modifiers). Multi-word forms like
       "off shoulder" / "wide leg" normalise to their hyphenated canonical
       ("off-shoulder", "wide-leg").
    4. PRIMARY (result[0]): colour + style modifiers + length modifiers + the
       anchor, joined in that order and capped at 5 words
       (_cap_words(..., max_words=5)).
    5. RECALL fallbacks: shorter "<modifier> <anchor>" pairings (and a two-anchor
       pairing when 2+ anchors are found), each capped at 3 words. Flatten and
       dedup, preserving order.
    6. If no anchor is found, fall back to the last 2-3 non-stop words.

    Examples:
        "women's black pleated midi skirt" -> ["black pleated midi skirt", "midi skirt", "pleated skirt"]
        "red floral satin midi dress"      -> ["red floral satin midi dress", "midi dress", "floral dress"]
        "high waisted wide leg jeans"      -> ["high-waisted wide-leg jeans", "high-waisted jeans"]
        "cute going out top"               -> ["top"]
        "black midi skirt"                 -> ["black midi skirt"]   # <=3 words, intact
    """
    raise NotImplementedError


def decompose_queries(queries: list[str]) -> list[str]:
    """Decompose a list of queries, flatten and deduplicate.

    Each input query is decomposed into 1-3 short queries.
    Results are deduped across all inputs.
    """
    raise NotImplementedError
