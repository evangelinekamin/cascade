"""In-memory quiz session manager for the onboarding style quiz.

Maintains per-user quiz state (profile vector, uncertainty vector,
shown images, reactions) keyed by a random session token. Sessions
are ephemeral — lost on server restart, which is acceptable for alpha.

The quiz runs unauthenticated, so sessions are identified by token only.
On account creation, the client sends the accumulated reactions to be
persisted via a separate endpoint.
"""

import logging
import math
import secrets
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Algorithm tuning constants
LEARNING_RATE = 0.15
LOVE_WEIGHT = 2.0
LIKE_WEIGHT = 0.5
PASS_WEIGHT = -1.0
NEGATIVE_DAMPING = 0.5
UNCERTAINTY_BASE_REDUCTION = 0.2
LOVE_UNCERTAINTY_BONUS = 1.5
PASS_UNCERTAINTY_BONUS = 1.2
AXIS_RELEVANCE_THRESHOLD = 0.2
POLARIZER_EXTREMENESS_THRESHOLD = 0.7
BRAND_DIVERSITY_WINDOW = 5
BRAND_GLOBAL_MAX = 3  # Hard cap: no brand shown more than 3 times total
GENDER_REPEAT_LIMIT = 3
AESTHETIC_DIVERSITY_WINDOW = 3
WARMUP_COUNT = 4
REFINEMENT_START = 17
MIN_SWIPES_FOR_EARLY_EXIT = 12
MAX_SWIPES = 40
RESOLVED_THRESHOLD = 0.6

# The 8 style axes
STYLE_AXES = [
    "structure", "edge", "color_boldness", "volume",
    "polish", "romance", "streetwear", "whimsy",
]

# Deterministic mapping from 8 axes to 6 categories
# Each category is a weighted combination of axes. Positive weight = higher axis
# value increases category score. Negative weight = higher axis value decreases it.
CATEGORY_WEIGHTS = {
    "refined_edge": {
        "structure": 0.25, "edge": 0.40, "polish": 0.20,
        "volume": -0.10, "romance": -0.15, "whimsy": -0.10,
    },
    "quiet_luxe": {
        "polish": 0.40, "structure": 0.25, "edge": -0.20,
        "volume": -0.15, "color_boldness": -0.15, "whimsy": -0.10,
    },
    "free_spirit": {
        "romance": 0.40, "whimsy": 0.20, "structure": -0.25,
        "polish": -0.15, "edge": -0.15, "streetwear": -0.10,
    },
    "main_character": {
        "color_boldness": 0.30, "volume": 0.25, "whimsy": 0.30,
        "edge": 0.10, "polish": -0.10,
    },
    "off_duty": {
        "streetwear": 0.35, "volume": 0.15, "structure": -0.15,
        "polish": -0.20, "whimsy": 0.10, "edge": 0.05,
    },
    "after_dark": {
        "edge": 0.25, "polish": 0.30, "color_boldness": 0.10,
        "streetwear": -0.20, "whimsy": -0.15, "romance": 0.10,
    },
}

STYLE_CATEGORIES = list(CATEGORY_WEIGHTS.keys())


@dataclass
class QuizReaction:
    """A single reaction during the quiz."""
    image_id: str
    reaction: str  # "love" | "like" | "pass"
    image_scores: dict[str, float]
    source_brand: str
    gender_presentation: str
    primary_aesthetics: list[str]
    swipe_number: int


@dataclass
class QuizSession:
    """State for an in-progress quiz session."""
    token: str
    created_at: float = field(default_factory=time.monotonic)

    # Running profile vector (8 axes, each 0.0-1.0, starts at 0.5)
    profile: dict[str, float] = field(
        default_factory=lambda: {axis: 0.5 for axis in STYLE_AXES}
    )

    # Uncertainty vector (8 axes, each 0.0-1.0, starts at 1.0)
    uncertainty: dict[str, float] = field(
        default_factory=lambda: {axis: 1.0 for axis in STYLE_AXES}
    )

    # Gender preference from user selection ("feminine", "masculine", "androgynous", or None)
    gender_preference: str | None = None

    # Tracking
    reactions: list[QuizReaction] = field(default_factory=list)
    shown_image_ids: set[str] = field(default_factory=set)
    shown_brands: list[str] = field(default_factory=list)
    swipe_count: int = 0
    _finished: bool = False

    @property
    def phase(self) -> str:
        """Current quiz phase based on swipe count."""
        if self.swipe_count < WARMUP_COUNT:
            return "warmup"
        elif self.swipe_count < MIN_SWIPES_FOR_EARLY_EXIT:
            return "exploration"
        else:
            return "refinement"

    @property
    def is_complete(self) -> bool:
        return self._finished or self.swipe_count >= MAX_SWIPES

    @property
    def can_finish_early(self) -> bool:
        """Whether the user has provided enough signal to finish early."""
        if self.swipe_count < MIN_SWIPES_FOR_EARLY_EXIT:
            return False
        # All uncertainty values must be below threshold
        if not all(u < RESOLVED_THRESHOLD for u in self.uncertainty.values()):
            return False
        # Need at least 2 non-pass reactions for meaningful signal
        non_pass_count = sum(
            1 for r in self.reactions if r.reaction != "pass"
        )
        return non_pass_count >= 2


def compute_style_dna(profile: dict[str, float]) -> dict[str, float]:
    """Compute 6-category Style DNA from the 8-axis profile vector.

    Uses a deterministic weighted sum for each category, clamped to 0.0-1.0.
    """
    raise NotImplementedError


def update_profile(session: QuizSession, image_scores: dict[str, float], reaction: str):
    """Update the profile and uncertainty vectors after a swipe.

    Implements the exponential moving average profile update and
    uncertainty reduction from the quiz serving algorithm spec.
    """
    raise NotImplementedError


def score_candidate(
    session: QuizSession,
    candidate: dict,
) -> float:
    """Score a candidate image for selection in Phase 2/3.

    Higher score = better candidate for reducing uncertainty.
    """
    score = 0.0
    for axis in STYLE_AXES:
        axis_extremeness = abs(candidate["scores"].get(axis, 0.5) - 0.5) * 2
        score += axis_extremeness * session.uncertainty.get(axis, 0.0)

    # Role bonuses
    quiz_role = candidate.get("quiz_role", "candy")
    if quiz_role == "polarizer":
        score *= 1.3
    elif quiz_role == "wildcard":
        score *= 1.1

    # --- Brand diversity ---
    brand = candidate.get("source_brand", "")

    # Hard cap: if this brand has appeared BRAND_GLOBAL_MAX times, eliminate it
    brand_total = session.shown_brands.count(brand)
    if brand_total >= BRAND_GLOBAL_MAX:
        return 0.0

    # Escalating penalty based on total appearances (0.5^count)
    if brand_total > 0:
        score *= 0.5 ** brand_total

    # Recent-window penalty on top of global penalty
    recent_brands = session.shown_brands[-BRAND_DIVERSITY_WINDOW:]
    if brand in recent_brands:
        score *= 0.2

    # --- Gender preference ---
    gender = candidate.get("gender_presentation", "")
    if session.gender_preference:
        if gender != session.gender_preference and gender != "androgynous":
            # Strong penalty for mismatched gender, but don't eliminate
            score *= 0.1

    # --- Gender repeat penalty ---
    if len(session.reactions) >= GENDER_REPEAT_LIMIT:
        recent_genders = [r.gender_presentation for r in session.reactions[-GENDER_REPEAT_LIMIT:]]
        if all(g == gender for g in recent_genders):
            score *= 0.5

    # --- Aesthetic diversity (check last N images, not just last 1) ---
    candidate_aesthetics = candidate.get("primary_aesthetics", [])
    if candidate_aesthetics and len(session.reactions) > 0:
        recent_aesthetics = []
        for r in session.reactions[-AESTHETIC_DIVERSITY_WINDOW:]:
            if r.primary_aesthetics:
                recent_aesthetics.append(r.primary_aesthetics[0])
        if candidate_aesthetics[0] in recent_aesthetics:
            # Penalize more if the aesthetic appeared multiple times recently
            repeat_count = recent_aesthetics.count(candidate_aesthetics[0])
            score *= 0.6 ** repeat_count

    # --- Appeal score multiplier ---
    appeal = candidate.get("quiz_appeal_score")
    if appeal is not None:
        score *= 0.7 + (appeal / 10.0) * 0.45

    return score


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Compute cosine similarity between two axis-keyed vectors."""
    raise NotImplementedError


class QuizSessionManager:
    """Manages in-memory quiz sessions keyed by token.

    Sessions are ephemeral and not persisted. A quiz takes 3-5 minutes,
    so losing state on server restart is acceptable for alpha.
    """

    # Sessions older than 1 hour are garbage collected
    SESSION_TTL_SECONDS = 3600

    def __init__(self):
        self._sessions: dict[str, QuizSession] = {}

    def create_session(self) -> QuizSession:
        """Create a new quiz session with a random token."""
        self._gc()
        token = secrets.token_urlsafe(32)
        session = QuizSession(token=token)
        self._sessions[token] = session
        return session

    def get_session(self, token: str) -> QuizSession | None:
        """Retrieve a session by token, or None if expired/missing."""
        session = self._sessions.get(token)
        if session is None:
            return None
        if time.monotonic() - session.created_at > self.SESSION_TTL_SECONDS:
            del self._sessions[token]
            return None
        return session

    def remove_session(self, token: str):
        """Remove a completed or abandoned session."""
        self._sessions.pop(token, None)

    def _gc(self):
        """Remove expired sessions. Called on create to prevent unbounded growth."""
        now = time.monotonic()
        expired = [
            token for token, session in self._sessions.items()
            if now - session.created_at > self.SESSION_TTL_SECONDS
        ]
        for token in expired:
            del self._sessions[token]


# Module-level singleton
_quiz_session_manager: QuizSessionManager | None = None


def get_quiz_session_manager() -> QuizSessionManager:
    """Get the singleton quiz session manager."""
    global _quiz_session_manager
    if _quiz_session_manager is None:
        _quiz_session_manager = QuizSessionManager()
    return _quiz_session_manager
