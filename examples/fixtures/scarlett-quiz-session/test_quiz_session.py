"""Tests for the quiz session manager and style profile algorithm."""

import pytest

from services.quiz_session import (
    MAX_SWIPES,
    STYLE_AXES,
    STYLE_CATEGORIES,
    QuizSession,
    QuizSessionManager,
    compute_style_dna,
    cosine_similarity,
    score_candidate,
    update_profile,
)


class TestComputeStyleDNA:
    """Tests for the deterministic 8-axis -> 6-category mapping."""

    def test_neutral_profile_produces_moderate_dna(self):
        """A neutral profile (all 0.5) should produce ~0.5 for all categories."""
        profile = {axis: 0.5 for axis in STYLE_AXES}
        dna = compute_style_dna(profile)
        assert set(dna.keys()) == set(STYLE_CATEGORIES)
        for cat, val in dna.items():
            assert 0.4 <= val <= 0.6, f"{cat} should be near 0.5, got {val}"

    def test_high_edge_maps_to_refined_edge(self):
        """High edge + structure should produce high refined_edge."""
        profile = {axis: 0.5 for axis in STYLE_AXES}
        profile["edge"] = 0.9
        profile["structure"] = 0.8
        profile["polish"] = 0.75
        dna = compute_style_dna(profile)
        # Refined edge should be the highest category
        top_cat = max(dna, key=dna.get)
        assert top_cat == "refined_edge"
        assert dna["refined_edge"] > 0.7

    def test_high_romance_maps_to_free_spirit(self):
        """High romance + low structure should produce high free_spirit."""
        profile = {axis: 0.5 for axis in STYLE_AXES}
        profile["romance"] = 0.9
        profile["whimsy"] = 0.7
        profile["structure"] = 0.2
        profile["polish"] = 0.3
        dna = compute_style_dna(profile)
        top_cat = max(dna, key=dna.get)
        assert top_cat == "free_spirit"
        assert dna["free_spirit"] > 0.6

    def test_high_color_volume_whimsy_maps_to_main_character(self):
        """High color_boldness + volume + whimsy -> main_character."""
        profile = {axis: 0.5 for axis in STYLE_AXES}
        profile["color_boldness"] = 0.9
        profile["volume"] = 0.8
        profile["whimsy"] = 0.85
        dna = compute_style_dna(profile)
        top_cat = max(dna, key=dna.get)
        assert top_cat == "main_character"

    def test_high_streetwear_maps_to_off_duty(self):
        """High streetwear + low polish -> off_duty."""
        profile = {axis: 0.5 for axis in STYLE_AXES}
        profile["streetwear"] = 0.9
        profile["polish"] = 0.2
        profile["structure"] = 0.3
        dna = compute_style_dna(profile)
        top_cat = max(dna, key=dna.get)
        assert top_cat == "off_duty"

    def test_dna_values_clamped_0_to_1(self):
        """DNA values should always be between 0 and 1."""
        # Extreme profile
        profile = {axis: 1.0 for axis in STYLE_AXES}
        dna = compute_style_dna(profile)
        for cat, val in dna.items():
            assert 0.0 <= val <= 1.0, f"{cat} out of range: {val}"

        profile = {axis: 0.0 for axis in STYLE_AXES}
        dna = compute_style_dna(profile)
        for cat, val in dna.items():
            assert 0.0 <= val <= 1.0, f"{cat} out of range: {val}"


class TestUpdateProfile:
    """Tests for the profile update formula."""

    def test_love_pulls_profile_toward_image(self):
        """Loving an image should pull the profile toward its scores."""
        session = QuizSession(token="test")
        image_scores = {axis: 0.5 for axis in STYLE_AXES}
        image_scores["edge"] = 0.9  # High edge image

        update_profile(session, image_scores, "love")
        assert session.profile["edge"] > 0.5, "Edge should increase after loving high-edge image"

    def test_pass_pushes_profile_toward_neutral(self):
        """Passing on an image should push the profile toward neutral."""
        session = QuizSession(token="test")
        image_scores = {axis: 0.5 for axis in STYLE_AXES}
        image_scores["edge"] = 0.9

        update_profile(session, image_scores, "pass")
        # Passing on high-edge should push edge toward neutral (lower, since image is 0.9)
        # The pass formula pushes toward 0.5, weighted by how extreme the image is
        assert session.profile["edge"] <= 0.5, "Edge should not increase after passing on high-edge"

    def test_neutral_image_axes_dont_update(self):
        """Axes where the image scores ~0.5 shouldn't change the profile."""
        session = QuizSession(token="test")
        image_scores = {axis: 0.5 for axis in STYLE_AXES}  # All neutral
        image_scores["edge"] = 0.55  # Nearly neutral

        update_profile(session, image_scores, "love")
        # Edge should barely move (axis_extremeness = 0.1, below threshold)
        assert abs(session.profile["edge"] - 0.5) < 0.01

    def test_love_stronger_than_like(self):
        """Love (+2.0) should move the profile more than like (+0.5)."""
        session_love = QuizSession(token="test1")
        session_like = QuizSession(token="test2")
        image_scores = {axis: 0.5 for axis in STYLE_AXES}
        image_scores["edge"] = 0.9

        update_profile(session_love, image_scores, "love")
        update_profile(session_like, image_scores, "like")

        love_delta = abs(session_love.profile["edge"] - 0.5)
        like_delta = abs(session_like.profile["edge"] - 0.5)
        assert love_delta > like_delta

    def test_profile_stays_clamped(self):
        """Profile values should stay between 0.0 and 1.0."""
        session = QuizSession(token="test")
        image_scores = {axis: 1.0 for axis in STYLE_AXES}

        # Love the same extreme image many times
        for _ in range(20):
            update_profile(session, image_scores, "love")

        for axis in STYLE_AXES:
            assert 0.0 <= session.profile[axis] <= 1.0

    def test_uncertainty_decreases_with_reactions(self):
        """Uncertainty should decrease after reacting to extreme images."""
        session = QuizSession(token="test")
        initial_uncertainty = session.uncertainty["edge"]

        image_scores = {axis: 0.5 for axis in STYLE_AXES}
        image_scores["edge"] = 0.9  # Extreme edge

        update_profile(session, image_scores, "love")
        assert session.uncertainty["edge"] < initial_uncertainty

    def test_love_reduces_uncertainty_most(self):
        """Love should reduce uncertainty more than like or pass."""
        sessions = {r: QuizSession(token=r) for r in ("love", "like", "pass")}
        image_scores = {axis: 0.5 for axis in STYLE_AXES}
        image_scores["edge"] = 0.9

        for reaction, session in sessions.items():
            update_profile(session, image_scores, reaction)

        # Love > Pass > Like for uncertainty reduction
        assert sessions["love"].uncertainty["edge"] < sessions["like"].uncertainty["edge"]
        assert sessions["pass"].uncertainty["edge"] < sessions["like"].uncertainty["edge"]


class TestScoreCandidate:
    """Tests for the candidate image scoring function."""

    def test_extreme_image_on_uncertain_axis_scores_high(self):
        """An image with extreme scores on uncertain axes should score well."""
        session = QuizSession(token="test")
        # All axes maximally uncertain (default)

        candidate = {
            "scores": {axis: 0.5 for axis in STYLE_AXES},
            "quiz_role": "polarizer",
            "source_brand": "brand_a",
            "gender_presentation": "feminine",
            "primary_aesthetics": ["goth"],
        }
        candidate["scores"]["edge"] = 0.95  # Very extreme

        score = score_candidate(session, candidate)
        assert score > 0

    def test_polarizer_bonus(self):
        """Polarizer images should get a score bonus."""
        session = QuizSession(token="test")
        base = {
            "scores": {"edge": 0.9, **{a: 0.5 for a in STYLE_AXES if a != "edge"}},
            "source_brand": "brand_a",
            "gender_presentation": "feminine",
            "primary_aesthetics": ["goth"],
        }

        candy_score = score_candidate(session, {**base, "quiz_role": "candy"})
        polarizer_score = score_candidate(session, {**base, "quiz_role": "polarizer"})
        assert polarizer_score > candy_score

    def test_recent_brand_penalty(self):
        """Images from recently shown brands should be penalized."""
        session = QuizSession(token="test")
        session.shown_brands = ["brand_a", "brand_b", "brand_a"]

        candidate = {
            "scores": {axis: 0.9 for axis in STYLE_AXES},
            "quiz_role": "polarizer",
            "source_brand": "brand_a",  # Recently shown
            "gender_presentation": "feminine",
            "primary_aesthetics": ["goth"],
        }

        penalized = score_candidate(session, candidate)

        candidate["source_brand"] = "brand_c"  # Not recently shown
        unpenalized = score_candidate(session, candidate)

        assert penalized < unpenalized


class TestCosineSimilarity:
    """Tests for cosine similarity between profile vectors."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        a = {axis: 0.7 for axis in STYLE_AXES}
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_zero_vector(self):
        """Zero vector should return 0.0 similarity."""
        a = {axis: 0.5 for axis in STYLE_AXES}
        b = {axis: 0.0 for axis in STYLE_AXES}
        assert cosine_similarity(a, b) == 0.0

    def test_similar_vectors_high_similarity(self):
        """Similar vectors should have high similarity."""
        a = {"structure": 0.8, "edge": 0.7, "polish": 0.9}
        b = {"structure": 0.75, "edge": 0.65, "polish": 0.85}
        sim = cosine_similarity(a, b)
        assert sim > 0.99


class TestQuizSessionManager:
    """Tests for the session lifecycle manager."""

    def test_create_and_retrieve(self):
        mgr = QuizSessionManager()
        session = mgr.create_session()
        assert mgr.get_session(session.token) is session

    def test_invalid_token_returns_none(self):
        mgr = QuizSessionManager()
        assert mgr.get_session("nonexistent") is None

    def test_remove_session(self):
        mgr = QuizSessionManager()
        session = mgr.create_session()
        mgr.remove_session(session.token)
        assert mgr.get_session(session.token) is None

    def test_session_phase_progression(self):
        session = QuizSession(token="test")
        assert session.phase == "warmup"

        session.swipe_count = 4
        assert session.phase == "exploration"

        session.swipe_count = 16
        assert session.phase == "refinement"

    def test_session_completion(self):
        session = QuizSession(token="test")
        assert not session.is_complete

        session.swipe_count = MAX_SWIPES
        assert session.is_complete

        session2 = QuizSession(token="test2")
        session2._finished = True
        assert session2.is_complete


class TestAlgorithmConvergence:
    """Test that the algorithm converges for simulated user personas."""

    def _simulate_persona(self, preference_fn):
        """Simulate a quiz with a given preference function.

        preference_fn takes image_scores and returns 'love', 'like', or 'pass'.
        """
        session = QuizSession(token="test")

        for i in range(20):
            # Generate a fake image with random-ish scores
            import random
            random.seed(i * 42)
            scores = {axis: random.random() for axis in STYLE_AXES}
            reaction = preference_fn(scores)
            update_profile(session, scores, reaction)
            session.swipe_count += 1

        return session.profile

    def test_minimalist_converges(self):
        """User who loves high-polish, high-structure, low-volume."""
        def pref(scores):
            if scores["polish"] > 0.7 and scores["volume"] < 0.4:
                return "love"
            elif scores["polish"] > 0.5:
                return "like"
            return "pass"

        profile = self._simulate_persona(pref)
        assert profile["polish"] > 0.55, f"Expected high polish, got {profile['polish']}"

    def test_goth_romantic_converges(self):
        """User who loves high-edge + high-romance."""
        def pref(scores):
            if scores["edge"] > 0.6 and scores["romance"] > 0.5:
                return "love"
            elif scores["edge"] > 0.5:
                return "like"
            return "pass"

        profile = self._simulate_persona(pref)
        assert profile["edge"] > 0.5, f"Expected high edge, got {profile['edge']}"
