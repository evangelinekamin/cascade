"""Solve summary attributes tokens/cost per provider (escalation regression).

An escalating /solve runs on two providers; the summary used to lump ALL
tokens under the base provider and label the cost "OpenRouter cost", so an
escalation's claude tokens (millions, on the subscription) were reported as
deepseek/OpenRouter. _solve_token_lines now breaks it out per provider.
"""

from cascade.commands import _solve_token_lines
from cascade.swarm.solve import SolveResult
from cascade.swarm.outcome import RunOutcome


def _result(**kw):
    base = dict(
        task="t", provider="openrouter", passed=True, iterations=2,
        attempts=(), worktree_path="/w", outcome=RunOutcome.SUCCEEDED,
    )
    base.update(kw)
    return SolveResult(**base)


def test_escalation_breaks_tokens_out_per_provider():
    r = _result(
        input_tokens=8_947_072, output_tokens=30_369, cost=0.012580,
        tokens_by_provider=(
            ("openrouter", 512_421, 13_282),
            ("claude", 8_434_651, 17_087),
        ),
        cost_by_provider=(("openrouter", 0.012580),),  # claude reported no cost
    )
    out = "\n".join(_solve_token_lines(r))
    assert "Tokens by provider:" in out
    assert "openrouter: 512,421 in / 13,282 out · 0.012580 credits" in out
    # Claude's millions are attributed to claude, NOT lumped/mislabeled.
    assert "claude: 8,434,651 in / 17,087 out · (no metered cost" in out
    assert "Total: 8,947,072 in / 30,369 out" in out
    # The misleading blanket "OpenRouter cost" line is gone.
    assert "OpenRouter cost" not in out


def test_single_provider_labels_cost_by_provider_not_openrouter():
    r = _result(
        provider="openai", input_tokens=1000, output_tokens=200, cost=0.05,
        tokens_by_provider=(("openai", 1000, 200),),
        cost_by_provider=(("openai", 0.05),),
    )
    out = "\n".join(_solve_token_lines(r))
    assert "Tokens: 1,000 in / 200 out" in out
    assert "openai cost: 0.050000 credits" in out
    assert "OpenRouter" not in out


def test_missing_breakdown_falls_back_gracefully():
    # pipeline/fanout results (or older results) without the per-provider fields.
    r = _result(input_tokens=500, output_tokens=100, cost=0.0,
                provider="openrouter")
    out = "\n".join(_solve_token_lines(r))
    assert "Tokens: 500 in / 100 out" in out  # flat line, no crash
