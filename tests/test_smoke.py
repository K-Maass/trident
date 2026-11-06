"""Repo-level smoke tests — grow real suites per phase (ledger tests are the crown jewels)."""

from trident.models import AgentName, BookSnapshot, Decision, Position


def test_decision_requires_rationale():
    import pytest

    with pytest.raises(ValueError):
        Decision(
            agent=AgentName.QUANT,
            asof="2026-01-05T15:00:00Z",
            ticker="aapl",
            target_weight=0.05,
            rationale={},
        )


def test_book_weight_math():
    book = BookSnapshot(
        agent=AgentName.QUANT,
        ts="2026-01-05T21:00:00Z",
        cash=7000,
        positions=[
            Position(agent=AgentName.QUANT, ticker="AAPL", qty=10, avg_cost=250, last_price=300)
        ],
    )
    assert book.equity == 10000
    assert abs(book.weight("AAPL") - 0.30) < 1e-9
