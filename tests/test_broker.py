"""Broker adapter unit tests — pure helpers only (no live broker; fully offline).

Importing trident.broker.alpaca loads alpaca-py but does NOT construct AlpacaBroker
(which needs credentials), so these run without network or keys.
"""

from __future__ import annotations

import pytest
from alpaca.trading.enums import OrderSide

from trident.broker.alpaca import _to_side
from trident.broker.base import BrokerError
from trident.models import Side


def test_to_side_maps_enum_buy_and_sell():
    assert _to_side(OrderSide.BUY) == Side.BUY
    assert _to_side(OrderSide.SELL) == Side.SELL


def test_to_side_handles_raw_string_forms():
    assert _to_side("buy") == Side.BUY
    assert _to_side("BUY") == Side.BUY
    assert _to_side("sell") == Side.SELL


def test_to_side_regression_the_str_enum_endswith_bug():
    # The original defect: str(OrderSide.BUY) == 'OrderSide.BUY', so `.endswith("buy")`
    # was ALWAYS False -> every buy fill was mislabeled a sell (would misbook the ledger).
    assert str(OrderSide.BUY).endswith("buy") is False  # why the old code failed
    assert _to_side(OrderSide.BUY) == Side.BUY  # the fix is correct


def test_to_side_unknown_raises_not_silently_wrong():
    with pytest.raises(BrokerError):
        _to_side("hold")  # fail safe (golden rule 8), never default-guess a side
