"""SENTINEL relevance-filter tests — parsing, tickerless auto-NO, and the fail-safe policy.

Offline: a fake Anthropic client (no network). Covers the batch parse, the conservative NO on a
parse miss, the auto-NO for ticker-less items (no API call), and the "leave unjudged on API
error" policy that protects the cache from being poisoned by a transient failure.
"""

from __future__ import annotations

from trident.agents.sentiment import relevance as R


class _Text:
    def __init__(self, t):
        self.text = t
        self.type = "text"


class _Resp:
    def __init__(self, t):
        self.content = [_Text(t)]


class _Messages:
    def __init__(self, body, raise_exc=False):
        self._body = body
        self._raise = raise_exc

    async def create(self, **kw):
        if self._raise:
            raise RuntimeError("simulated API outage")
        return _Resp(self._body)


class _Client:
    def __init__(self, body="", raise_exc=False):
        self.messages = _Messages(body, raise_exc)


def _items(*specs):
    return [{"url_hash": h, "ticker": t, "title": ttl} for h, t, ttl in specs]


async def test_parses_yes_no_and_tickerless_auto_no():
    items = _items(
        ("h1", "AAPL", "Apple beats earnings"),
        ("h2", "MSFT", "Broad market wrap"),
        ("h3", None, "CNBC markets roundup"),  # tickerless → auto NO, no API
    )
    client = _Client("1. YES - about apple\n2. NO - index roundup\n")
    verdicts, rep = await R.filter_relevance(items, client=client)
    assert verdicts == {"h1": True, "h2": False, "h3": False}
    assert rep.api_calls == 1
    assert rep.tickerless_auto_no == 1
    assert rep.relevant == 1


async def test_parse_miss_is_conservative_no():
    items = _items(("h1", "AAPL", "x"), ("h2", "MSFT", "y"))
    client = _Client("1. YES - ok\n")  # item 2 line missing
    verdicts, rep = await R.filter_relevance(items, client=client)
    assert verdicts == {"h1": True, "h2": False}
    assert rep.parse_misses == 1


async def test_api_error_leaves_items_unjudged():
    items = _items(("h1", "AAPL", "x"), ("h2", "MSFT", "y"))
    client = _Client(raise_exc=True)
    verdicts, rep = await R.filter_relevance(items, client=client)
    assert verdicts == {}  # nothing written back → relevant stays NULL → retried next cycle
    assert rep.errored_items == 2
    assert rep.errors


def test_prompt_lists_ticker_and_title():
    prompt = R._build_user_prompt(_items(("h1", "NVDA", "Nvidia unveils chip")))
    assert "[NVDA]" in prompt and "Nvidia unveils chip" in prompt
