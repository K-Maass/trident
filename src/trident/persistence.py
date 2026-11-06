"""Money-path persistence (Phase 2).

Golden rule 2: every decision is logged BEFORE any order is sent. The Book is
event-sourced from start_cash + fills (the fills table is the ledger's event log and the
single source of truth for per-agent P&L), so positions/equity_snapshots are materialized
views we can rebuild at any time. Dividends (broker activities) are a later Phase 2 add;
until then the book reflects trades only.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal

from sqlalchemy import text

from trident.db import get_engine
from trident.ledger.book import Book
from trident.models import AgentName, AgentStatus, Decision, Fill, InvestmentThesis, Order, Side

_OPEN_STATUSES = ("pending", "submitted", "partially_filled")


# --------------------------------------------------------------- decisions
async def log_decision(d: Decision) -> None:
    async with get_engine().begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO decisions (id, agent_id, asof, ticker, target_weight, horizon_days, "
                "rationale, inputs_digest) VALUES (:id,:a,:asof,:t,:w,:h,CAST(:r AS JSONB),:d)"
            ),
            {
                "id": d.id,
                "a": d.agent.value,
                "asof": d.asof,
                "t": d.ticker,
                "w": d.target_weight,
                "h": d.horizon_days,
                "r": json.dumps(d.rationale),
                "d": d.inputs_digest,
            },
        )


# ------------------------------------------------------------------ orders
async def log_order(o: Order) -> None:
    async with get_engine().begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO orders (id, decision_id, agent_id, client_order_id, broker_order_id, "
                "ticker, side, qty, limit_price, status, risk_notes) "
                "VALUES (:id,:did,:a,:coid,:boid,:t,:side,:qty,:lp,:st,CAST(:rn AS JSONB))"
            ),
            {
                "id": o.id,
                "did": o.decision_id,
                "a": o.agent.value,
                "coid": o.client_order_id,
                "boid": o.broker_order_id,
                "t": o.ticker,
                "side": o.side.value,
                "qty": o.qty,
                "lp": o.limit_price,
                "st": o.status.value,
                "rn": json.dumps(o.risk_notes),
            },
        )


async def update_order(
    client_order_id: str, status: str, broker_order_id: str | None = None
) -> None:
    async with get_engine().begin() as conn:
        await conn.execute(
            text(
                "UPDATE orders SET status=:st, broker_order_id=COALESCE(:boid, broker_order_id) "
                "WHERE client_order_id=:coid"
            ),
            {"st": status, "boid": broker_order_id, "coid": client_order_id},
        )


async def open_order_exists(agent: AgentName, ticker: str, side: Side) -> bool:
    """Idempotency guard (doc 06): a live order for this agent+ticker+side already exists."""
    async with get_engine().connect() as conn:
        n = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM orders WHERE agent_id=:a AND ticker=:t AND side=:s "
                    "AND status = ANY(:open)"
                ),
                {
                    "a": agent.value,
                    "t": ticker.upper(),
                    "s": side.value,
                    "open": list(_OPEN_STATUSES),
                },
            )
        ).scalar_one()
    return n > 0


# ------------------------------------------------------------------- fills
async def log_fill(fill: Fill, realized_pnl: Decimal | None = None) -> None:
    async with get_engine().begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO fills (order_id, agent_id, ticker, side, qty, price, fees, "
                "realized_pnl, ts) VALUES (:oid,:a,:t,:side,:qty,:price,:fees,:pnl,:ts)"
            ),
            {
                "oid": fill.order_id,
                "a": fill.agent.value,
                "t": fill.ticker,
                "side": fill.side.value,
                "qty": fill.qty,
                "price": fill.price,
                "fees": fill.fees,
                "pnl": None if realized_pnl is None else float(realized_pnl),
                "ts": fill.ts,
            },
        )


# --------------------------------------------------------- book (event-sourced)
async def load_book(agent: AgentName) -> Book:
    """Rebuild the agent's Book from start_cash + every fill, in order. The fills table is
    the ledger's event log, so this is always the authoritative current state."""
    async with get_engine().connect() as conn:
        start_cash = (
            await conn.execute(
                text("SELECT start_cash FROM agents WHERE id=:a"), {"a": agent.value}
            )
        ).scalar_one()
        rows = (
            await conn.execute(
                text(
                    "SELECT ticker, side, qty, price, fees, ts FROM fills "
                    "WHERE agent_id=:a ORDER BY ts, id"
                ),
                {"a": agent.value},
            )
        ).all()
    book = Book.new(agent, Decimal(str(start_cash)))
    for r in rows:
        book.apply_fill(
            Fill(
                order_id="replay",
                agent=agent,
                ticker=r.ticker,
                side=Side(r.side),
                qty=float(r.qty),
                price=float(r.price),
                fees=float(r.fees),
                ts=r.ts,
            )
        )
    return book


async def save_positions(book: Book, ts: datetime) -> None:
    """Materialize the positions table from the book (delete-then-insert per agent)."""
    snap = book.snapshot(ts)
    async with get_engine().begin() as conn:
        await conn.execute(text("DELETE FROM positions WHERE agent_id=:a"), {"a": book.agent.value})
        for p in snap.positions:
            await conn.execute(
                text(
                    "INSERT INTO positions (agent_id, ticker, qty, avg_cost, updated_at) "
                    "VALUES (:a,:t,:q,:c,:ts)"
                ),
                {"a": book.agent.value, "t": p.ticker, "q": p.qty, "c": p.avg_cost, "ts": ts},
            )


# --------------------------------------------------------------- snapshots
async def write_equity_snapshot(
    agent: AgentName, ts: datetime, equity: float | Decimal, cash: float | Decimal
) -> None:
    async with get_engine().begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO equity_snapshots (agent_id, ts, equity, cash) VALUES (:a,:ts,:e,:c) "
                "ON CONFLICT (agent_id, ts) DO UPDATE SET equity=EXCLUDED.equity, "
                "cash=EXCLUDED.cash"
            ),
            {"a": agent.value, "ts": ts, "e": float(equity), "c": float(cash)},
        )


# --------------------------------------------------------------- theses (ORACLE)
async def upsert_thesis(t: InvestmentThesis) -> None:
    """Persist one thesis, upserting on (agent_id, ticker, asof) so re-running a cycle
    overwrites rather than duplicates. The tool_trace is stored STRUCTURALLY (JSONB) so every
    citation stays auditable against the exact result it cited."""
    async with get_engine().begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO theses (agent_id, asof, ticker, rating, conviction, target_weight, "
                "thesis, key_risks, invalidation_triggers, horizon_days, tool_trace, citations, "
                "evidence_basis, narrative_summary) VALUES "
                "(:a,:asof,:t,:rating,:conv,:tw,:thesis,CAST(:kr AS JSONB),CAST(:it AS JSONB),:hd,"
                "CAST(:tt AS JSONB),CAST(:cit AS JSONB),CAST(:eb AS JSONB),:ns) "
                "ON CONFLICT (agent_id, ticker, asof) DO UPDATE SET "
                "rating=EXCLUDED.rating, conviction=EXCLUDED.conviction, "
                "target_weight=EXCLUDED.target_weight, thesis=EXCLUDED.thesis, "
                "key_risks=EXCLUDED.key_risks, horizon_days=EXCLUDED.horizon_days, "
                "invalidation_triggers=EXCLUDED.invalidation_triggers, "
                "tool_trace=EXCLUDED.tool_trace, citations=EXCLUDED.citations, "
                "evidence_basis=EXCLUDED.evidence_basis, "
                "narrative_summary=EXCLUDED.narrative_summary"
            ),
            {
                "a": t.agent.value,
                "asof": t.asof,
                "t": t.ticker,
                "rating": t.rating.value,
                "conv": t.conviction,
                "tw": t.target_weight,
                "thesis": t.thesis,
                "kr": json.dumps(t.key_risks),
                "it": json.dumps(t.invalidation_triggers),
                "hd": t.horizon_days,
                "tt": json.dumps(t.tool_trace or {}),
                "cit": json.dumps([c.model_dump() for c in t.citations]),
                "eb": json.dumps(t.evidence_basis),
                "ns": t.narrative_summary,
            },
        )


async def get_theses(agent: AgentName, asof: datetime | None = None) -> list[dict]:
    """Stored theses for an agent (all, or one asof), newest first — verification + artifact."""
    sql = (
        "SELECT ticker, rating, conviction, target_weight, thesis, key_risks, "
        "invalidation_triggers, horizon_days, tool_trace, citations, evidence_basis, "
        "narrative_summary, asof FROM theses WHERE agent_id=:a"
    )
    params: dict = {"a": agent.value}
    if asof is not None:
        sql += " AND asof=:asof"
        params["asof"] = asof
    sql += " ORDER BY asof DESC, ticker"
    async with get_engine().connect() as conn:
        rows = (await conn.execute(text(sql), params)).all()
    return [dict(r._mapping) for r in rows]


# ---------------------------------------------------------- agent status / HWM
async def get_status(agent: AgentName) -> AgentStatus:
    async with get_engine().connect() as conn:
        s = (
            await conn.execute(text("SELECT status FROM agents WHERE id=:a"), {"a": agent.value})
        ).scalar_one()
    return AgentStatus(s)


async def set_status(agent: AgentName, status: AgentStatus) -> None:
    async with get_engine().begin() as conn:
        await conn.execute(
            text("UPDATE agents SET status=:s WHERE id=:a"),
            {"s": status.value, "a": agent.value},
        )


async def get_high_water_mark(agent: AgentName) -> Decimal:
    async with get_engine().connect() as conn:
        hwm = (
            await conn.execute(
                text("SELECT high_water_mark FROM agents WHERE id=:a"), {"a": agent.value}
            )
        ).scalar_one()
    return Decimal(str(hwm))


async def update_high_water_mark(agent: AgentName, equity: float | Decimal) -> Decimal:
    """Ratchet the high-water mark up to `equity`; returns the new HWM."""
    async with get_engine().begin() as conn:
        hwm = (
            await conn.execute(
                text(
                    "UPDATE agents SET high_water_mark = GREATEST(high_water_mark, :e) "
                    "WHERE id=:a RETURNING high_water_mark"
                ),
                {"e": float(equity), "a": agent.value},
            )
        ).scalar_one()
    return Decimal(str(hwm))


# ------------------------------------------------------------ reconciliation
async def sum_ledger_cash() -> Decimal:
    """Total cash across all agent books (event-sourced). Reconciled against broker cash:
    since ledger shares are separately reconciled against broker shares, cash equality is
    the mark-independent form of doc 01:55's equity invariant — and it fails LOUD on an
    unledgered cash event (e.g. a dividend before broker activities are wired)."""
    total = Decimal("0")
    for agent in AgentName:
        book = await load_book(agent)
        total += book.cash
    return total


async def sum_ledger_positions() -> dict[str, float]:
    """Aggregate net share quantity per ticker across ALL agent books (for reconciliation
    against broker positions)."""
    async with get_engine().connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT ticker, sum(qty) AS q FROM positions "
                    "GROUP BY ticker HAVING sum(qty) <> 0"
                )
            )
        ).all()
    return {r.ticker: float(r.q) for r in rows}
