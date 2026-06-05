"""Phase 9b — Empty-run digest tests.

Covers the _build_empty_digest formatter and the runner routing logic that
decides between full digest / empty digest / silent based on
TELEGRAM_SEND_EMPTY_DIGEST and whether any segment produced leads.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sinks.telegram_sink import TelegramSink


@pytest.fixture
def tg_settings(test_settings):
    test_settings.TELEGRAM_BOT_TOKEN = "test-bot-token"
    test_settings.TELEGRAM_CHAT_ID = "12345"
    test_settings.SHEET_ID = "sheet-abc"
    return test_settings


@pytest.fixture
def sink(tg_settings):
    return TelegramSink(settings=tg_settings)


def _make_mock_bot(message_id=555):
    bot = MagicMock()
    msg = MagicMock()
    msg.message_id = message_id
    bot.send_message = AsyncMock(return_value=msg)
    return bot


_SEGMENT_STATS = {
    "tutrain": {"hunt_count": 0, "qualified_count": 0, "after_dedupe": 0},
    "eqourse_content": {"hunt_count": 8, "qualified_count": 0, "after_dedupe": 3},
    "eqourse_ai_data": {"hunt_count": 25, "qualified_count": 0, "after_dedupe": 4},
}


# ---------------------------------------------------------------------------
# Formatter tests
# ---------------------------------------------------------------------------

def test_empty_digest_builds_correct_format(sink):
    text = sink._build_empty_digest(_SEGMENT_STATS, "run-empty-1", "https://sheets.test")
    assert isinstance(text, str)
    assert 0 < len(text) <= 4096
    # Title and the "no qualified leads" line
    assert "Lead Gen Run" in text
    assert "No qualified leads across" in text
    # Run id escaped (hyphen → \-)
    assert "run\\-empty\\-1" in text


def test_empty_digest_includes_funnel_breakdown(sink):
    text = sink._build_empty_digest(_SEGMENT_STATS, "run-empty-2", "")
    # Each segment label should appear (title-cased, underscores → spaces)
    assert "Tutrain" in text
    assert "Eqourse Content" in text
    assert "Eqourse Ai Data" in text
    # Dedupe drop annotation for eqourse_content (8 hunt, 3 after dedupe → 5 dropped)
    assert "5 dropped at dedupe" in text


def test_empty_digest_handles_missing_dedupe_field(sink):
    stats = {"tutrain": {"hunt_count": 0, "qualified_count": 0}}  # no after_dedupe
    text = sink._build_empty_digest(stats, "run-x", "")
    assert "Tutrain" in text
    # No dedupe annotation when field absent
    assert "dropped at dedupe" not in text


@pytest.mark.asyncio
async def test_send_empty_run_digest_returns_message_id(sink):
    mock_bot = _make_mock_bot()
    with patch.object(sink, "_make_bot", return_value=mock_bot):
        msg_id = await sink.send_empty_run_digest(_SEGMENT_STATS, "run-empty-3", "https://sheets.test")
    assert msg_id == 555
    assert mock_bot.send_message.call_args.kwargs.get("parse_mode") == "MarkdownV2"


@pytest.mark.asyncio
async def test_send_empty_run_digest_returns_none_on_error(sink):
    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=RuntimeError("telegram down"))
    with patch.object(sink, "_make_bot", return_value=bot):
        msg_id = await sink.send_empty_run_digest(_SEGMENT_STATS, "run-empty-4", "")
    assert msg_id is None


# ---------------------------------------------------------------------------
# Runner routing tests
# ---------------------------------------------------------------------------

def _make_settings(tmp_path, send_empty: bool):
    from scripts.init_db import init_db
    db_path = tmp_path / "leads.db"
    init_db(str(db_path))
    from config.settings import Settings
    return Settings(
        _env_file=None,
        GEMINI_API_KEY="test-key",
        HUNTER_API_KEY="h", SCRAPEGRAPH_API_KEY="s", SERPAPI_KEY="se",
        NEWSDATA_API_KEY="n", COMPANIES_API_TOKEN="c", APIFY_TOKEN_1="a",
        SQLITE_PATH=str(db_path),
        TELEGRAM_BOT_TOKEN="bot", TELEGRAM_CHAT_ID="chat", SHEET_ID="sheet",
        TELEGRAM_SEND_EMPTY_DIGEST=send_empty,
    )


def _empty_segment_state(segment):
    """A state dict representing a segment that produced 0 validated leads."""
    from orchestrator.state import make_initial_state
    from agents._models import HuntResult
    state = make_initial_state(segment, f"{segment}_run", 30)
    state["hunt_result"] = HuntResult(
        segment=segment, run_id=f"{segment}_run",
        candidates=[], source_counts={}, merged_count=8, after_filter=8,
        after_dedupe=3, enriched_count=0, api_credits_used={}, errors=[],
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc), duration_seconds=1.0,
    )
    state["qualified_result"] = None
    state["validated_result"] = None
    state["final_status"] = "success"
    return state


@pytest.mark.asyncio
async def test_runner_sends_empty_digest_when_setting_true_and_zero_leads(tmp_path):
    from orchestrator.runner import PipelineRunner

    settings = _make_settings(tmp_path, send_empty=True)

    async def zero_lead_segment(segment, target_count=30, thread_id=None):
        return _empty_segment_state(segment)

    sent = {"empty": 0, "full": 0}

    async def fake_empty(segment_stats, run_id, sheets_url=""):
        sent["empty"] += 1
        return 123

    async def fake_full(*a, **k):
        sent["full"] += 1
        return {"message_id": 1, "leads_included": 0}

    async with PipelineRunner(settings) as runner:
        runner.run_segment = zero_lead_segment
        runner.agents.icp_strategist.list_segments = lambda: ["tutrain", "eqourse_content", "eqourse_ai_data"]
        runner.agents.telegram_sink.send_empty_run_digest = fake_empty
        runner.agents.telegram_sink.send_run_digest = fake_full
        await runner.run_all_segments(target_count=30)

    assert sent["empty"] == 1
    assert sent["full"] == 0


@pytest.mark.asyncio
async def test_runner_skips_digest_when_setting_false_and_zero_leads(tmp_path):
    from orchestrator.runner import PipelineRunner

    settings = _make_settings(tmp_path, send_empty=False)

    async def zero_lead_segment(segment, target_count=30, thread_id=None):
        return _empty_segment_state(segment)

    sent = {"empty": 0, "full": 0}

    async def fake_empty(segment_stats, run_id, sheets_url=""):
        sent["empty"] += 1
        return 123

    async def fake_full(*a, **k):
        sent["full"] += 1
        return {"message_id": 1, "leads_included": 0}

    async with PipelineRunner(settings) as runner:
        runner.run_segment = zero_lead_segment
        runner.agents.icp_strategist.list_segments = lambda: ["tutrain", "eqourse_content", "eqourse_ai_data"]
        runner.agents.telegram_sink.send_empty_run_digest = fake_empty
        runner.agents.telegram_sink.send_run_digest = fake_full
        await runner.run_all_segments(target_count=30)

    assert sent["empty"] == 0
    assert sent["full"] == 0


@pytest.mark.asyncio
async def test_runner_sends_full_digest_when_leads_present(tmp_path):
    """When at least one segment has validated leads, full digest fires
    regardless of the empty-digest setting."""
    from orchestrator.runner import PipelineRunner
    from agents._models import ValidatedResult

    settings = _make_settings(tmp_path, send_empty=False)  # even with empty=False

    def _validated(segment):
        now = datetime.now(timezone.utc)
        return ValidatedResult(
            segment=segment, run_id=f"{segment}_run",
            validated_candidates=[],
            stats={"ready_to_send": 1, "needs_review": 0, "rejected": 0},
            api_credits_used={}, started_at=now, completed_at=now, duration_seconds=1.0,
        )

    async def lead_segment(segment, target_count=30, thread_id=None):
        state = _empty_segment_state(segment)
        if segment == "eqourse_ai_data":
            state["validated_result"] = _validated(segment)
        return state

    sent = {"empty": 0, "full": 0}

    async def fake_empty(segment_stats, run_id, sheets_url=""):
        sent["empty"] += 1
        return 123

    async def fake_full(validated_by_segment, sheets_url=""):
        sent["full"] += 1
        return {"message_id": 1, "leads_included": 1}

    async with PipelineRunner(settings) as runner:
        runner.run_segment = lead_segment
        runner.agents.icp_strategist.list_segments = lambda: ["tutrain", "eqourse_content", "eqourse_ai_data"]
        runner.agents.telegram_sink.send_empty_run_digest = fake_empty
        runner.agents.telegram_sink.send_run_digest = fake_full
        await runner.run_all_segments(target_count=30)

    assert sent["full"] == 1
    assert sent["empty"] == 0
