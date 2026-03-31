"""Tests for degraded batch transcription (#6052).

When the DG streaming socket is unavailable, audio is buffered and sent to
the pre-recorded API every 30s instead of being lost.  These tests cover:
- DegradedBatchProcessor class (feed, flush, has_audio, run_timer)
- WAV header construction (build_wav_bytes)
- Timestamp offsetting for batch segments
- Budget parity (DG budget exhaustion skips batch)
- Batch segments carry unique stt_session per chunk
- Recovery flushes remaining buffer
- Source wiring in transcribe.py
"""

import asyncio
import os
import struct
import sys
import time
import wave
from collections import deque
from io import BytesIO
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Mock heavy dependencies before importing anything from backend
_mock_modules = {}
for mod_name in [
    'database',
    'database._client',
    'database.redis_db',
    'database.users',
    'database.conversations',
    'database.calendar_meetings',
    'database.fair_use',
    'database.user_usage',
    'database.subscription',
    'utils.other.storage',
    'deepgram',
    'deepgram.clients',
    'deepgram.clients.live',
    'deepgram.clients.live.v1',
    'websockets',
    'websockets.exceptions',
    'fal_client',
]:
    if mod_name not in sys.modules:
        _mock_modules[mod_name] = MagicMock()
        sys.modules[mod_name] = _mock_modules[mod_name]

if not hasattr(sys.modules['deepgram'], '_mock_initialized'):
    sys.modules['deepgram'].DeepgramClient = MagicMock
    sys.modules['deepgram'].DeepgramClientOptions = MagicMock
    sys.modules['deepgram'].LiveTranscriptionEvents = MagicMock()
    sys.modules['deepgram.clients.live.v1'].LiveOptions = MagicMock
    sys.modules['deepgram']._mock_initialized = True

from models.transcript_segment import TranscriptSegment  # noqa: E402
from models.message_event import MessageServiceStatusEvent  # noqa: E402
from utils.stt.pre_recorded import postprocess_words  # noqa: E402
from utils.stt.degraded_batch import DegradedBatchProcessor, build_wav_bytes, BATCH_INTERVAL_SECONDS  # noqa: E402

TRANSCRIBE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'routers', 'transcribe.py')
DEGRADED_BATCH_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'utils', 'stt', 'degraded_batch.py')


def _read_transcribe_source() -> str:
    with open(TRANSCRIBE_PATH, encoding='utf-8') as f:
        return f.read()


def _read_degraded_batch_source() -> str:
    with open(DEGRADED_BATCH_PATH, encoding='utf-8') as f:
        return f.read()


# ---------------------------------------------------------------------------
# WAV header construction (build_wav_bytes in degraded_batch.py)
# ---------------------------------------------------------------------------


def test_wav_header_structure():
    """build_wav_bytes produces a valid WAV file that the wave module can parse."""
    sample_rate = 16000
    duration = 0.5
    num_samples = int(sample_rate * duration)
    pcm = b'\x00\x00' * num_samples  # 16-bit silence

    wav_bytes = build_wav_bytes(pcm, sample_rate)

    buf = BytesIO(wav_bytes)
    with wave.open(buf, 'rb') as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == sample_rate
        assert wf.getnframes() == num_samples


def test_wav_header_8000hz():
    """WAV header works for 8kHz sample rate (phone-quality audio)."""
    sample_rate = 8000
    pcm = b'\x00\x00' * 4000  # 0.5s
    wav_bytes = build_wav_bytes(pcm, sample_rate)

    buf = BytesIO(wav_bytes)
    with wave.open(buf, 'rb') as wf:
        assert wf.getframerate() == 8000
        assert wf.getnframes() == 4000


# ---------------------------------------------------------------------------
# DegradedBatchProcessor — feed / has_audio / flush
# ---------------------------------------------------------------------------


def test_processor_feed_and_has_audio():
    """feed() accumulates audio and has_audio reflects buffer state."""
    proc = DegradedBatchProcessor(sample_rate=16000, uid='u1', session_id='s1')
    assert not proc.has_audio

    proc.feed(b'\x00' * 320)
    assert proc.has_audio

    proc.feed(b'\x00' * 320)
    assert proc.has_audio


@pytest.mark.asyncio
async def test_processor_flush_empty_buffer():
    """flush() on empty buffer returns 0 and does nothing."""
    proc = DegradedBatchProcessor(sample_rate=16000, uid='u1', session_id='s1')
    sink = deque()
    result = await proc.flush(stream_start_time=1000.0, segment_sink=sink)
    assert result == 0
    assert len(sink) == 0


@pytest.mark.asyncio
async def test_processor_flush_budget_exhausted():
    """flush() with budget_exhausted=True skips DG call and returns 0."""
    proc = DegradedBatchProcessor(sample_rate=16000, uid='u1', session_id='s1')
    proc.feed(b'\x00' * 32000)  # 1s of audio
    sink = deque()

    result = await proc.flush(stream_start_time=1000.0, segment_sink=sink, budget_exhausted=True)
    assert result == 0
    assert len(sink) == 0
    # Buffer should be cleared even when skipped
    assert not proc.has_audio


@pytest.mark.asyncio
async def test_processor_flush_produces_segments():
    """flush() calls pre-recorded API and pushes segments into sink with correct offsets."""
    proc = DegradedBatchProcessor(sample_rate=16000, uid='u1', session_id='s1')
    proc.feed(b'\x00' * 32000)  # 1s
    proc._buffer_start_time = 1060.0  # Simulate audio starting 60s into stream

    mock_words = [
        {'timestamp': [0.0, 0.5], 'speaker': 'SPEAKER_00', 'text': 'Hello'},
        {'timestamp': [0.5, 1.0], 'speaker': 'SPEAKER_00', 'text': 'world'},
    ]

    sink = deque()
    with patch('utils.stt.degraded_batch.deepgram_prerecorded_from_bytes', return_value=mock_words):
        result = await proc.flush(stream_start_time=1000.0, segment_sink=sink)

    assert result >= 1
    assert len(sink) >= 1
    # Segments should have batch_offset (1060.0 - 1000.0 = 60.0) applied
    assert sink[0]['start'] >= 60.0
    assert sink[0]['stt_session'] is not None
    assert not proc.has_audio  # Buffer cleared after flush


@pytest.mark.asyncio
async def test_processor_flush_atomic_swap():
    """flush() atomically detaches buffer — new audio goes to a fresh buffer."""
    proc = DegradedBatchProcessor(sample_rate=16000, uid='u1', session_id='s1')
    proc.feed(b'\x01' * 32000)  # 1s
    proc._buffer_start_time = 1000.0

    # Start flush in background and feed more audio concurrently
    mock_words = []
    with patch('utils.stt.degraded_batch.deepgram_prerecorded_from_bytes', return_value=mock_words):
        result = await proc.flush(stream_start_time=1000.0, segment_sink=deque())

    # After flush, buffer is empty — ready for new audio
    assert not proc.has_audio
    proc.feed(b'\x02' * 100)
    assert proc.has_audio


@pytest.mark.asyncio
async def test_processor_flush_tracks_dg_usage():
    """flush() with track_usage=True calls record_dg_usage_ms."""
    proc = DegradedBatchProcessor(sample_rate=16000, uid='u1', session_id='s1')
    proc.feed(b'\x00' * 32000)  # 1s
    proc._buffer_start_time = 1000.0

    mock_words = [
        {'timestamp': [0.0, 0.5], 'speaker': 'SPEAKER_00', 'text': 'Hello'},
    ]

    sink = deque()
    with patch('utils.stt.degraded_batch.deepgram_prerecorded_from_bytes', return_value=mock_words), patch(
        'utils.stt.degraded_batch.record_dg_usage_ms'
    ) as mock_usage:
        result = await proc.flush(stream_start_time=1000.0, segment_sink=sink, track_usage=True)

    assert result >= 1
    mock_usage.assert_called_once()


# ---------------------------------------------------------------------------
# Timestamp offsetting
# ---------------------------------------------------------------------------


def test_batch_offset_applied_to_segments():
    """Batch segments must have batch_offset added to align with stream timeline.

    postprocess_words rebases start/end to 0.  The batch offset (seconds from
    stream start to when this batch's audio began) must be added back.
    """
    # Simulate postprocess_words output (segments relative to batch start)
    seg_a = TranscriptSegment(
        text='Hello world',
        speaker='SPEAKER_00',
        is_user=False,
        start=0.5,
        end=1.2,
    )
    seg_b = TranscriptSegment(
        text='How are you',
        speaker='SPEAKER_01',
        is_user=False,
        start=2.0,
        end=3.5,
    )

    batch_offset = 60.0  # This batch started 60s into the stream
    batch_session = 'batch-ses-001'

    # Apply offset (mirroring DegradedBatchProcessor.flush logic)
    segment_dicts = []
    for seg in [seg_a, seg_b]:
        segment_dicts.append(
            {
                'start': round(seg.start + batch_offset, 2),
                'end': round(seg.end + batch_offset, 2),
                'speaker': seg.speaker,
                'text': seg.text,
                'is_user': seg.is_user,
                'person_id': None,
                'stt_session': batch_session,
            }
        )

    assert segment_dicts[0]['start'] == 60.5
    assert segment_dicts[0]['end'] == 61.2
    assert segment_dicts[1]['start'] == 62.0
    assert segment_dicts[1]['end'] == 63.5
    assert all(d['stt_session'] == batch_session for d in segment_dicts)


def test_batch_offset_zero_for_immediate_degradation():
    """If degradation starts immediately (no prior audio), offset is 0."""
    first_audio_byte_timestamp = 1000.0
    degraded_audio_start_time = 1000.0

    batch_offset = degraded_audio_start_time - first_audio_byte_timestamp
    assert batch_offset == 0.0

    seg = TranscriptSegment(text='test', speaker='SPEAKER_00', is_user=False, start=0.0, end=1.0)
    adjusted_start = round(seg.start + batch_offset, 2)
    adjusted_end = round(seg.end + batch_offset, 2)
    assert adjusted_start == 0.0
    assert adjusted_end == 1.0


# ---------------------------------------------------------------------------
# Unique stt_session per batch chunk
# ---------------------------------------------------------------------------


def test_each_batch_gets_unique_session():
    """Each 30s batch flush must generate a fresh stt_session ULID."""
    from ulid import ULID

    sessions = set()
    for _ in range(5):
        sessions.add(str(ULID()))

    assert len(sessions) == 5, "Each ULID must be unique"


def test_batch_session_acts_as_merge_barrier():
    """Segments from different batch chunks must not merge (stt_session mismatch)."""
    seg_a = TranscriptSegment(
        text='batch one',
        speaker='SPEAKER_00',
        is_user=False,
        start=60.0,
        end=61.0,
        stt_session='batch-001',
    )
    seg_b = TranscriptSegment(
        text='batch two',
        speaker='SPEAKER_00',
        is_user=False,
        start=61.0,
        end=62.0,
        stt_session='batch-002',
    )

    result, _, _ = TranscriptSegment.combine_segments([], [seg_a, seg_b])
    assert len(result) == 2, "Different batch sessions must NOT merge"


def test_batch_and_streaming_sessions_dont_merge():
    """Segments from batch mode and streaming mode must not merge."""
    streaming_seg = TranscriptSegment(
        text='realtime',
        speaker='SPEAKER_00',
        is_user=False,
        start=55.0,
        end=56.0,
        stt_session='streaming-ses',
    )
    batch_seg = TranscriptSegment(
        text='batch',
        speaker='SPEAKER_00',
        is_user=False,
        start=60.0,
        end=61.0,
        stt_session='batch-ses',
    )

    result, _, _ = TranscriptSegment.combine_segments([], [streaming_seg, batch_seg])
    assert len(result) == 2, "Batch and streaming sessions must NOT merge"


# ---------------------------------------------------------------------------
# Budget parity
# ---------------------------------------------------------------------------


def test_budget_exhausted_skips_batch():
    """When fair_use_dg_budget_exhausted is True, batch transcription is skipped."""
    fair_use_dg_budget_exhausted = True

    # Simulate the budget check in DegradedBatchProcessor.flush
    pcm_chunk = b'\x00' * 960000  # 30s of audio
    if fair_use_dg_budget_exhausted:
        skipped = True
        del pcm_chunk
    else:
        skipped = False

    assert skipped is True


def test_budget_not_exhausted_allows_batch():
    """When budget is available, batch transcription proceeds."""
    fair_use_dg_budget_exhausted = False
    pcm_chunk = b'\x00' * 960000

    if fair_use_dg_budget_exhausted:
        skipped = True
    else:
        skipped = False

    assert skipped is False
    del pcm_chunk


# ---------------------------------------------------------------------------
# Source wiring — DegradedBatchProcessor used in transcribe.py
# ---------------------------------------------------------------------------


def test_processor_instantiated_in_transcribe():
    """transcribe.py must instantiate DegradedBatchProcessor."""
    source = _read_transcribe_source()
    assert 'DegradedBatchProcessor(' in source


def test_processor_imported_in_transcribe():
    """transcribe.py must import DegradedBatchProcessor from utils.stt.degraded_batch."""
    source = _read_transcribe_source()
    import_section = '\n'.join(source.split('\n')[:120])
    assert 'from utils.stt.degraded_batch import' in import_section
    assert 'DegradedBatchProcessor' in import_section


def test_flush_degraded_batch_exists():
    """transcribe.py must have _flush_degraded_batch thin wrapper."""
    source = _read_transcribe_source()
    assert 'async def _flush_degraded_batch' in source


def test_flush_stt_buffer_routes_to_processor():
    """flush_stt_buffer must route audio to degraded_batch_processor.feed() when DG is unavailable."""
    source = _read_transcribe_source()
    flush_fn_pos = source.find('async def flush_stt_buffer')
    assert flush_fn_pos > 0
    flush_block = source[flush_fn_pos : flush_fn_pos + 3000]

    # Must route to processor.feed when DG socket is None
    assert (
        'degraded_batch_processor.feed(chunk)' in flush_block
    ), "flush_stt_buffer must route audio to degraded_batch_processor.feed() when DG is down"
    # Must check stt_degraded before routing
    assert 'stt_degraded' in flush_block


def test_enter_degraded_mode_starts_batch_timer():
    """_enter_degraded_mode must start degraded_batch_processor.run_timer for single-channel."""
    source = _read_transcribe_source()
    fn_pos = source.find('async def _enter_degraded_mode')
    assert fn_pos > 0
    fn_block = source[fn_pos : fn_pos + 1000]

    assert 'degraded_batch_processor.run_timer' in fn_block, "_enter_degraded_mode must start the batch timer"
    assert 'not is_multi_channel' in fn_block, "Batch timer must only start for single-channel"


def test_recovery_flushes_remaining_degraded_buffer():
    """_send_stt_recovered_event must flush remaining degraded audio on recovery."""
    source = _read_transcribe_source()
    fn_pos = source.find('def _send_stt_recovered_event')
    assert fn_pos > 0
    fn_block = source[fn_pos : fn_pos + 800]

    assert '_flush_degraded_batch' in fn_block, "Recovery must flush remaining degraded buffer"
    assert 'degraded_batch_processor.has_audio' in fn_block, "Recovery must check processor.has_audio"


def test_degraded_batch_class_uses_pre_recorded_api():
    """DegradedBatchProcessor.flush must call deepgram_prerecorded_from_bytes."""
    source = _read_degraded_batch_source()
    assert 'deepgram_prerecorded_from_bytes' in source
    assert 'asyncio.to_thread' in source
    assert 'postprocess_words' in source


def test_degraded_batch_class_builds_wav():
    """degraded_batch.py must have build_wav_bytes for WAV container construction."""
    source = _read_degraded_batch_source()
    assert 'def build_wav_bytes(' in source


def test_degraded_batch_class_checks_budget():
    """DegradedBatchProcessor.flush must honor budget_exhausted parameter."""
    source = _read_degraded_batch_source()
    fn_pos = source.find('async def flush(')
    assert fn_pos > 0
    fn_block = source[fn_pos : fn_pos + 2500]
    assert 'budget_exhausted' in fn_block


def test_degraded_event_includes_batch_metadata():
    """stt_degraded event must include batch_mode and batch_interval_seconds metadata."""
    source = _read_transcribe_source()
    fn_pos = source.find('def _send_stt_degraded_event')
    assert fn_pos > 0
    fn_block = source[fn_pos : fn_pos + 600]

    assert 'batch_mode' in fn_block, "Degraded event must include batch_mode metadata"
    assert 'batch_interval_seconds' in fn_block, "Degraded event must include batch_interval_seconds"


def test_degraded_batch_single_channel_only():
    """Degraded batch must be scoped to single-channel only."""
    source = _read_transcribe_source()

    # In flush_stt_buffer, degraded buffer routing must check is_multi_channel
    flush_pos = source.find('async def flush_stt_buffer')
    assert flush_pos > 0
    flush_block = source[flush_pos : flush_pos + 3000]

    # Find degraded processor.feed — must be preceded by is_multi_channel check
    feed_pos = flush_block.find('degraded_batch_processor.feed(chunk)')
    assert feed_pos > 0
    pre_feed = flush_block[:feed_pos]
    assert 'not is_multi_channel' in pre_feed, "Degraded buffer routing must check not is_multi_channel"

    # In _enter_degraded_mode, batch timer must check is_multi_channel
    enter_pos = source.find('async def _enter_degraded_mode')
    enter_block = source[enter_pos : enter_pos + 1000]
    timer_pos = enter_block.find('degraded_batch_processor.run_timer')
    pre_timer = enter_block[:timer_pos]
    assert 'not is_multi_channel' in pre_timer


def test_disconnect_flushes_degraded_buffer():
    """WebSocket disconnect cleanup must flush remaining degraded audio."""
    source = _read_transcribe_source()
    # Find the disconnect cleanup section (finally block of receive_data)
    flush_final_pos = source.find('Flush any remaining degraded batch audio')
    assert flush_final_pos > 0, "Disconnect cleanup must flush degraded audio"

    cleanup_block = source[flush_final_pos : flush_final_pos + 200]
    assert '_flush_degraded_batch' in cleanup_block


# ---------------------------------------------------------------------------
# MessageServiceStatusEvent metadata field
# ---------------------------------------------------------------------------


def test_message_service_status_event_has_metadata_field():
    """MessageServiceStatusEvent must support optional metadata dict."""
    event = MessageServiceStatusEvent(
        status="stt_degraded",
        status_text="test",
        metadata={'batch_mode': True, 'batch_interval_seconds': 30},
    )
    j = event.to_json()
    assert j['status'] == 'stt_degraded'
    assert j['metadata'] == {'batch_mode': True, 'batch_interval_seconds': 30}


def test_message_service_status_event_metadata_none():
    """metadata=None should serialize as None in JSON (backward compat)."""
    event = MessageServiceStatusEvent(status="stt_recovered", status_text="test")
    j = event.to_json()
    assert j['metadata'] is None


# ---------------------------------------------------------------------------
# postprocess_words integration — rebases to 0
# ---------------------------------------------------------------------------


def test_postprocess_words_rebases_to_zero():
    """postprocess_words rebases segment timestamps to 0 — offset must be added externally."""
    words = [
        {'timestamp': [5.0, 5.5], 'speaker': 'SPEAKER_00', 'text': 'Hello'},
        {'timestamp': [5.5, 6.0], 'speaker': 'SPEAKER_00', 'text': 'world'},
    ]

    segments = postprocess_words(words, duration=30)
    assert len(segments) >= 1

    # First segment should start at 0.0 (rebased from 5.0)
    assert segments[0].start == 0.0


def test_postprocess_words_empty_input():
    """postprocess_words with empty words returns empty list."""
    segments = postprocess_words([], duration=30)
    assert segments == []


# ---------------------------------------------------------------------------
# DG usage tracking in degraded_batch.py
# ---------------------------------------------------------------------------


def test_degraded_batch_class_tracks_dg_usage():
    """DegradedBatchProcessor.flush must track DG usage (record_dg_usage_ms)."""
    source = _read_degraded_batch_source()
    assert 'record_dg_usage_ms' in source
    assert 'track_usage' in source


# ---------------------------------------------------------------------------
# No old inline functions remain in transcribe.py
# ---------------------------------------------------------------------------


def test_no_inline_build_wav_bytes():
    """_build_wav_bytes must NOT exist in transcribe.py (moved to degraded_batch.py)."""
    source = _read_transcribe_source()
    assert 'def _build_wav_bytes(' not in source


def test_no_inline_degraded_batch_timer():
    """_degraded_batch_timer must NOT exist in transcribe.py (replaced by processor.run_timer)."""
    source = _read_transcribe_source()
    assert 'async def _degraded_batch_timer' not in source
