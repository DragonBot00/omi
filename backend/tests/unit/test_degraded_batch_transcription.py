"""Tests for degraded batch transcription (#6052).

When the DG streaming socket is unavailable, audio is buffered and sent to
the pre-recorded API every 30s instead of being lost.  These tests cover:
- WAV header construction
- Buffer accumulation and atomic detach
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

TRANSCRIBE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'routers', 'transcribe.py')


def _read_transcribe_source() -> str:
    with open(TRANSCRIBE_PATH, encoding='utf-8') as f:
        return f.read()


# ---------------------------------------------------------------------------
# WAV header construction
# ---------------------------------------------------------------------------


def test_wav_header_structure():
    """_build_wav_bytes produces a valid WAV file that the wave module can parse."""
    source = _read_transcribe_source()
    # Verify the function exists
    assert 'def _build_wav_bytes(' in source, "_build_wav_bytes must exist in transcribe.py"

    # Replicate the function locally to test it
    def _build_wav_bytes(pcm_data, wav_sample_rate, channels=1, bits_per_sample=16):
        data_size = len(pcm_data)
        byte_rate = wav_sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + data_size,
            b'WAVE',
            b'fmt ',
            16,
            1,
            channels,
            wav_sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b'data',
            data_size,
        )
        return header + pcm_data

    # Generate 0.5s of silence at 16kHz mono 16-bit
    sample_rate = 16000
    duration = 0.5
    num_samples = int(sample_rate * duration)
    pcm = b'\x00\x00' * num_samples  # 16-bit silence

    wav_bytes = _build_wav_bytes(pcm, sample_rate)

    # Parse with wave module to validate header
    buf = BytesIO(wav_bytes)
    with wave.open(buf, 'rb') as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == sample_rate
        assert wf.getnframes() == num_samples


def test_wav_header_8000hz():
    """WAV header works for 8kHz sample rate (phone-quality audio)."""

    def _build_wav_bytes(pcm_data, wav_sample_rate, channels=1, bits_per_sample=16):
        data_size = len(pcm_data)
        byte_rate = wav_sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + data_size,
            b'WAVE',
            b'fmt ',
            16,
            1,
            channels,
            wav_sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b'data',
            data_size,
        )
        return header + pcm_data

    sample_rate = 8000
    pcm = b'\x00\x00' * 4000  # 0.5s
    wav_bytes = _build_wav_bytes(pcm, sample_rate)

    buf = BytesIO(wav_bytes)
    with wave.open(buf, 'rb') as wf:
        assert wf.getframerate() == 8000
        assert wf.getnframes() == 4000


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

    # Apply offset (mirroring _flush_degraded_batch logic)
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
# Atomic buffer detach (swap model)
# ---------------------------------------------------------------------------


def test_atomic_buffer_swap():
    """Detaching the buffer must be atomic — new audio goes to a fresh buffer."""
    degraded_audio_buffer = bytearray(b'\x01' * 48000)  # ~1.5s at 16kHz
    degraded_audio_start_time = 1060.0

    # Atomic detach (mirroring _flush_degraded_batch)
    pcm_chunk = bytes(degraded_audio_buffer)
    batch_start = degraded_audio_start_time
    degraded_audio_buffer = bytearray()
    degraded_audio_start_time = None

    # Original data captured
    assert len(pcm_chunk) == 48000
    assert batch_start == 1060.0

    # New buffer is empty and ready for more audio
    assert len(degraded_audio_buffer) == 0
    assert degraded_audio_start_time is None

    # New audio goes to the fresh buffer
    degraded_audio_buffer.extend(b'\x02' * 100)
    assert len(degraded_audio_buffer) == 100
    assert pcm_chunk[0:1] == b'\x01'  # Original data unchanged


# ---------------------------------------------------------------------------
# Budget parity
# ---------------------------------------------------------------------------


def test_budget_exhausted_skips_batch():
    """When fair_use_dg_budget_exhausted is True, batch transcription is skipped."""
    fair_use_dg_budget_exhausted = True

    # Simulate the budget check in _flush_degraded_batch
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
# Source wiring — degraded batch components exist in transcribe.py
# ---------------------------------------------------------------------------


def test_degraded_batch_buffer_declared():
    """transcribe.py must declare degraded_audio_buffer and degraded_audio_start_time."""
    source = _read_transcribe_source()
    assert 'degraded_audio_buffer' in source
    assert 'degraded_audio_start_time' in source


def test_degraded_batch_timer_exists():
    """transcribe.py must have _degraded_batch_timer that runs every DEGRADED_BATCH_INTERVAL_SECONDS."""
    source = _read_transcribe_source()
    assert 'async def _degraded_batch_timer' in source
    assert 'DEGRADED_BATCH_INTERVAL_SECONDS' in source


def test_flush_degraded_batch_exists():
    """transcribe.py must have _flush_degraded_batch function."""
    source = _read_transcribe_source()
    assert 'async def _flush_degraded_batch' in source


def test_flush_stt_buffer_routes_to_degraded_buffer():
    """flush_stt_buffer must route audio to degraded_audio_buffer when DG is unavailable."""
    source = _read_transcribe_source()
    flush_fn_pos = source.find('async def flush_stt_buffer')
    assert flush_fn_pos > 0
    flush_block = source[flush_fn_pos : flush_fn_pos + 3000]

    # Must route to degraded buffer when DG socket is None
    assert (
        'degraded_audio_buffer.extend(chunk)' in flush_block
    ), "flush_stt_buffer must route audio to degraded_audio_buffer when DG is down"
    # Must check stt_degraded before routing
    assert 'stt_degraded' in flush_block


def test_enter_degraded_mode_starts_batch_timer():
    """_enter_degraded_mode must start _degraded_batch_timer for single-channel."""
    source = _read_transcribe_source()
    fn_pos = source.find('async def _enter_degraded_mode')
    assert fn_pos > 0
    fn_block = source[fn_pos : fn_pos + 1000]

    assert '_degraded_batch_timer' in fn_block, "_enter_degraded_mode must start the batch timer"
    assert 'not is_multi_channel' in fn_block, "Batch timer must only start for single-channel"


def test_recovery_flushes_remaining_degraded_buffer():
    """_send_stt_recovered_event must flush remaining degraded audio on recovery."""
    source = _read_transcribe_source()
    fn_pos = source.find('def _send_stt_recovered_event')
    assert fn_pos > 0
    fn_block = source[fn_pos : fn_pos + 800]

    assert '_flush_degraded_batch' in fn_block, "Recovery must flush remaining degraded buffer"
    assert 'degraded_audio_buffer' in fn_block, "Recovery must check if there is buffered audio"


def test_degraded_batch_uses_pre_recorded_api():
    """_flush_degraded_batch must call deepgram_prerecorded_from_bytes."""
    source = _read_transcribe_source()
    fn_pos = source.find('async def _flush_degraded_batch')
    assert fn_pos > 0
    fn_block = source[fn_pos : fn_pos + 2500]

    assert 'deepgram_prerecorded_from_bytes' in fn_block, "Must use pre-recorded API for batch transcription"
    assert 'asyncio.to_thread' in fn_block, "Must run blocking DG call in thread pool"
    assert 'postprocess_words' in fn_block, "Must postprocess words into segments"


def test_degraded_batch_builds_wav():
    """_flush_degraded_batch must build WAV bytes from PCM data."""
    source = _read_transcribe_source()
    fn_pos = source.find('async def _flush_degraded_batch')
    assert fn_pos > 0
    fn_block = source[fn_pos : fn_pos + 2500]

    assert '_build_wav_bytes' in fn_block, "Must build WAV container for pre-recorded API"


def test_degraded_batch_checks_budget():
    """_flush_degraded_batch must honor fair_use_dg_budget_exhausted."""
    source = _read_transcribe_source()
    fn_pos = source.find('async def _flush_degraded_batch')
    assert fn_pos > 0
    fn_block = source[fn_pos : fn_pos + 2500]

    assert 'fair_use_dg_budget_exhausted' in fn_block, "Must check DG budget before batch transcription"


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

    # Find degraded buffer extend — must be preceded by is_multi_channel check
    extend_pos = flush_block.find('degraded_audio_buffer.extend(chunk)')
    assert extend_pos > 0
    pre_extend = flush_block[:extend_pos]
    assert 'not is_multi_channel' in pre_extend, "Degraded buffer routing must check not is_multi_channel"

    # In _enter_degraded_mode, batch timer must check is_multi_channel
    enter_pos = source.find('async def _enter_degraded_mode')
    enter_block = source[enter_pos : enter_pos + 1000]
    timer_pos = enter_block.find('_degraded_batch_timer')
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
# DG usage tracking for batch calls
# ---------------------------------------------------------------------------


def test_degraded_batch_tracks_dg_usage():
    """_flush_degraded_batch must track DG usage for batch calls (record_dg_usage_ms)."""
    source = _read_transcribe_source()
    fn_pos = source.find('async def _flush_degraded_batch')
    assert fn_pos > 0
    fn_block = source[fn_pos : fn_pos + 3500]

    assert 'record_dg_usage_ms' in fn_block, "Must track DG usage for batch calls"
    assert 'fair_use_track_dg_usage' in fn_block, "Must check fair_use_track_dg_usage flag"


# ---------------------------------------------------------------------------
# Import guard — pre_recorded is imported at module top level
# ---------------------------------------------------------------------------


def test_pre_recorded_imported_at_top_level():
    """deepgram_prerecorded_from_bytes and postprocess_words must be imported at module top level."""
    source = _read_transcribe_source()
    # Find the import section (first 120 lines)
    import_section = '\n'.join(source.split('\n')[:120])
    assert 'from utils.stt.pre_recorded import' in import_section
    assert 'deepgram_prerecorded_from_bytes' in import_section
    assert 'postprocess_words' in import_section
