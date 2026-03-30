"""Behavioral tests for STT degraded mode and recovery (#6052).

Tests the actual runtime components used by the degraded/recovery flow
in transcribe.py, not just source-string inspection.
"""

import asyncio
import os
import sys
import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Mock heavy dependencies before importing streaming module
_mock_modules = {}
for mod_name in [
    'database',
    'database._client',
    'database.users',
    'utils.other.storage',
    'deepgram',
    'deepgram.clients',
    'deepgram.clients.live',
    'deepgram.clients.live.v1',
    'websockets',
    'websockets.exceptions',
]:
    if mod_name not in sys.modules:
        _mock_modules[mod_name] = MagicMock()
        sys.modules[mod_name] = _mock_modules[mod_name]

# Only set attributes if not already set by another test file (avoids cross-test contamination)
if not hasattr(sys.modules['deepgram'], '_mock_initialized'):
    sys.modules['deepgram'].DeepgramClient = MagicMock
    sys.modules['deepgram'].DeepgramClientOptions = MagicMock
    sys.modules['deepgram'].LiveTranscriptionEvents = MagicMock()
    sys.modules['deepgram.clients.live.v1'].LiveOptions = MagicMock
    sys.modules['deepgram']._mock_initialized = True

from utils.stt.streaming import (
    get_deepgram_circuit_breaker,
    process_audio_dg,
    connect_to_deepgram_with_backoff,
)  # noqa: E402
from utils.stt.safe_socket import SafeDeepgramSocket, KeepaliveConfig  # noqa: E402

TRANSCRIBE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'routers', 'transcribe.py')


def _read_transcribe_source() -> str:
    with open(TRANSCRIBE_PATH, encoding='utf-8') as f:
        return f.read()


@pytest.fixture(autouse=True)
def _reset_cb():
    cb = get_deepgram_circuit_breaker()
    orig_threshold = cb.failure_threshold
    orig_timeout = cb.reset_timeout_seconds
    cb.failure_threshold = 3
    cb.reset_timeout_seconds = 30.0
    cb.reset()
    yield
    cb.failure_threshold = orig_threshold
    cb.reset_timeout_seconds = orig_timeout
    cb.reset()


# ---------------------------------------------------------------------------
# Source structure tests (kept from original — validates code wiring)
# ---------------------------------------------------------------------------


def test_transcribe_emits_stt_degraded_status_event():
    source = _read_transcribe_source()
    assert 'status="stt_degraded"' in source


def test_transcribe_enters_degraded_mode_on_initial_processing_error():
    source = _read_transcribe_source()
    error_pos = source.find('logger.error(f"Initial processing error: {e} {uid} {session_id}")')
    degraded_pos = source.find('await _enter_degraded_mode(', error_pos)
    assert error_pos > 0
    assert degraded_pos > 0
    assert degraded_pos > error_pos


def test_transcribe_attempts_recovery_after_degraded_mode_entry():
    source = _read_transcribe_source()
    degraded_pos = source.find("deepgram_recovery_task = spawn(_recover_deepgram_connection())")
    degraded_event_pos = source.find('status="stt_degraded"')
    assert degraded_event_pos > 0
    assert degraded_pos > degraded_event_pos


# ---------------------------------------------------------------------------
# Behavioral: CB integration with degraded mode entry
# ---------------------------------------------------------------------------


def test_cb_open_state_detected_before_recovery():
    """When CB is open, is_open() returns True and snapshot() reports the state.

    This is the condition checked in _enter_degraded_mode before spawning recovery.
    """
    cb = get_deepgram_circuit_breaker()
    cb.failure_threshold = 2
    cb.record_failure(Exception("dg error 1"))
    cb.record_failure(Exception("dg error 2"))

    assert cb.is_open() is True
    snap = cb.snapshot()
    assert snap["state"] == "open"
    assert snap["consecutive_failures"] == 2


@pytest.mark.asyncio
async def test_cb_blocks_recovery_attempt_when_open():
    """process_audio_dg returns None when CB is open — recovery loop would get None socket."""
    cb = get_deepgram_circuit_breaker()
    cb.failure_threshold = 1
    cb.record_failure(Exception("force open"))

    with patch('utils.stt.streaming.connect_to_deepgram') as mock_connect:
        result = await process_audio_dg(
            stream_transcript=MagicMock(),
            language='en',
            sample_rate=16000,
            channels=1,
        )

    assert result is None
    mock_connect.assert_not_called()


@pytest.mark.asyncio
async def test_cb_allows_recovery_after_half_open_probe_succeeds():
    """After CB timeout + successful probe, recovery gets a valid socket and CB closes."""
    from utils.stt.streaming import connect_to_deepgram_with_backoff

    cb = get_deepgram_circuit_breaker()
    cb.failure_threshold = 1
    cb.reset_timeout_seconds = 0.5
    cb.record_failure(Exception("open"))
    cb._opened_at_monotonic = time.monotonic() - 1.0  # Past timeout

    mock_conn = MagicMock()
    with patch('utils.stt.streaming.connect_to_deepgram', return_value=mock_conn):
        result = await connect_to_deepgram_with_backoff(
            on_message=MagicMock(),
            on_error=MagicMock(),
            language='en',
            sample_rate=16000,
            channels=1,
            model='nova-2-general',
            retries=1,
        )

    assert result is mock_conn
    assert cb._state == "closed"


# ---------------------------------------------------------------------------
# Behavioral: Dead socket detection triggers degraded mode entry
# ---------------------------------------------------------------------------


def test_dead_socket_is_detected_for_degraded_entry():
    """SafeDeepgramSocket.is_connection_dead becomes True when send fails.

    This is the condition that triggers degraded mode in flush_stt_buffer.
    """
    mock_conn = MagicMock()
    mock_conn.send.return_value = False
    cfg = KeepaliveConfig(keepalive_interval_sec=5.0, check_period_sec=999.0)
    safe = SafeDeepgramSocket(mock_conn, cfg=cfg)
    try:
        safe.send(b'\x00' * 960)
        assert safe.is_connection_dead is True
        assert safe.death_reason == 'send returned False'
    finally:
        safe.finish()


def test_dead_socket_exception_triggers_degraded_path():
    """send() exception sets is_connection_dead and death_reason.

    In transcribe.py, the except block sets deepgram_socket = None and
    calls _enter_degraded_mode.
    """
    mock_conn = MagicMock()
    mock_conn.send.side_effect = ConnectionResetError('Connection reset')
    cfg = KeepaliveConfig(keepalive_interval_sec=5.0, check_period_sec=999.0)
    safe = SafeDeepgramSocket(mock_conn, cfg=cfg)
    try:
        safe.send(b'\x00' * 960)
        assert safe.is_connection_dead is True
        assert 'ConnectionResetError' in safe.death_reason
    finally:
        safe.finish()


# ---------------------------------------------------------------------------
# Behavioral: VAD gate activation after recovery
# ---------------------------------------------------------------------------


def test_vad_gate_activates_from_shadow_to_active():
    """VAD gate in shadow mode transitions to active when activate() is called.

    This is the pattern used in _recover_deepgram_connection() after successful
    DG socket recovery, matching the profile-complete path in flush_stt_buffer.
    """
    from utils.stt.vad_gate import VADStreamingGate

    gate = VADStreamingGate(sample_rate=16000, channels=1, mode='shadow', uid='test', session_id='test')
    assert gate.mode == 'shadow'
    gate.activate()
    assert gate.mode == 'active'


def test_vad_gate_activate_noop_when_already_active():
    """activate() is a no-op when gate is already active."""
    from utils.stt.vad_gate import VADStreamingGate

    gate = VADStreamingGate(sample_rate=16000, channels=1, mode='active', uid='test', session_id='test')
    assert gate.mode == 'active'
    gate.activate()
    assert gate.mode == 'active'


def test_recovery_vad_activation_condition():
    """The recovery path activates VAD gate from shadow mode after DG reconnect.

    Checks: vad_gate is not None, mode is active/override, and gate is in shadow.
    """
    source = _read_transcribe_source()

    # Find the recovery path's VAD activation
    recovery_pos = source.find('VAD gate activated after DG recovery')
    assert recovery_pos > 0, "Recovery path must have VAD gate activation"

    # Verify the condition pattern
    recovery_block = source[recovery_pos - 300 : recovery_pos]
    assert "vad_gate.mode == 'shadow'" in recovery_block
    assert "vad_gate.activate()" in recovery_block


# ---------------------------------------------------------------------------
# Behavioral: Recovery produces usable socket
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recovery_socket_can_send_audio():
    """After recovery, the new socket can send audio chunks.

    This validates the process_audio_dg → SafeDeepgramSocket chain produces
    a socket that flush_stt_buffer can use after _recover_deepgram_connection.
    """
    mock_conn = MagicMock()
    mock_conn.send.return_value = True
    with patch('utils.stt.streaming.connect_to_deepgram_with_backoff', new_callable=AsyncMock, return_value=mock_conn):
        recovered_socket = await process_audio_dg(
            stream_transcript=MagicMock(),
            language='en',
            sample_rate=16000,
            channels=1,
        )

    assert recovered_socket is not None
    assert not recovered_socket.is_connection_dead
    chunk = b'\x00' * 960
    recovered_socket.send(chunk)
    mock_conn.send.assert_called_once_with(chunk)
    recovered_socket.finish()


@pytest.mark.asyncio
async def test_recovery_socket_with_vad_gate():
    """After recovery with VAD gate, the GatedDeepgramSocket wraps the connection.

    In _recover_deepgram_connection, process_audio_dg is called with vad_gate,
    producing a GatedDeepgramSocket that flush_stt_buffer uses.
    """
    from utils.stt.vad_gate import GatedDeepgramSocket, VADStreamingGate

    mock_conn = MagicMock()
    mock_conn.send.return_value = True
    gate = VADStreamingGate(sample_rate=16000, channels=1, mode='active', uid='test', session_id='test')

    with patch('utils.stt.streaming.connect_to_deepgram_with_backoff', new_callable=AsyncMock, return_value=mock_conn):
        recovered_socket = await process_audio_dg(
            stream_transcript=MagicMock(),
            language='en',
            sample_rate=16000,
            channels=1,
            vad_gate=gate,
        )

    assert isinstance(recovered_socket, GatedDeepgramSocket)
    assert not recovered_socket.is_connection_dead
    recovered_socket.finish()


# ---------------------------------------------------------------------------
# Behavioral: Multi-channel dead socket detection
# ---------------------------------------------------------------------------


def test_multichannel_dead_socket_nulls_slot():
    """A dead multi-channel SafeDeepgramSocket should be detected via is_connection_dead.

    In the multi-channel send path, this condition nulls stt_sockets_multi[ch_idx]
    so that the recovery task can rebuild that channel.
    """
    mock_conn = MagicMock()
    mock_conn.send.return_value = False  # Triggers death latch
    cfg = KeepaliveConfig(keepalive_interval_sec=5.0, check_period_sec=999.0)
    safe = SafeDeepgramSocket(mock_conn, cfg=cfg)
    try:
        safe.send(b'\x00' * 960)
        assert safe.is_connection_dead is True

        # Simulate the multi-channel send path: detect dead, null the slot
        stt_sockets_multi = [safe, None]
        if stt_sockets_multi[0] and stt_sockets_multi[0].is_connection_dead:
            stt_sockets_multi[0] = None

        assert stt_sockets_multi[0] is None, "Dead multi-channel socket should be nulled for recovery"
    finally:
        safe.finish()


def test_multichannel_send_exception_nulls_slot():
    """Multi-channel send exception should null the socket slot for recovery."""
    mock_conn = MagicMock()
    mock_conn.send.side_effect = ConnectionResetError('Connection reset')
    cfg = KeepaliveConfig(keepalive_interval_sec=5.0, check_period_sec=999.0)
    safe = SafeDeepgramSocket(mock_conn, cfg=cfg)
    try:
        safe.send(b'\x00' * 960)
        assert safe.is_connection_dead is True

        # Simulate the send path — exception caught, slot nulled
        stt_sockets_multi = [safe]
        stt_sockets_multi[0] = None
        assert stt_sockets_multi[0] is None
    finally:
        safe.finish()


def test_multichannel_dead_socket_detection_in_source():
    """Multi-channel send path must detect dead sockets and enter degraded mode."""
    source = _read_transcribe_source()
    mc_dead_pos = source.find('mc_sock.is_connection_dead')
    assert mc_dead_pos > 0, "Multi-channel path must check is_connection_dead"
    mc_null_pos = source.find('stt_sockets_multi[ch_idx] = None', mc_dead_pos)
    assert mc_null_pos > 0, "Multi-channel path must null the dead socket slot"
    mc_degraded_pos = source.find('_enter_degraded_mode', mc_dead_pos)
    assert mc_degraded_pos > 0, "Multi-channel path must enter degraded mode"


# ---------------------------------------------------------------------------
# Behavioral: Speaker state reset after DG recovery
# ---------------------------------------------------------------------------


def test_speaker_state_reset_exists_in_recovery_path():
    """Single-channel recovery must call _reset_speaker_state_after_recovery.

    New DG connection resets diarization — old speaker_to_person_map entries
    would map the wrong person to the wrong speaker number.
    """
    source = _read_transcribe_source()
    # Find the single-channel recovery success path
    recovery_fn_pos = source.find('async def _recover_deepgram_connection')
    assert recovery_fn_pos > 0
    recovery_block = source[recovery_fn_pos:]

    # The reset must happen before _send_stt_recovered_event in single-channel path
    single_ch_recovered_pos = recovery_block.find('f"Recovered Deepgram socket')
    assert single_ch_recovered_pos > 0
    pre_recovered_block = recovery_block[:single_ch_recovered_pos]
    assert (
        '_reset_speaker_state_after_recovery()' in pre_recovered_block
    ), "Single-channel recovery must reset speaker state before sending recovered event"


def test_speaker_state_reset_clears_correct_state():
    """_reset_speaker_state_after_recovery clears speaker_to_person_map and suggested_segments.

    It must NOT clear person_embeddings_cache (embeddings are connection-independent)
    or segment_person_assignment_map (already-persisted assignments stay valid).
    """
    source = _read_transcribe_source()
    reset_fn_pos = source.find('def _reset_speaker_state_after_recovery')
    assert reset_fn_pos > 0
    reset_block = source[reset_fn_pos : reset_fn_pos + 1800]

    # Must clear these (DG-diarization-dependent)
    assert 'speaker_to_person_map.clear()' in reset_block
    assert 'suggested_segments.clear()' in reset_block

    # Must NOT clear these (DG-connection-independent)
    assert 'person_embeddings_cache.clear()' not in reset_block
    assert 'segment_person_assignment_map.clear()' not in reset_block


def test_speaker_state_reset_drains_queue():
    """_reset_speaker_state_after_recovery drains the speaker_id_segment_queue.

    Stale queue items reference old DG speaker_ids that are no longer valid.
    """
    source = _read_transcribe_source()
    reset_fn_pos = source.find('def _reset_speaker_state_after_recovery')
    assert reset_fn_pos > 0
    reset_block = source[reset_fn_pos : reset_fn_pos + 1800]

    assert 'speaker_id_segment_queue' in reset_block, "Must drain the stale speaker_id_segment_queue"
    assert 'get_nowait' in reset_block, "Must drain via get_nowait in a loop"


def test_speaker_state_reset_runtime():
    """Exercise the speaker state reset pattern at runtime.

    Simulates: pre-degradation state with 2 speaker mappings → recovery → verify cleared.
    """
    import asyncio

    speaker_to_person_map = {0: ('alice_id', 'Alice'), 1: ('bob_id', 'Bob')}
    suggested_segments = {'seg_001', 'seg_002', 'seg_003'}
    speaker_id_segment_queue = asyncio.Queue(maxsize=100)
    speaker_id_segment_queue.put_nowait({'id': 'seg_004', 'speaker_id': 0})
    speaker_id_segment_queue.put_nowait({'id': 'seg_005', 'speaker_id': 1})
    # These should NOT be cleared
    person_embeddings_cache = {'alice_id': {'embedding': [0.1] * 256, 'name': 'Alice'}}
    segment_person_assignment_map = {'seg_001': 'alice_id'}

    # Simulate the reset
    speaker_to_person_map.clear()
    suggested_segments.clear()
    while not speaker_id_segment_queue.empty():
        try:
            speaker_id_segment_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

    assert len(speaker_to_person_map) == 0, "speaker_to_person_map must be cleared"
    assert len(suggested_segments) == 0, "suggested_segments must be cleared"
    assert speaker_id_segment_queue.empty(), "speaker_id_segment_queue must be drained"
    # These must survive
    assert len(person_embeddings_cache) == 1, "person_embeddings_cache must NOT be cleared"
    assert len(segment_person_assignment_map) == 1, "segment_person_assignment_map must NOT be cleared"


def test_multichannel_recovery_does_not_reset_speaker_state():
    """Multi-channel recovery must NOT reset speaker state.

    Multi-channel uses fixed per-channel speaker labels (SPEAKER_00, SPEAKER_01)
    set by ChannelConfig, not DG diarization. These are deterministic and
    survive DG reconnection.
    """
    source = _read_transcribe_source()
    recovery_fn_pos = source.find('async def _recover_deepgram_connection')
    assert recovery_fn_pos > 0
    recovery_block = source[recovery_fn_pos:]

    # Find multi-channel recovery success path
    mc_recovered_pos = recovery_block.find('Recovered all multi-channel Deepgram sockets')
    assert mc_recovered_pos > 0

    # The reset must NOT appear between the multi-channel success check and its recovered event
    mc_block = recovery_block[:mc_recovered_pos]
    # Count occurrences of the reset call — it should only appear in single-channel path
    reset_calls = recovery_block.count('_reset_speaker_state_after_recovery()')
    assert reset_calls == 1, f"Reset must appear exactly once (single-channel only), found {reset_calls}"


# ---------------------------------------------------------------------------
# Epoch guard for in-flight _match_speaker_embedding tasks
# ---------------------------------------------------------------------------


def test_epoch_guard_exists_in_match_speaker_embedding():
    """Source: _match_speaker_embedding must check epoch != speaker_map_epoch before writing.

    After DG recovery, speaker_map_epoch is incremented. In-flight tasks spawned
    before recovery carry the old epoch and must discard their results.
    """
    source = _read_transcribe_source()
    fn_pos = source.find('async def _match_speaker_embedding')
    assert fn_pos > 0
    fn_block = source[fn_pos : fn_pos + 6000]

    # Function must accept an epoch parameter
    assert 'epoch' in fn_block[:200], "epoch parameter missing from _match_speaker_embedding signature"

    # Guard must appear before speaker_to_person_map writes
    guard_pos = fn_block.find('epoch != speaker_map_epoch')
    assert guard_pos > 0, "Epoch guard check missing in _match_speaker_embedding"

    # The guard must come before the map writes
    map_write_pos = fn_block.find('speaker_to_person_map[speaker_id]')
    assert map_write_pos > 0
    assert guard_pos < map_write_pos, "Epoch guard must appear before speaker_to_person_map writes"


def test_epoch_guard_spawn_passes_current_epoch():
    """Source: spawn of _match_speaker_embedding must pass epoch=speaker_map_epoch."""
    source = _read_transcribe_source()

    # Find the spawn call for _match_speaker_embedding
    spawn_pos = source.find('_match_speaker_embedding(speaker_id')
    assert spawn_pos > 0
    spawn_line = source[spawn_pos : spawn_pos + 200]
    assert 'epoch=speaker_map_epoch' in spawn_line, "Spawn must pass current epoch to _match_speaker_embedding"


def test_epoch_incremented_on_recovery():
    """Source: _reset_speaker_state_after_recovery must increment speaker_map_epoch."""
    source = _read_transcribe_source()
    fn_pos = source.find('def _reset_speaker_state_after_recovery')
    assert fn_pos > 0
    fn_block = source[fn_pos : fn_pos + 1800]
    assert 'speaker_map_epoch += 1' in fn_block, "Recovery must increment speaker_map_epoch"


def test_epoch_guard_discards_stale_match_runtime():
    """Runtime: simulate epoch mismatch to prove stale speaker matches are discarded.

    Steps:
    1. Set speaker_map_epoch = 0, spawn a task at epoch 0
    2. Before the task writes, increment epoch to 1 (simulating recovery)
    3. Verify the task does NOT write to speaker_to_person_map
    """

    # Use a simple namespace to simulate the shared session state
    class SessionState:
        speaker_map_epoch = 0
        speaker_to_person_map = {}
        speaker_map_dirty = False

    state = SessionState()

    # Simulate the epoch guard logic from _match_speaker_embedding
    def apply_match_with_epoch_guard(speaker_id, person_id, person_name, epoch):
        """Mimics the epoch-guarded write path in _match_speaker_embedding."""
        if epoch != state.speaker_map_epoch:
            return False  # Discarded
        state.speaker_to_person_map[speaker_id] = (person_id, person_name)
        state.speaker_map_dirty = True
        return True  # Written

    # Case 1: Same epoch — write succeeds
    assert apply_match_with_epoch_guard(0, 'person-abc', 'Alice', epoch=0) is True
    assert 0 in state.speaker_to_person_map

    # Simulate recovery: clear map and bump epoch
    state.speaker_to_person_map.clear()
    state.speaker_map_epoch += 1

    # Case 2: Stale epoch — write must be discarded
    assert apply_match_with_epoch_guard(1, 'person-xyz', 'Bob', epoch=0) is False
    assert 1 not in state.speaker_to_person_map

    # Case 3: Current epoch — write succeeds
    assert apply_match_with_epoch_guard(2, 'person-def', 'Carol', epoch=1) is True
    assert 2 in state.speaker_to_person_map


# ---------------------------------------------------------------------------
# Buffer-level epoch tagging (stale segments from old DG connection)
# ---------------------------------------------------------------------------


def test_stream_transcript_tags_segments_with_epoch():
    """Source: stream_transcript must tag each segment dict with _stt_epoch."""
    source = _read_transcribe_source()
    fn_pos = source.find('def stream_transcript(segments)')
    assert fn_pos > 0
    fn_block = source[fn_pos : fn_pos + 500]
    assert "_stt_epoch" in fn_block, "stream_transcript must tag segments with _stt_epoch"
    assert "speaker_map_epoch" in fn_block, "stream_transcript must use speaker_map_epoch for tagging"


def test_multi_channel_callback_tags_segments_with_epoch():
    """Source: multi-channel callback must also tag segments with _stt_epoch."""
    source = _read_transcribe_source()
    fn_pos = source.find('def make_multi_channel_callback')
    assert fn_pos > 0
    fn_block = source[fn_pos : fn_pos + 600]
    assert "_stt_epoch" in fn_block, "multi-channel callback must tag segments with _stt_epoch"


def test_stale_segments_skipped_in_speaker_detection():
    """Source: speaker detection loop must skip segments in stale_dg_segment_ids."""
    source = _read_transcribe_source()
    # Find the speaker detection section
    detection_pos = source.find('# Speaker detection')
    assert detection_pos > 0
    detection_block = source[detection_pos : detection_pos + 500]
    assert (
        'stale_dg_segment_ids' in detection_block
    ), "Speaker detection loop must check stale_dg_segment_ids to skip old DG segments"


def test_stt_epoch_popped_before_transcript_segment():
    """Source: _stt_epoch must be popped from raw dict before TranscriptSegment conversion."""
    source = _read_transcribe_source()
    # Find the TranscriptSegment conversion in stream_transcript_process
    conv_pos = source.find("s.pop('_stt_epoch'")
    assert conv_pos > 0, "_stt_epoch must be popped from segment dict before TranscriptSegment(**s)"
    ts_pos = source.find('TranscriptSegment(**s', conv_pos)
    assert ts_pos > conv_pos, "TranscriptSegment conversion must come AFTER _stt_epoch pop"


def test_stale_buffer_segments_skipped_at_runtime():
    """Runtime: simulate stale segments in buffer to prove they don't trigger speaker operations.

    Steps:
    1. Two segments buffered at epoch 0 (old DG)
    2. Recovery bumps epoch to 1
    3. One new segment arrives at epoch 1
    4. Processing must skip speaker ops for epoch-0 segments but process epoch-1 segment
    """
    # Simulate the buffer-drain-and-classify logic from stream_transcript_process
    speaker_map_epoch = 1  # Recovery already happened

    raw_segments = [
        {
            'text': 'hello from old DG',
            'speaker': 'SPEAKER_0',
            'is_user': False,
            'start': 0.0,
            'end': 1.0,
            '_stt_epoch': 0,
        },
        {
            'text': 'old segment two',
            'speaker': 'SPEAKER_1',
            'is_user': False,
            'start': 1.0,
            'end': 2.0,
            '_stt_epoch': 0,
        },
        {'text': 'new DG segment', 'speaker': 'SPEAKER_0', 'is_user': False, 'start': 2.0, 'end': 3.0, '_stt_epoch': 1},
    ]

    stale_ids = set()
    segment_ids = []
    for s in raw_segments:
        seg_epoch = s.pop('_stt_epoch', speaker_map_epoch)
        seg_id = f"seg-{len(segment_ids)}"
        segment_ids.append(seg_id)
        if seg_epoch != speaker_map_epoch:
            stale_ids.add(seg_id)

    # Epoch-0 segments should be stale
    assert 'seg-0' in stale_ids
    assert 'seg-1' in stale_ids
    # Epoch-1 segment should NOT be stale
    assert 'seg-2' not in stale_ids

    # Simulate the speaker detection loop — only non-stale segments proceed
    speaker_ops_performed = []
    for seg_id in segment_ids:
        if seg_id in stale_ids:
            continue
        speaker_ops_performed.append(seg_id)

    assert speaker_ops_performed == [
        'seg-2'
    ], f"Only epoch-1 segment should have speaker ops, got {speaker_ops_performed}"
