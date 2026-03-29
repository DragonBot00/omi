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
    'utils.stt.soniox_util',
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

sys.modules['deepgram'].DeepgramClient = MagicMock
sys.modules['deepgram'].DeepgramClientOptions = MagicMock
sys.modules['deepgram'].LiveTranscriptionEvents = MagicMock()
sys.modules['deepgram.clients.live.v1'].LiveOptions = MagicMock

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


def test_transcribe_has_deepgram_degraded_branch_before_1011_close():
    source = _read_transcribe_source()
    error_pos = source.find('logger.error(f"Initial processing error: {e} {uid} {session_id}")')
    dg_branch = source.find("if stt_service == STTService.deepgram:", error_pos)
    close_branch = source.find("await websocket.close(code=websocket_close_code)", error_pos)
    assert error_pos > 0
    assert dg_branch > 0
    assert close_branch > 0
    assert dg_branch < close_branch


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
    """The recovery path condition matches the profile-complete condition in flush_stt_buffer.

    Both paths check: vad_gate is not None, mode is active/override, and gate is in shadow.
    """
    source = _read_transcribe_source()

    # Find the recovery path's VAD activation
    recovery_pos = source.find('VAD gate activated after DG recovery')
    assert recovery_pos > 0, "Recovery path must have VAD gate activation"

    # Find the profile-complete path's VAD activation
    profile_pos = source.find('VAD gate activated after speech profile')
    assert profile_pos > 0, "Profile-complete path must have VAD gate activation"

    # Both should use the same condition pattern
    recovery_block = source[recovery_pos - 300 : recovery_pos]
    profile_block = source[profile_pos - 300 : profile_pos]

    assert "vad_gate.mode == 'shadow'" in recovery_block
    assert "vad_gate.mode == 'shadow'" in profile_block
    assert "vad_gate.activate()" in recovery_block
    assert "vad_gate.activate()" in profile_block


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
