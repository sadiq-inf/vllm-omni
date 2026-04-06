import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, WebSocket
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vllm_omni.entrypoints.openai import serving_speech_stream as streaming_speech_module
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.serving_speech_stream import OmniStreamingSpeechHandler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _build_test_app(speech_service=None, *, idle_timeout=30.0, config_timeout=10.0):
    if speech_service is None:
        speech_service = MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = AsyncMock(return_value=(b"RIFF" + b"\x00" * 32, "audio/wav"))
        speech_service._prepare_speech_generation = AsyncMock(return_value=("req-1", object(), {}))

        async def mock_generate_audio_chunks(_generator, _request_id):
            for chunk in (b"\x01\x02", b"\x03\x04\x05"):
                yield chunk

        speech_service._generate_audio_chunks = mock_generate_audio_chunks
        speech_service.engine_client = MagicMock()
        speech_service.engine_client.abort = AsyncMock()

    handler = OmniStreamingSpeechHandler(
        speech_service=speech_service,
        idle_timeout=idle_timeout,
        config_timeout=config_timeout,
    )
    app = FastAPI()

    @app.websocket("/v1/audio/speech/stream")
    async def ws_endpoint(websocket: WebSocket):
        await handler.handle_session(websocket)

    return app, speech_service


class TestStreamingSpeechWebSocket:
    def test_non_streaming_single_frame(self):
        app, speech_service = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "Hello world. "})

                start = ws.receive_json()
                assert start["type"] == "audio.start"
                assert start["sentence_index"] == 0
                assert start["sentence_text"] == "Hello world."
                assert start["format"] == "wav"

                audio = ws.receive_bytes()
                assert audio.startswith(b"RIFF")

                done = ws.receive_json()
                assert done == {"type": "audio.done", "sentence_index": 0, "total_bytes": len(audio), "error": False}

                ws.send_json({"type": "input.done"})
                session_done = ws.receive_json()
                assert session_done == {"type": "session.done", "total_sentences": 1}

        assert speech_service._generate_audio_bytes.await_count == 1

    def test_streaming_multiple_binary_frames(self):
        captured_requests = []

        speech_service = MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = AsyncMock(return_value=(b"", "audio/wav"))
        speech_service.engine_client = MagicMock()
        speech_service.engine_client.abort = AsyncMock()

        async def mock_prepare_speech_generation(request):
            captured_requests.append(request)
            return "req-stream", object(), {}

        speech_service._prepare_speech_generation = mock_prepare_speech_generation

        async def mock_generate_audio_chunks(_generator, _request_id):
            for chunk in (b"\x01\x02", b"\x03\x04\x05", b"\x06"):
                yield chunk

        speech_service._generate_audio_chunks = mock_generate_audio_chunks
        app, _ = _build_test_app(speech_service)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "stream_audio": True,
                        "response_format": "pcm",
                        "initial_codec_chunk_frames": 12,
                    }
                )
                ws.send_json({"type": "input.text", "text": "Hello world. "})

                start = ws.receive_json()
                assert start["type"] == "audio.start"
                assert start["format"] == "pcm"
                assert start["sample_rate"] == 24000

                assert ws.receive_bytes() == b"\x01\x02"
                assert ws.receive_bytes() == b"\x03\x04\x05"
                assert ws.receive_bytes() == b"\x06"

                done = ws.receive_json()
                assert done == {"type": "audio.done", "sentence_index": 0, "total_bytes": 6, "error": False}

                ws.send_json({"type": "input.done"})
                assert ws.receive_json() == {"type": "session.done", "total_sentences": 1}

        assert len(captured_requests) == 1
        assert captured_requests[0].stream is True
        assert captured_requests[0].response_format == "pcm"
        assert captured_requests[0].initial_codec_chunk_frames == 12
        assert speech_service._generate_audio_bytes.await_count == 0

    def test_flush_on_input_done(self):
        app, _ = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "Hello world without punctuation"})
                ws.send_json({"type": "input.done"})

                assert ws.receive_json()["type"] == "audio.start"
                assert ws.receive_bytes()
                assert ws.receive_json() == {
                    "type": "audio.done",
                    "sentence_index": 0,
                    "total_bytes": 36,
                    "error": False,
                }
                assert ws.receive_json() == {"type": "session.done", "total_sentences": 1}

    def test_invalid_streaming_config(self):
        app, _ = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "stream_audio": True,
                        "response_format": "wav",
                    }
                )
                error = ws.receive_json()
                assert error["type"] == "error"
                assert "response_format='pcm'" in error["message"]

    def test_empty_input_text_emits_no_audio(self):
        app, speech_service = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": ""})
                ws.send_json({"type": "input.done"})

                assert ws.receive_json() == {"type": "session.done", "total_sentences": 0}

        assert speech_service._generate_audio_bytes.await_count == 0

    def test_multiple_sentences_increment_indices(self):
        app, _ = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "First sentence. Second sentence. "})

                first_start = ws.receive_json()
                assert first_start["sentence_index"] == 0
                ws.receive_bytes()
                assert ws.receive_json() == {
                    "type": "audio.done",
                    "sentence_index": 0,
                    "total_bytes": 36,
                    "error": False,
                }

                second_start = ws.receive_json()
                assert second_start["sentence_index"] == 1
                ws.receive_bytes()
                assert ws.receive_json() == {
                    "type": "audio.done",
                    "sentence_index": 1,
                    "total_bytes": 36,
                    "error": False,
                }

                ws.send_json({"type": "input.done"})
                assert ws.receive_json() == {"type": "session.done", "total_sentences": 2}

    def test_unknown_message_type_keeps_session_open(self):
        app, _ = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "unknown"})

                error = ws.receive_json()
                assert error == {"type": "error", "message": "Unknown message type: unknown"}

                ws.send_json({"type": "input.text", "text": "Hello world. "})
                assert ws.receive_json()["type"] == "audio.start"
                ws.receive_bytes()
                assert ws.receive_json() == {
                    "type": "audio.done",
                    "sentence_index": 0,
                    "total_bytes": 36,
                    "error": False,
                }

                ws.send_json({"type": "input.done"})
                assert ws.receive_json() == {"type": "session.done", "total_sentences": 1}

    def test_config_timeout_closes_session(self):
        app, _ = _build_test_app(config_timeout=0.01)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                error = ws.receive_json()
                assert error == {"type": "error", "message": "Timeout waiting for session.config"}

    def test_generation_error_marks_audio_done(self):
        speech_service = MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = AsyncMock(side_effect=RuntimeError("boom"))
        speech_service._prepare_speech_generation = AsyncMock(return_value=("req-err", object(), {}))
        speech_service._generate_audio_chunks = AsyncMock()
        speech_service.engine_client = MagicMock()
        speech_service.engine_client.abort = AsyncMock()
        app, _ = _build_test_app(speech_service)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "Hello world. "})

                assert ws.receive_json()["type"] == "audio.start"
                assert ws.receive_json() == {"type": "error", "message": "Generation failed for sentence 0: boom"}
                assert ws.receive_json() == {"type": "audio.done", "sentence_index": 0, "total_bytes": 0, "error": True}

                ws.send_json({"type": "input.done"})
                assert ws.receive_json() == {"type": "session.done", "total_sentences": 1}

    def test_streaming_generation_error_marks_audio_done(self):
        speech_service = MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = AsyncMock(return_value=(b"", "audio/wav"))
        speech_service._prepare_speech_generation = AsyncMock(return_value=("req-stream-err", object(), {}))
        speech_service.engine_client = MagicMock()
        speech_service.engine_client.abort = AsyncMock()

        async def mock_generate_audio_chunks(_generator, _request_id):
            yield b"\x01\x02"
            raise RuntimeError("stream boom")

        speech_service._generate_audio_chunks = mock_generate_audio_chunks
        app, _ = _build_test_app(speech_service)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "stream_audio": True,
                        "response_format": "pcm",
                    }
                )
                ws.send_json({"type": "input.text", "text": "Hello world. "})

                assert ws.receive_json()["type"] == "audio.start"
                assert ws.receive_bytes() == b"\x01\x02"
                assert ws.receive_json() == {
                    "type": "error",
                    "message": "Generation failed for sentence 0: stream boom",
                }
                assert ws.receive_json() == {
                    "type": "audio.done",
                    "sentence_index": 0,
                    "total_bytes": 2,
                    "error": True,
                }

                ws.send_json({"type": "input.done"})
                assert ws.receive_json() == {"type": "session.done", "total_sentences": 1}

    def test_invalid_input_text_type_returns_validation_error(self):
        app, speech_service = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": 123})

                assert ws.receive_json() == {
                    "type": "error",
                    "message": "input.text requires a string value",
                }

                ws.send_json({"type": "input.done"})
                assert ws.receive_json() == {"type": "session.done", "total_sentences": 0}

        assert speech_service._generate_audio_bytes.await_count == 0

    def test_input_text_message_too_large(self, monkeypatch):
        monkeypatch.setattr(streaming_speech_module, "_MAX_INPUT_TEXT_MESSAGE_SIZE", 32)
        app, speech_service = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "x" * 128})

                assert ws.receive_json() == {
                    "type": "error",
                    "message": "input.text message too large",
                }

                ws.send_json({"type": "input.done"})
                assert ws.receive_json() == {"type": "session.done", "total_sentences": 0}

        assert speech_service._generate_audio_bytes.await_count == 0

    def test_session_config_message_too_large(self, monkeypatch):
        monkeypatch.setattr(streaming_speech_module, "_MAX_CONFIG_MESSAGE_SIZE", 64)
        app, _ = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian", "ref_audio": "x" * 512})

                assert ws.receive_json() == {
                    "type": "error",
                    "message": "session.config message too large",
                }

    def test_disconnect_aborts_streaming_request(self):
        speech_service = MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = AsyncMock(return_value=(b"", "audio/wav"))
        speech_service._prepare_speech_generation = AsyncMock(return_value=("req-abort", object(), {}))
        speech_service.engine_client = MagicMock()
        speech_service.engine_client.abort = AsyncMock()

        async def mock_generate_audio_chunks(_generator, _request_id):
            yield b"\x01\x02"

        speech_service._generate_audio_chunks = mock_generate_audio_chunks
        handler = OmniStreamingSpeechHandler(speech_service=speech_service)

        websocket = MagicMock()
        websocket.send_json = AsyncMock(side_effect=[None, WebSocketDisconnect()])
        websocket.send_bytes = AsyncMock(side_effect=WebSocketDisconnect())

        config = MagicMock()
        config.model = None
        config.voice = "Vivian"
        config.task_type = None
        config.language = None
        config.instructions = None
        config.response_format = "pcm"
        config.speed = 1.0
        config.max_new_tokens = None
        config.initial_codec_chunk_frames = None
        config.ref_audio = None
        config.ref_text = None
        config.x_vector_only_mode = None
        config.stream_audio = True

        with pytest.raises(WebSocketDisconnect):
            asyncio.run(handler._generate_and_send(websocket, config, "Hello world.", 0))

        speech_service.engine_client.abort.assert_awaited_once_with("req-abort")
        assert websocket.send_json.await_count == 2


class TestChunkModeStreaming:
    """Tests for stream_mode='chunk' which bypasses sentence splitting."""

    def test_chunk_mode_streams_without_sentence_splitting(self):
        """Chunk mode sends all text as a single generation request."""
        captured_requests = []

        speech_service = MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service.engine_client = MagicMock()
        speech_service.engine_client.abort = AsyncMock()

        async def mock_prepare(request):
            captured_requests.append(request)
            return "req-chunk", object(), {}

        speech_service._prepare_speech_generation = mock_prepare

        async def mock_generate_audio_chunks(_generator, _request_id):
            for chunk in (b"\x01\x02", b"\x03\x04\x05"):
                yield chunk

        speech_service._generate_audio_chunks = mock_generate_audio_chunks
        app, _ = _build_test_app(speech_service)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "stream_mode": "chunk",
                        "stream_audio": True,
                        "response_format": "pcm",
                    }
                )
                # Send multiple text chunks — they should be concatenated, not split
                ws.send_json({"type": "input.text", "text": "First sentence. "})
                ws.send_json({"type": "input.text", "text": "Second sentence. "})
                ws.send_json({"type": "input.done"})

                start = ws.receive_json()
                assert start["type"] == "audio.start"
                assert start["format"] == "pcm"
                assert start["sample_rate"] == 24000
                assert "sentence_index" not in start

                assert ws.receive_bytes() == b"\x01\x02"
                assert ws.receive_bytes() == b"\x03\x04\x05"

                done = ws.receive_json()
                assert done["type"] == "audio.done"
                assert done["total_bytes"] == 5
                assert done["error"] is False
                assert "sentence_index" not in done

                session_done = ws.receive_json()
                assert session_done == {"type": "session.done"}

        # Only one generation request for the full concatenated text
        assert len(captured_requests) == 1
        assert captured_requests[0].input == "First sentence. Second sentence. "
        assert captured_requests[0].stream is True

    def test_chunk_mode_requires_stream_audio(self):
        """stream_mode='chunk' without stream_audio=True should be rejected."""
        app, _ = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "stream_mode": "chunk",
                        "stream_audio": False,
                        "response_format": "pcm",
                    }
                )
                error = ws.receive_json()
                assert error["type"] == "error"
                assert "stream_audio" in error["message"]

    def test_chunk_mode_empty_text(self):
        """Chunk mode with no text should send session.done without generation."""
        app, speech_service = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "stream_mode": "chunk",
                        "stream_audio": True,
                        "response_format": "pcm",
                    }
                )
                ws.send_json({"type": "input.done"})

                session_done = ws.receive_json()
                assert session_done == {"type": "session.done"}

    def test_chunk_mode_generation_error(self):
        """Chunk mode should report generation errors properly."""
        speech_service = MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._prepare_speech_generation = AsyncMock(
            return_value=("req-chunk-err", object(), {})
        )
        speech_service.engine_client = MagicMock()
        speech_service.engine_client.abort = AsyncMock()

        async def mock_generate_audio_chunks(_generator, _request_id):
            yield b"\x01\x02"
            raise RuntimeError("chunk boom")

        speech_service._generate_audio_chunks = mock_generate_audio_chunks
        app, _ = _build_test_app(speech_service)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "stream_mode": "chunk",
                        "stream_audio": True,
                        "response_format": "pcm",
                    }
                )
                ws.send_json({"type": "input.text", "text": "Hello world"})
                ws.send_json({"type": "input.done"})

                assert ws.receive_json()["type"] == "audio.start"
                assert ws.receive_bytes() == b"\x01\x02"
                error = ws.receive_json()
                assert error["type"] == "error"
                assert "chunk boom" in error["message"]

                done = ws.receive_json()
                assert done["type"] == "audio.done"
                assert done["total_bytes"] == 2
                assert done["error"] is True

                assert ws.receive_json() == {"type": "session.done"}

    def test_sentence_mode_unchanged_with_stream_mode_param(self):
        """Existing sentence mode works when stream_mode is explicitly 'sentence'."""
        app, speech_service = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "stream_mode": "sentence",
                    }
                )
                ws.send_json({"type": "input.text", "text": "Hello world. "})

                start = ws.receive_json()
                assert start["type"] == "audio.start"
                assert start["sentence_index"] == 0
                assert start["sentence_text"] == "Hello world."

                ws.receive_bytes()
                assert ws.receive_json()["type"] == "audio.done"

                ws.send_json({"type": "input.done"})
                assert ws.receive_json() == {"type": "session.done", "total_sentences": 1}

        assert speech_service._generate_audio_bytes.await_count == 1


class TestEagerChunkModeStreaming:
    """Tests for eager_generation=True with stream_mode='chunk'."""

    def test_eager_mode_starts_audio_before_input_done(self):
        """Audio should start flowing after first input.text, before input.done."""
        captured_requests = []

        speech_service = MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service.engine_client = MagicMock()
        speech_service.engine_client.abort = AsyncMock()

        async def mock_prepare(request):
            captured_requests.append(request)
            return "req-eager", object(), {}

        speech_service._prepare_speech_generation = mock_prepare

        async def mock_generate_audio_chunks(_generator, _request_id):
            for chunk in (b"\xAA\xBB", b"\xCC\xDD"):
                yield chunk

        speech_service._generate_audio_chunks = mock_generate_audio_chunks
        app, _ = _build_test_app(speech_service)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "stream_mode": "chunk",
                        "eager_generation": True,
                        "stream_audio": True,
                        "response_format": "pcm",
                    }
                )
                # Send first text — generation starts immediately
                ws.send_json({"type": "input.text", "text": "Hello eager world"})

                start = ws.receive_json()
                assert start["type"] == "audio.start"
                assert start["format"] == "pcm"

                assert ws.receive_bytes() == b"\xAA\xBB"
                assert ws.receive_bytes() == b"\xCC\xDD"

                done = ws.receive_json()
                assert done["type"] == "audio.done"
                assert done["total_bytes"] == 4
                assert done["error"] is False

                # Now send input.done to cleanly close
                ws.send_json({"type": "input.done"})

                session_done = ws.receive_json()
                assert session_done == {"type": "session.done"}

        assert len(captured_requests) == 1
        assert captured_requests[0].input == "Hello eager world"

    def test_eager_mode_requires_chunk_stream_mode(self):
        """eager_generation=True without stream_mode='chunk' should fail."""
        app, _ = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "eager_generation": True,
                        "stream_mode": "sentence",
                        "stream_audio": True,
                        "response_format": "pcm",
                    }
                )
                error = ws.receive_json()
                assert error["type"] == "error"
                assert "stream_mode" in error["message"]

    def test_eager_mode_empty_then_done(self):
        """Sending input.done without text should end cleanly."""
        app, _ = _build_test_app()

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "stream_mode": "chunk",
                        "eager_generation": True,
                        "stream_audio": True,
                        "response_format": "pcm",
                    }
                )
                ws.send_json({"type": "input.done"})
                assert ws.receive_json() == {"type": "session.done"}
