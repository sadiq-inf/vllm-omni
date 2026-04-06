"""WebSocket handler for streaming text input TTS.

Accepts text incrementally via WebSocket, buffers and splits at sentence
boundaries, and generates audio per sentence using the existing TTS pipeline.

Protocol:
    Client -> Server:
        {"type": "session.config", ...}   # Session config (sent once first)
        {"type": "input.text", "text": "..."} # Text chunks
        {"type": "input.done"}            # End of input

    Server -> Client:
        {"type": "audio.start", "sentence_index": 0, "sentence_text": "...", "format": "wav"}
        <binary frame: audio bytes>
        {"type": "audio.done", "sentence_index": 0}
        {"type": "session.done", "total_sentences": N}
        {"type": "error", "message": "..."}
"""

import asyncio
import json
from contextlib import aclosing

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import (
    OpenAICreateSpeechRequest,
    StreamingSpeechSessionConfig,
)
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.text_splitter import (
    SPLIT_CLAUSE,
    SPLIT_SENTENCE,
    SentenceSplitter,
)

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 30.0  # seconds
_DEFAULT_CONFIG_TIMEOUT = 10.0  # seconds
_PCM_SAMPLE_RATE = 24000
_MAX_CONFIG_MESSAGE_SIZE = 4 * 1024 * 1024  # allow large ref_audio payloads
_MAX_INPUT_TEXT_MESSAGE_SIZE = 128 * 1024


class OmniStreamingSpeechHandler:
    """Handles WebSocket sessions for streaming text-input TTS.

    Each WebSocket connection is an independent session. Text arrives
    incrementally, is split at sentence boundaries, and audio is generated
    per sentence using the existing OmniOpenAIServingSpeech pipeline.

    Args:
        speech_service: The existing TTS serving instance (reused for
            validation and audio generation).
        idle_timeout: Max seconds to wait for a message before closing.
        config_timeout: Max seconds to wait for the initial session.config.
    """

    def __init__(
        self,
        speech_service: OmniOpenAIServingSpeech,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
    ) -> None:
        self._speech_service = speech_service
        self._idle_timeout = idle_timeout
        self._config_timeout = config_timeout

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main session loop for a single WebSocket connection."""
        await websocket.accept()

        try:
            # 1. Wait for session.config
            config = await self._receive_config(websocket)
            if config is None:
                return  # Error already sent, connection closing

            # Validate model if specified
            if config.model and hasattr(self._speech_service, "_check_model"):
                error = await self._speech_service._check_model(
                    OpenAICreateSpeechRequest(input="ping", model=config.model)
                )
                if error is not None:
                    await self._send_error(websocket, str(error))
                    return

            # Branch on stream_mode
            if config.stream_mode == "chunk":
                if config.eager_generation:
                    await self._handle_eager_chunk_mode(websocket, config)
                else:
                    await self._handle_chunk_mode(websocket, config)
            else:
                await self._handle_sentence_mode(websocket, config)

        except WebSocketDisconnect:
            logger.info("Streaming speech: client disconnected")
        except Exception as e:
            logger.exception("Streaming speech session error: %s", e)
            try:
                await self._send_error(websocket, f"Internal error: {e}")
            except Exception:
                logger.debug("Failed to send error to streaming speech client", exc_info=True)

    async def _handle_sentence_mode(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
    ) -> None:
        """Original sentence/clause-level streaming path."""
        boundary_re = SPLIT_CLAUSE if config.split_granularity == "clause" else SPLIT_SENTENCE
        splitter = SentenceSplitter(boundary_re=boundary_re)
        sentence_index = 0

        while True:
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=self._idle_timeout,
                )
            except asyncio.TimeoutError:
                await self._send_error(websocket, "Idle timeout: no message received")
                return

            if len(raw) > _MAX_INPUT_TEXT_MESSAGE_SIZE:
                await self._send_error(websocket, "input.text message too large")
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await self._send_error(websocket, "Invalid JSON message")
                continue

            if not isinstance(msg, dict):
                await self._send_error(websocket, "WebSocket messages must be JSON objects")
                continue

            msg_type = msg.get("type")

            if msg_type == "input.text":
                text = msg.get("text", "")
                if not isinstance(text, str):
                    await self._send_error(websocket, "input.text requires a string value")
                    continue
                sentences = splitter.add_text(text)
                for sentence in sentences:
                    await self._generate_and_send(websocket, config, sentence, sentence_index)
                    sentence_index += 1

            elif msg_type == "input.done":
                remaining = splitter.flush()
                if remaining:
                    await self._generate_and_send(websocket, config, remaining, sentence_index)
                    sentence_index += 1

                await websocket.send_json(
                    {
                        "type": "session.done",
                        "total_sentences": sentence_index,
                    }
                )
                return

            else:
                await self._send_error(
                    websocket,
                    f"Unknown message type: {msg_type}",
                )

    async def _handle_chunk_mode(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
    ) -> None:
        """Chunk-level streaming: bypass sentence splitting for minimal TTFA.

        All input.text messages are accumulated into a single buffer.
        On input.done the full text is submitted as one generation request
        and audio chunks are streamed back as the codec produces them.
        """
        text_buffer: list[str] = []

        # Accumulate text until input.done
        while True:
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=self._idle_timeout,
                )
            except asyncio.TimeoutError:
                await self._send_error(websocket, "Idle timeout: no message received")
                return

            if len(raw) > _MAX_INPUT_TEXT_MESSAGE_SIZE:
                await self._send_error(websocket, "input.text message too large")
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await self._send_error(websocket, "Invalid JSON message")
                continue

            if not isinstance(msg, dict):
                await self._send_error(websocket, "WebSocket messages must be JSON objects")
                continue

            msg_type = msg.get("type")

            if msg_type == "input.text":
                text = msg.get("text", "")
                if not isinstance(text, str):
                    await self._send_error(websocket, "input.text requires a string value")
                    continue
                if text:
                    text_buffer.append(text)

            elif msg_type == "input.done":
                break

            else:
                await self._send_error(
                    websocket,
                    f"Unknown message type: {msg_type}",
                )

        full_text = "".join(text_buffer)
        if not full_text.strip():
            await websocket.send_json({"type": "session.done"})
            return

        # Submit full text as a single streaming generation request
        request = OpenAICreateSpeechRequest(
            input=full_text,
            model=config.model,
            voice=config.voice,
            task_type=config.task_type,
            language=config.language,
            instructions=config.instructions,
            response_format="pcm",
            speed=config.speed,
            max_new_tokens=config.max_new_tokens,
            initial_codec_chunk_frames=config.initial_codec_chunk_frames,
            ref_audio=config.ref_audio,
            ref_text=config.ref_text,
            x_vector_only_mode=config.x_vector_only_mode,
            speaker_embedding=config.speaker_embedding,
            stream=True,
        )

        await websocket.send_json(
            {
                "type": "audio.start",
                "format": "pcm",
                "sample_rate": _PCM_SAMPLE_RATE,
            }
        )

        total_bytes = 0
        generation_failed = False
        request_id = None
        try:
            request_id, generator, _ = await self._speech_service._prepare_speech_generation(request)
            async with aclosing(self._speech_service._generate_audio_chunks(generator, request_id)) as stream:
                async for chunk in stream:
                    total_bytes += len(chunk)
                    await websocket.send_bytes(chunk)
        except WebSocketDisconnect:
            if request_id is not None:
                try:
                    await self._speech_service.engine_client.abort(request_id)
                except Exception:
                    logger.debug("Failed to abort chunk-mode request %s", request_id, exc_info=True)
            raise
        except Exception as e:
            generation_failed = True
            logger.error("Chunk-mode generation failed: %s", e)
            await self._send_error(websocket, f"Generation failed: {e}")
        finally:
            try:
                await websocket.send_json(
                    {
                        "type": "audio.done",
                        "total_bytes": total_bytes,
                        "error": generation_failed,
                    }
                )
            except Exception:
                logger.debug("Failed to send audio.done in chunk mode", exc_info=True)

        await websocket.send_json({"type": "session.done"})

    async def _handle_eager_chunk_mode(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
    ) -> None:
        """Eager chunk streaming: start generation on first input.text.

        Audio generation begins immediately when the first text arrives,
        without waiting for input.done.  This minimises TTFA for
        single-turn use cases where all text is sent in one message.
        Additional input.text messages after the first are ignored for
        generation (they arrive too late — the request is already in flight).
        """
        # Wait for the first input.text
        first_text: str | None = None
        while first_text is None:
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=self._idle_timeout,
                )
            except asyncio.TimeoutError:
                await self._send_error(websocket, "Idle timeout: no message received")
                return

            if len(raw) > _MAX_INPUT_TEXT_MESSAGE_SIZE:
                await self._send_error(websocket, "input.text message too large")
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await self._send_error(websocket, "Invalid JSON message")
                continue

            if not isinstance(msg, dict):
                await self._send_error(websocket, "WebSocket messages must be JSON objects")
                continue

            msg_type = msg.get("type")
            if msg_type == "input.text":
                text = msg.get("text", "")
                if not isinstance(text, str):
                    await self._send_error(websocket, "input.text requires a string value")
                    continue
                if text.strip():
                    first_text = text
            elif msg_type == "input.done":
                await websocket.send_json({"type": "session.done"})
                return
            else:
                await self._send_error(websocket, f"Unknown message type: {msg_type}")

        # Start generation immediately with whatever text we have
        request = OpenAICreateSpeechRequest(
            input=first_text,
            model=config.model,
            voice=config.voice,
            task_type=config.task_type,
            language=config.language,
            instructions=config.instructions,
            response_format="pcm",
            speed=config.speed,
            max_new_tokens=config.max_new_tokens,
            initial_codec_chunk_frames=config.initial_codec_chunk_frames,
            ref_audio=config.ref_audio,
            ref_text=config.ref_text,
            x_vector_only_mode=config.x_vector_only_mode,
            speaker_embedding=config.speaker_embedding,
            stream=True,
        )

        await websocket.send_json(
            {
                "type": "audio.start",
                "format": "pcm",
                "sample_rate": _PCM_SAMPLE_RATE,
            }
        )

        total_bytes = 0
        generation_failed = False
        request_id = None

        # Start generation in a background task so we can drain audio
        # while still consuming remaining WebSocket messages.
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        async def _generate_to_queue():
            nonlocal request_id
            try:
                request_id, generator, _ = await self._speech_service._prepare_speech_generation(request)
                async with aclosing(self._speech_service._generate_audio_chunks(generator, request_id)) as stream:
                    async for chunk in stream:
                        await audio_queue.put(chunk)
            except Exception as exc:
                await audio_queue.put(exc)  # type: ignore[arg-type]
            finally:
                await audio_queue.put(None)  # Sentinel

        gen_task = asyncio.create_task(_generate_to_queue())

        # Drain WebSocket messages (input.text ignored, input.done ends session)
        async def _drain_ws():
            while True:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=self._idle_timeout,
                    )
                except (asyncio.TimeoutError, WebSocketDisconnect):
                    return
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(msg, dict) and msg.get("type") == "input.done":
                    return

        drain_task = asyncio.create_task(_drain_ws())

        try:
            # Stream audio chunks to the client
            while True:
                item = await audio_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    generation_failed = True
                    logger.error("Eager chunk-mode generation failed: %s", item)
                    await self._send_error(websocket, f"Generation failed: {item}")
                    break
                total_bytes += len(item)
                await websocket.send_bytes(item)
        except WebSocketDisconnect:
            if request_id is not None:
                try:
                    await self._speech_service.engine_client.abort(request_id)
                except Exception:
                    logger.debug("Failed to abort eager chunk-mode request %s", request_id, exc_info=True)
            raise
        finally:
            gen_task.cancel()
            drain_task.cancel()
            try:
                await asyncio.gather(gen_task, drain_task, return_exceptions=True)
            except Exception:
                pass
            try:
                await websocket.send_json(
                    {
                        "type": "audio.done",
                        "total_bytes": total_bytes,
                        "error": generation_failed,
                    }
                )
            except Exception:
                logger.debug("Failed to send audio.done in eager chunk mode", exc_info=True)

        await websocket.send_json({"type": "session.done"})

    async def _receive_config(self, websocket: WebSocket) -> StreamingSpeechSessionConfig | None:
        """Wait for and validate the session.config message."""
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=self._config_timeout,
            )
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Timeout waiting for session.config")
            return None

        if len(raw) > _MAX_CONFIG_MESSAGE_SIZE:
            await self._send_error(websocket, "session.config message too large")
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON in session.config")
            return None

        if not isinstance(msg, dict):
            await self._send_error(websocket, "session.config must be a JSON object")
            return None

        if msg.get("type") != "session.config":
            await self._send_error(
                websocket,
                f"Expected session.config, got: {msg.get('type')}",
            )
            return None

        try:
            config = StreamingSpeechSessionConfig(**{k: v for k, v in msg.items() if k != "type"})
        except ValidationError as e:
            await self._send_error(websocket, f"Invalid session config: {e}")
            return None

        return config

    async def _generate_and_send(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
        sentence_text: str,
        sentence_index: int,
    ) -> None:
        """Generate audio for a single sentence and send it over WebSocket."""
        response_format = config.response_format or "wav"

        request = OpenAICreateSpeechRequest(
            input=sentence_text,
            model=config.model,
            voice=config.voice,
            task_type=config.task_type,
            language=config.language,
            instructions=config.instructions,
            response_format=response_format,
            speed=config.speed,
            max_new_tokens=config.max_new_tokens,
            initial_codec_chunk_frames=config.initial_codec_chunk_frames,
            ref_audio=config.ref_audio,
            ref_text=config.ref_text,
            x_vector_only_mode=config.x_vector_only_mode,
            speaker_embedding=config.speaker_embedding,
            stream=config.stream_audio,
        )

        start_payload = {
            "type": "audio.start",
            "sentence_index": sentence_index,
            "sentence_text": sentence_text,
            "format": response_format,
        }
        if config.stream_audio and response_format == "pcm":
            start_payload["sample_rate"] = _PCM_SAMPLE_RATE
        await websocket.send_json(start_payload)

        total_bytes = 0
        generation_failed = False
        request_id = None
        try:
            if config.stream_audio:
                request_id, generator, _ = await self._speech_service._prepare_speech_generation(request)
                async with aclosing(self._speech_service._generate_audio_chunks(generator, request_id)) as stream:
                    async for chunk in stream:
                        total_bytes += len(chunk)
                        await websocket.send_bytes(chunk)
            else:
                audio_bytes, _ = await self._speech_service._generate_audio_bytes(request)
                total_bytes = len(audio_bytes)
                await websocket.send_bytes(audio_bytes)
        except WebSocketDisconnect:
            if request_id is not None:
                try:
                    await self._speech_service.engine_client.abort(request_id)
                except Exception:
                    logger.debug("Failed to abort streaming speech request %s", request_id, exc_info=True)
            raise
        except Exception as e:
            generation_failed = True
            logger.error("Generation failed for sentence %d: %s", sentence_index, e)
            await self._send_error(websocket, f"Generation failed for sentence {sentence_index}: {e}")
        finally:
            try:
                await websocket.send_json(
                    {
                        "type": "audio.done",
                        "sentence_index": sentence_index,
                        "total_bytes": total_bytes,
                        "error": generation_failed,
                    }
                )
            except Exception:
                logger.debug("Failed to send audio.done for sentence %d", sentence_index, exc_info=True)

    @staticmethod
    async def _send_error(websocket: WebSocket, message: str) -> None:
        """Send an error message to the client."""
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": message,
                }
            )
        except Exception:
            pass  # Connection may already be closed; safe to ignore
