"""WebSocket endpoint: /ws/v1/session/{session_id}?token=<access_token>

Close codes:
    4401  missing, invalid or expired token, or the token is for another session
    4403  Origin header not allowed
    4408  nothing received for `idle_timeout_s` (clients send HEARTBEAT every 15 s)
    4429  too many connections for this user

The token is only ever read from the query string; it is never logged.
"""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, WebSocket

from app.services.session_manager import SessionManager

logger = logging.getLogger("app.ws")

router = APIRouter()

CLOSE_UNAUTHORIZED = 4401
CLOSE_FORBIDDEN_ORIGIN = 4403
CLOSE_IDLE_TIMEOUT = 4408
CLOSE_TOO_MANY = 4429


async def serve(
    websocket: Any,
    session_id: str,
    token: str | None,
    manager: SessionManager,
) -> None:
    """Run one WebSocket connection until it ends.

    `websocket` needs only accept(), receive(), send_text(), close() and `headers`, so
    tests can use a fake socket.
    """
    await websocket.accept()

    if not manager.origin_allowed(websocket.headers.get("origin")):
        await websocket.close(code=CLOSE_FORBIDDEN_ORIGIN)
        return
    principal = await manager.backend.authenticate(session_id, token)
    if principal is None:
        await websocket.close(code=CLOSE_UNAUTHORIZED)
        return
    if not manager.try_reserve(principal.user_id):
        await websocket.close(code=CLOSE_TOO_MANY)
        return

    session = manager.create_session(principal, websocket.send_text)
    try:
        await session.send_sync()
        while not session.closed:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(), manager.idle_timeout_s
                )
            except TimeoutError:
                await websocket.close(code=CLOSE_IDLE_TIMEOUT)
                break
            if message["type"] == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data is not None:
                session.on_audio_frame(data)
                continue
            text = message.get("text")
            if text is not None:
                await session.on_text(text)
    except Exception:
        logger.exception("WebSocket session %s crashed", session_id)
    finally:
        await session.close()
        manager.release(principal.user_id)


@router.websocket("/ws/v1/session/{session_id}")
async def session_socket(
    websocket: WebSocket, session_id: str, token: str | None = None
) -> None:
    manager: SessionManager = websocket.app.state.session_manager
    await serve(websocket, session_id, token, manager)
