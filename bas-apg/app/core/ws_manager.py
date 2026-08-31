"""
BAS-APG — WebSocket Connection Manager

Handles multiple browser clients connecting to the live feed.
Uses asyncio for non-blocking broadcast to all connected clients.
Automatically cleans up dead connections.
"""

from fastapi import WebSocket


class WSManager:
    """Manages WebSocket connections for real-time streaming.

    Usage:
        manager = WSManager()
        await manager.connect(websocket)
        await manager.broadcast_json({"type": "update", ...})
        manager.disconnect(websocket)
    """

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WS] Client connected. Total: {self.client_count}")

    def disconnect(self, websocket: WebSocket):
        """Remove a disconnected client."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[WS] Client disconnected. Total: {self.client_count}")

    async def broadcast_json(self, data: dict):
        """Send JSON data to all connected clients.

        Automatically removes clients that fail to receive.
        """
        dead: list[WebSocket] = []
        for conn in self.active_connections:
            try:
                await conn.send_json(data)
            except Exception:
                dead.append(conn)

        for d in dead:
            self.disconnect(d)

    async def broadcast_bytes(self, data: bytes):
        """Send raw bytes to all connected clients (for binary JPEG frames)."""
        dead: list[WebSocket] = []
        for conn in self.active_connections:
            try:
                await conn.send_bytes(data)
            except Exception:
                dead.append(conn)
        for d in dead:
            self.disconnect(d)

    @property
    def client_count(self) -> int:
        return len(self.active_connections)
