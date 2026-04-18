import asyncio
import websockets
import json

async def test():
    async with websockets.connect("ws://localhost:8000/ws/stream") as ws:
        await ws.send(json.dumps({"scenario_id": 1, "speed": 1000}))
        msg = await ws.recv()
        print("MSG:", msg)

asyncio.run(test())
