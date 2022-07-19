import asyncio

import pyarrow as pa
import websockets


def test_read_messages():

    async def hello():
        async with websockets.connect("ws://localhost:8765") as websocket:

            async for message in websocket:

                with pa.ipc.open_stream(message) as reader:
                    df = reader.read_pandas()
                    print("Got rows: {}".format(len(df)))

            print("Exited")

    asyncio.run(hello())
