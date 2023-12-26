from fastapi import FastAPI, WebSocket
from bluetooth_hr import HeartRateClient, callback_handler

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await read_hr_data()  
        await websocket.send_json(data)

def read_hr_data():
    hr_client = HeartRateClient()
    hr_client.set_adress("B4:C2:6A:A9:A4:04")
    hr_data = hr_client.connect_and_read()
    return hr_data