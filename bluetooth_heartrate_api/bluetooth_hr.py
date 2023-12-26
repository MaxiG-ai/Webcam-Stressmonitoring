import time
import csv
import asyncio
from fastAPI import FastAPI, WebSocket
from bleak import BleakClient, BleakScanner

HRM_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

async def callback_handler(sender, data):
    # Process the heart rate data received
    print("Heart Rate: ", int.from_bytes(data, byteorder='little'))

class HeartRateClient():
    def __init__(self):
        pass

    async def show_devices(self):
        devices = await BleakScanner.discover()
        num_devices = 0
        for d in devices:
            if "fenix" in d.__str__():
                print(d)
            else:
                num_devices += 1
        print(f"Found {num_devices} devices")  

    def set_adress(self, address):
        self.address = address
    
    async def connect_to_adress(self):
        async with BleakClient(self.address) as client:
            model_number = await client.read_gatt_char(HRM_UUID)
            print("Model Number: {0}".format("".join(map(chr, model_number))))

    async def connect_and_read(self):
        async with BleakClient(self.address) as client:
            if client.is_connected:
                print("Connected to device")
                await client.start_notify(HRM_UUID, callback_handler)

if __name__ == "__main__":
    hr_client = HeartRateClient()
    asyncio.run(hr_client.show_devices())
    hr_client.set_adress("B4:C2:6A:A9:A4:04")
    # asyncio.run(hr_client.connect_to_adress())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(hr_client.connect_and_read())

    