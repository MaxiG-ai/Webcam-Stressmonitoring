import time
import csv
from bleak import BleakClient
import asyncio
from bleak import BleakScanner

# discover devices
async def discover():
    devices = await BleakScanner.discover()

    num_devices = 0
    for d in devices:
        if "fenix" in d.__str__():
            print(d)
        else:
            num_devices += 1
    
    print(f"Found {num_devices} devices")

asyncio.run(discover())


address = "B4:C2:6A:A9:A4:04"
HRM_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# connect to device
async def main(address):
    async with BleakClient(address) as client:
        model_number = await client.read_gatt_char(HRM_UUID)
        print("Model Number: {0}".format("".join(map(chr, model_number))))

asyncio.run(main(address))

# connect and read data

async def connect_and_read(address):
    async with BleakClient(address) as client:
        if await client.is_connected():
            print("Connected to device")
            await client.start_notify(HRM_UUID, callback_handler)


async def callback_handler(sender, data):
    # Process the heart rate data received
    print("Heart Rate:", int.from_bytes(data, byteorder='little'))

loop = asyncio.get_event_loop()
loop.run_until_complete(connect_and_read(address))

# write data to file
heart_rates = []

# def callback_handler(sender, data):
#     heart_rate = int.from_bytes(data, byteorder='little')
#     heart_rates.append(heart_rate)
#     print("Heart Rate:", heart_rate)

# Run for 1 minute
start_time = time.time()
while time.time() - start_time < 60:
    loop.run_until_complete(connect_and_read(address))

# Save to CSV
with open('heart_rate_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Heart Rate"])
    for rate in heart_rates:
        writer.writerow([rate])
