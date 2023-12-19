import asyncio
import csv
import time
from bleak import BleakClient, BleakScanner, discover
from datetime import datetime as dt

# BLE Heart Rate Service UUID (Standard UUID for Heart Rate Monitors)
HRM_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# Global list to store heart rate values
heart_rates = []

# Callback function to handle received data

def callback_handler(sender, data):
    # Decode the heart rate data (assuming the first byte is the heart rate value)
    decoded = [byte for byte in data]
    hr = decoded[1]
    print("Heart Rate:", hr)
    heart_rates.append((dt.now(), hr))

# Function to discover BLE devices and return the address of the desired device

async def scan_for_devices():
    devices = await BleakScanner.discover()
    num_devices = 0
    for device in devices:
        print(f"Device: {device.name}, Address: {device.address}")
        num_devices += 1
    # Add logic here to return the address of your specific device
        if "fenix" in device.__str__():
            address = device.address
    print(f"Found {num_devices} devices")
    try:
        return address
    except:
        print("Device not found")
        return None
    # For example: if device.name == "MyHeartRateMonitor": return device.address

# Function to connect to the device and start notification

async def connect_and_read(address):
    async with BleakClient(address) as client:
        if await client.is_connected():
            print("Connected to device")
            # Start receiving notifications
            await client.start_notify(HRM_UUID, callback_handler)
            # Keep the connection for 60 seconds
            await asyncio.sleep(60)

# Main function to run the BLE heart rate monitor routine

async def main():
    # Replace this with the address of your BLE device
    # Or use the `scan_for_devices` function to find it
    print(f"Scanning for devices...")
    address = await scan_for_devices()
    await connect_and_read(address)

# Run the main function
loop = asyncio.get_event_loop()
loop.run_until_complete(main())

# Saving the collected heart rate data to a CSV file
with open('heart_rate_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Heart Rate"])
    for rate in heart_rates:
        writer.writerow([rate])

print("Data saved to heart_rate_data.csv")
