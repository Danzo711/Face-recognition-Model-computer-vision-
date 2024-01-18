import cv2
from pyzbar.pyzbar import decode

with open("names.txt","r") as file:
    names = file.read().split(",")

for name in names:
    image = cv2.imread(filename=f"{name}.png")
    decoded_objects = decode(image)
    
    for obj in decoded_objects:
        qr_data = obj.data.decode('utf-8')
        print(f"QR code data for {name}: {qr_data}")