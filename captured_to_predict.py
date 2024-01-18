import cv2
import tkinter as tk
from tkinter import Button, Label, messagebox
from PIL import Image, ImageTk
import os

class CameraCaptureApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0  # Use the default camera (change if necessary)
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=self.vid.get(3), height=self.vid.get(4))
        self.canvas.pack()

        self.btn_capture = Button(window, text="Capture", width=10, command=self.capture)
        self.btn_capture.pack(padx=10, pady=10)

        self.capture_count = 0
        self.capture_limit = 1
        self.output_folder = os.path.dirname(os.path.abspath(__file__))
        self.image_path = os.path.join(self.output_folder, "captured_image.jpg")  # Image path

        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def capture(self):
        if self.capture_count < self.capture_limit:
            ret, frame = self.vid.read()
            if ret:
                self.capture_count += 1
                # Delete the old image if it exists
                if os.path.exists(self.image_path):
                    os.remove(self.image_path)
                cv2.imwrite(self.image_path, frame)
                self.show_capture_message()
            else:
                self.show_error_message("Capture Error", "Failed to capture image.")
        else:
            self.show_info_message("Capture Complete", "Image captured. The application will now exit.")
            self.window.quit()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(10, self.update)

    def on_closing(self):
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()

    def show_capture_message(self):
        self.show_info_message("Capture Successful", f"Image saved as captured_image.jpg")

    def show_info_message(self, title, message):
        messagebox.showinfo(title, message)

    def show_error_message(self, title, message):
        messagebox.showerror(title, message)

# Create a window and pass it to the CameraCaptureApp class
root = tk.Tk()
app = CameraCaptureApp(root, "Camera Capture App")
