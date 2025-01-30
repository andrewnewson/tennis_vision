import tkinter as tk
from tkinter import filedialog

# Function for selecting an input video file
def select_file():
    # Create a hidden Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    # Ensure the window updates to prevent freezing in VSCode
    root.update()

    # Open file dialog
    file_path = filedialog.askopenfilename(title="Select input video file")
    
    # Destroy the root window after use (fixes lingering GUI issues)
    root.destroy()

    # Print or return the selected file path
    if file_path:
        print(f"Selected file: {file_path}")
        return file_path
    else:
        print("No file selected.")
        return None
