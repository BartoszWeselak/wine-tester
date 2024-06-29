import tkinter as tk
import loader as load
import gui

root = tk.Tk()
root.title("wine tester")
loaded_data=load.read_csv_file()
gui.create_tabs(root, loaded_data)
root.geometry("300x300")

root.mainloop()