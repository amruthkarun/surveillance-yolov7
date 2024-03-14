import tkinter as tk
from PIL import Image, ImageTk
import cv2
import json

rect_coord = {}

object_mapping = {
    "Laptop": 63,
    "Cell Phone": 67,
    "Backpack": 24
}
class ROIApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self.canvas = tk.Canvas(self, width=640, height=640, cursor="cross")
        self.title("Intelligent Surveillance System")
        self.canvas.grid()
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)


        self.clicked = tk.StringVar()
        self.clicked.set("Cell Phone")

        # Create Dropdown menu
        drop = tk.OptionMenu(self, self.clicked, *list(object_mapping.keys()))
        drop.grid(row=1, column=0, pady=10, padx=100)

        button = tk.Button(self, text="Save", command=self.save, height = 5, width = 10)
        button.grid(row=2, column=0, pady=10, padx=100)


        self.rect = None
        self.total_count = 0
	
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None

        self._draw_image()


    def _draw_image(self):
         self.im = Image.open('input_frame.jpg')
         self.tk_im = ImageTk.PhotoImage(self.im)
         self.canvas.create_image(0,0,anchor="nw",image=self.tk_im)


    def save(self):
        object_name = self.clicked.get()
        print(object_name)
        object_of_interest = {object_name: object_mapping[object_name]}
        out_file = open("object_of_interest.json", "w")
        json.dump(object_of_interest, out_file)


    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y

        self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='green')

    def on_move_press(self, event):
        curX, curY = (event.x, event.y)

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)
        self.end_x = curX
        self.end_y = curY
        
    def on_button_release(self, event):
        self.total_count += 1
        rect_coord[self.total_count] = [self.start_x, self.start_y, self.end_x, self.end_y]


def preprocess():
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    #frame = cv2.resize(frame, (640, 640))
    cv2.imwrite("input_frame.jpg", frame)
    print()
    app = ROIApp()
    app.mainloop()
    app.quit()
    return rect_coord
