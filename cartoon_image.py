#Cartoon Image 
import tkinter as tk
import numpy as np
from skimage.io import imread
from PIL import Image, ImageTk
import cv2

img1 = imread("../images/1.jpg")
imgc = img1.copy()
img2 = img1.copy()

def cartoon_filter(img):
    h, w = img.shape[:2]
    img2 = cv2.resize(img, (w//2, h//2))

    blr = cv2.bilateralFilter(img2, -1, 20, 7)
    edge = 255 - cv2.Canny(img2, 80, 120)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    dstimg = cv2.bitwise_and(blr, edge)
    dstimg = cv2.resize(dstimg, (w, h), interpolation=cv2.INTER_NEAREST)
                                                                  
    return dstimg

img2 = cartoon_filter(img2)

img1 = img1 / 255
img2 = img2 / 255
imgc = imgc / 255

w = img1.shape[0]
h = img1.shape[1]

w = w + 100
    
root = tk.Tk()
root.title("Cartoon Image")

lFrame = tk.Frame(root, width=w, height=h)
lFrame.grid(row=0, column=0, padx=0, pady=0)

cFrame = tk.Frame(root, width=w, height=h)
cFrame.grid(row=0, column=1, padx=0, pady=0)

rFrame = tk.Frame(root, width=w, height=h)
rFrame.grid(row=0, column=2, padx=0, pady=0)

lImg =  ImageTk.PhotoImage(image=Image.fromarray(np.uint8(img1*255)))
lCanvas = tk.Canvas(lFrame, width=w, height=h)
lCanvasView = lCanvas.create_image(0, 0, anchor="nw", image=lImg)
lCanvas.pack()

cImg =  ImageTk.PhotoImage(image=Image.fromarray(np.uint8(imgc*255)))
cCanvas = tk.Canvas(cFrame, width=w, height=h)
cCanvasView = cCanvas.create_image(0, 0, anchor="nw", image=cImg)
cCanvas.pack()

rImg =  ImageTk.PhotoImage(image=Image.fromarray(np.uint8(img2*255)))
rCanvas = tk.Canvas(rFrame, width=w, height=h)
rCanvasView = rCanvas.create_image(0, 0, anchor="nw", image=rImg)
rCanvas.pack()

def Run(value):
    global cImg
    alpha = float(value) / 100
    imgc = (1-alpha)*img1 + alpha*img2
    cImg =  ImageTk.PhotoImage(image=Image.fromarray(np.uint8(imgc*255)))
    cCanvas.itemconfig(cCanvasView,image=cImg)   

lScale = tk.Scale(cFrame, length = 500, from_=0, to=100, orient=tk.HORIZONTAL, command=Run)
lScale.pack()


root.mainloop()

#%% 1??????
import cv2

img1 = imread("../images/1.jpg")
imgc = img1.copy()
img2 = img1.copy()

img2 = cv2.bilateralFilter(img2, -1, 20, 7)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
imgc = cv2.cvtColor(imgc, cv2.COLOR_BGR2RGB)

cv2.imshow('1', img1)
cv2.imshow('c', imgc)
cv2.imshow('2', img2)
cv2.waitKey()

cv2.destroyAllWindows()

#%% 2??????
import cv2

img1 = imread("../images/1.jpg")
imgc = img1.copy()
img2 = img1.copy()

edge = 255 - cv2.Canny(img2, 80, 120)
edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
imgc = cv2.cvtColor(imgc, cv2.COLOR_BGR2RGB)

cv2.imshow('1', img1)
cv2.imshow('c', imgc)
cv2.imshow('2', img2)
cv2.waitKey()

cv2.destroyAllWindows()
