import DefFuncAdv as MyFn
from PIL import Image, ImageFilter, ImageOps
import matplotlib.pylab as pylab
from scipy import signal
from skimage.color import rgb2gray
import numpy as np

imIn = Image.open("../images/mandrill.jpg")
imNoise = MyFn.SaltPepperNoise(imIn, 30)

px, py = 2, 3
pylab.figure(figsize=(20,10))
pylab.subplot(px, py, 1), MyFn.plot_image(imIn)
pylab.subplot(px, py, 2), MyFn.plot_image(imNoise)

# Median Filter
i=0
for sz in [3,5,7]:    
    pylab.subplot(px, py, i+4)
    
    # Median Filter
    imFilter = imIn.filter(ImageFilter.MedianFilter(size=sz))
    # Max Filter
    imFilter = imIn.filter(ImageFilter.MaxFilter(size=sz))
    # Min Filter
    imFilter = imIn.filter(ImageFilter.MinFilter(size=sz))
    
    MyFn.plot_image(imFilter, 'Output (Filter size=' +  str(sz) + ')')
       
    i += 1

# Non-Local-Mean
imFilter = MyFn.NonLocalMean(imNoise)
pylab.subplot(px, py, 3)
MyFn.plot_image(imFilter, 'Non Local Means')
   
# Prewitt Operator
imGray = rgb2gray(np.array(imIn))
kernelX = np.array([ [1, 0, -1], [1, 0, -1], [1, 0, -1] ], dtype=np.float64)*(1.0/6.0)
kernelY = np.array([ [1, 1, 1], [0, 0, 0], [-1, -1, -1] ], dtype=np.float64)*(1.0/6.0)
imFilterX = signal.convolve2d(imGray, kernelX, mode='same')
imFilterY = signal.convolve2d(imGray, kernelY, mode='same')
imMagnitude = np.sqrt(imFilterX**2 + imFilterY**2)

pylab.gray()
pylab.subplot(px, py, 4), MyFn.plot_image(imFilterX, 'X-axis')
pylab.subplot(px, py, 5), MyFn.plot_image(imFilterY, 'Y-axis')
pylab.subplot(px, py, 6), MyFn.plot_image(imMagnitude, 'Magnitude')
    
pylab.show()

#%%
import tkinter as tk
import numpy as np
from skimage.io import imread
from PIL import Image, ImageTk

# from matplotlib.figure import Figure
# from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
# import DefFuncWithKernel as myfn
# import matplotlib.pylab as pylab

im1 = imread("../images/messi.jpg") / 255
im2 = imread("../images/ronaldo.jpg") / 255
imc = im1.copy()

w = im1.shape[0]
h = im1.shape[1]
    
root = tk.Tk()
root.title("Morphing")

lFrame = tk.Frame(root, width=w, height=h)
lFrame.grid(row=0, column=0, padx=5, pady=5)

cFrame = tk.Frame(root, width=w, height=h)
cFrame.grid(row=0, column=1, padx=5, pady=5)

rFrame = tk.Frame(root, width=w, height=h)
rFrame.grid(row=0, column=2, padx=5, pady=5)

lImg =  ImageTk.PhotoImage(image=Image.fromarray(np.uint8(im1*255)))
lCanvas = tk.Canvas(lFrame, width=w, height=h)
lCanvasView = lCanvas.create_image(0, 0, anchor="nw", image=lImg)
lCanvas.pack()

# fig = pylab.figure(figsize=(3,3), dpi=100)
# myfn.plot_comp_image(im1, im2)
# lCanvas = FigureCanvasTkAgg(fig, master=lFrame)
# lCanvas.get_tk_widget().pack()

cImg =  ImageTk.PhotoImage(image=Image.fromarray(np.uint8(imc*255)))
cCanvas = tk.Canvas(cFrame, width=w, height=h)
cCanvasView = cCanvas.create_image(0, 0, anchor="nw", image=cImg)
cCanvas.pack()

rImg =  ImageTk.PhotoImage(image=Image.fromarray(np.uint8(im2*255)))
rCanvas = tk.Canvas(rFrame, width=w, height=h)
rCanvasView = rCanvas.create_image(0, 0, anchor="nw", image=rImg)
rCanvas.pack()

def Run(value):
    global cImg
    alpha = float(value) / 100
    imc = (1-alpha)*im1 + alpha*im2
    cImg =  ImageTk.PhotoImage(image=Image.fromarray(np.uint8(imc*255)))
    cCanvas.itemconfig(cCanvasView,image=cImg)  
    
im1=tk.PhotoImage(file="C:\DIP\project", master=lFrame)
imc=tk.PhotoImage(file="C:\DIP\project", master=cFrame)
im2=tk.PhotoImage(file="C:\DIP\project", master=rFrame)

lScale = tk.Scale(cFrame, length = 200, from_=0, to=100, orient=tk.HORIZONTAL, command=Run)
lScale.pack()

root.mainloop()

#%%
import tkinter as tk
import numpy as np
from skimage.io import imread
from PIL import Image, ImageTk

# from matplotlib.figure import Figure
# from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
# import DefFuncWithKernel as myfn
# import matplotlib.pylab as pylab

im1 = imread("../images/messi.jpg") / 255
im2 = imread("../images/ronaldo.jpg") / 255
imc = im1.copy()

w = im1.shape[0]
h = im1.shape[1]
    
root = tk.Tk()
root.title("Morphing")

lFrame = tk.Frame(root, width=w, height=h)
lFrame.grid(row=0, column=0, padx=5, pady=5)

cFrame = tk.Frame(root, width=w, height=h)
cFrame.grid(row=0, column=1, padx=5, pady=5)

rFrame = tk.Frame(root, width=w, height=h)
rFrame.grid(row=0, column=2, padx=5, pady=5)

lImg =  ImageTk.PhotoImage(image=Image.fromarray(np.uint8(im1*255)))
lCanvas = tk.Canvas(lFrame, width=w, height=h)
lCanvasView = lCanvas.create_image(0, 0, anchor="nw", image=lImg)
lCanvas.pack()

# fig = pylab.figure(figsize=(3,3), dpi=100)
# myfn.plot_comp_image(im1, im2)
# lCanvas = FigureCanvasTkAgg(fig, master=lFrame)
# lCanvas.get_tk_widget().pack()

cImg =  ImageTk.PhotoImage(image=Image.fromarray(np.uint8(imc*255)))
cCanvas = tk.Canvas(cFrame, width=w, height=h)
cCanvasView = cCanvas.create_image(0, 0, anchor="nw", image=cImg)
cCanvas.pack()

rImg =  ImageTk.PhotoImage(image=Image.fromarray(np.uint8(im2*255)))
rCanvas = tk.Canvas(rFrame, width=w, height=h)
rCanvasView = rCanvas.create_image(0, 0, anchor="nw", image=rImg)
rCanvas.pack()

def Run(value):
    global cImg
    alpha = float(value) / 100
    imc = (1-alpha)*im1 + alpha*im2
    cImg =  ImageTk.PhotoImage(image=Image.fromarray(np.uint8(imc*255)))
    cCanvas.itemconfig(cCanvasView,image=cImg)   

lScale = tk.Scale(cFrame, length = 200, from_=0, to=100, orient=tk.HORIZONTAL, command=Run)
lScale.pack()

root.mainloop()


