""" Common Function Defintion """

import matplotlib.pylab as pylab
import numpy as np
from skimage import img_as_ubyte, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from PIL import Image

def plot_image(image, title=''):
    """ Plot One Image """
    pylab.title(title, size=20) 
    pylab.imshow(image)
    pylab.axis('off') 
    
def plot_comp_image(image1, image2, title=''):
    pylab.title(title, size=20) 
    pylab.subplot(121), pylab.imshow(image1, cmap='gray'), pylab.axis('off')     
    pylab.subplot(122), pylab.imshow(image2, cmap='gray'), pylab.axis('off')    
    
def plot_hist(r, g, b, title=''):
    r, g, b = img_as_ubyte(r), img_as_ubyte(g), img_as_ubyte(b)
    pylab.hist(np.array(r).ravel(), bins=256, range=(0, 256), color='r', alpha=0.5)
    pylab.hist(np.array(g).ravel(), bins=256, range=(0, 256), color='g', alpha=0.5)
    pylab.hist(np.array(b).ravel(), bins=256, range=(0, 256), color='b', alpha=0.5)
    pylab.xlabel('pixel value', size=20), pylab.ylabel('frequency', size=20)
    pylab.title(title, size=20)    
    
def plot_gray_hist(image, title=''):
    gray = img_as_ubyte(image)
    pylab.hist(np.array(gray).ravel(), bins=256, range=(0, 256), color='gray', alpha=1.0)
    pylab.xlabel('pixel value', size=20), pylab.ylabel('frequency', size=20)
    pylab.title(title, size=20)      

def SaltPepperNoise(im, percent):
    imTmp = im.copy()
    n = int(im.width * im.height * percent / 100)
    x = np.random.randint(0, im.width, n)
    y = np.random.randint(0, im.height, n)
    for (x,y) in zip(x,y):
        px = ( (0, 0, 0) if (np.random.rand() < 0.5) else (255, 255, 255) )
        imTmp.putpixel( (x,y), px)
    return imTmp

def NonLocalMean(im):
    imFloat = img_as_float(im)
    sigmaEst = np.mean(estimate_sigma(imFloat, multichannel=True))
    patchKernel = dict(patch_size=5, # 5x5 patches
                       patch_distance=6, # 13x13 search area
                       multichannel=True)
    denoise = denoise_nl_means(imFloat, h=0.8 * sigmaEst, fast_mode=True, **patchKernel)
    denoiseImg = Image.fromarray(img_as_ubyte(denoise))
    return denoiseImg
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    