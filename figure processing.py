import imageio
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2 as cv

im = imageio.v2.imread('D:\wp\Dataset2023.1.2/new dataset\patient21/003_tumor.png')
hist = ndi.histogram(im,min=0,max=255,bins=256)
cdf = hist.cumsum()/hist.sum()
print(hist.shape)
print(cdf.shape)
plt.plot(hist)
#9plt.plot(cdf)
plt.show()
im_equalized = cdf[im]*255
hist2 = ndi.histogram(im_equalized,min=0,max=255,bins=256)
plt.plot(hist2)
fig, axes = plt.subplots(2,2)
im_tissue = np.where(im>90,im_equalized,0)
im_tissue2 = np.where((im_equalized>87)&(im>90),im_equalized,0)
m = np.where(im>150,1,0)
im_tissue3 = ndi.binary_erosion(m,iterations=5)
axes[0,0].imshow(im)
axes[0,1].imshow(im_equalized)
axes[1,0].imshow(im_tissue)
axes[1,1].imshow(im_tissue2)
img = im_tissue2

#cv.imwrite('D:\wp\Dataset2023.1.2/new dataset\patient21/014.png',im_tissue2)
plt.show()

