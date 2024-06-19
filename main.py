import numpy as np
import pydicom
from PIL import Image
import os
#format conversion
def convert_function(name):
    ds = pydicom.dcmread('D:\wp\dataset\segmentation-0.nii/'+name)
    image = ds.pixel_array.astype(float) #像素阵列
    scaled_image = (np.maximum(image,0) / image.max() * 255.0) #调整像素
    scaled_image = np.uint8(scaled_image)
    png_image = Image.fromarray(scaled_image)
    #png_image.show()
    return png_image

def get_names(path):
    names = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in ['.dcm']:
                names.append(filename)

    return names


if __name__ == '__main__':
    names = get_names('D:\wp\dataset\segmentation-0.nii')
    for name in names:
        image = convert_function(name)
        print(name)
        image.save('D:\wp\dataset\segmentation-0.nii/'+name+'.png')

