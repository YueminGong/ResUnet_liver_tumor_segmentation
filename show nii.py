import SimpleITK as sitk
from matplotlib import pyplot as plt


def showNii(img):
    for i in range(img.shape[0]):
        plt.imshow(img[i, :, :], cmap='gray')
        plt.show()
   

itk_img = sitk.ReadImage('D:\wp\Dataset2023.1.2/figture/segmentation-21.nii')
img = sitk.GetArrayFromImage(itk_img)
showNii(img)
