import numpy as np
import os
from skimage import io, measure, morphology
from mayavi import mlab
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



filepath = 'D:\wp/veseel/2D IMAGE\denoise'
files = os.listdir(filepath)
count = 0
#im3d = np.zeros(shape=(len(files),io.imread(os.path.join(filepath,files[0])).shape[0],
                #io.imread(os.path.join(filepath,files[0])).shape[1]),
                #dtype='uint16')
im3d = np.zeros(shape=(io.imread(os.path.join(filepath,files[0])).shape[0],
                io.imread(os.path.join(filepath,files[0])).shape[1],
                len(files)),
                dtype='uint16')
count=0
for file_ in files:
    im2d = io.imread(os.path.join(filepath,file_),True)
    im3d[:,:,count] = im2d
    #im3d[count] = im2d
    count +=1

print(im3d.shape)
print(im3d.dtype)
verts, faces, _, _ = measure.marching_cubes(im3d, 0)
#cv.imwrite('D:\wp/2D IMAGE/stack.tif',im3d)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
mesh = Poly3DCollection(verts[faces], alpha=0.70)
face_color = [0.45, 0.45, 0.75]
mesh.set_facecolor(face_color)
ax.add_collection3d(mesh)

ax.set_xlim(0, im3d.shape[0])
ax.set_ylim(0, im3d.shape[1])
ax.set_zlim(0, im3d.shape[2])

plt.show()

mlab.show()
