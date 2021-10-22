from skimage import io, color
from skimage.viewer import ImageViewer

import numpy as np

def xyz2ill(xyz_tensor):
    input_shape = xyz_tensor.shape
    A = [[27.07439, -22.80783, -1.806681],
        [-5.646736, -7.722125, 12.86503],
        [-4.163133, -4.579428, -4.576049]]
    B = [[0.9465229, 0.2946927, -0.1313419],
        [-0.1179179, 0.9929960, 0.007371554],
        [0.09230461, -0.04645794, 0.9946464]]

    xyz_tensor = np.reshape(xyz_tensor, (input_shape[0] * input_shape[1], input_shape[2])).T
    ill = np.matmul(A, np.log(np.matmul(B, xyz_tensor)))
    ill = np.reshape(ill.T, input_shape)

    return ill


filename = './noisy_1/0.png'
rgb = io.imread(filename)
xyz = color.rgb2xyz(rgb)

ill = xyz2ill(xyz)
print(ill.shape)

viewer = ImageViewer(ill)
viewer.show()