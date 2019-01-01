import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.misc as misc
from scipy.ndimage import distance_transform_cdt

def get_hand_center(img):
    img = distance_transform_cdt(img)
    center = np.unravel_index(np.argmax(img), img.shape)
    return center

basename = "Data/Boundaries/G3_P1_6"

pts = sio.loadmat("{}.mat".format(basename))
img = misc.imread("{}.bmp".format(basename))

center = get_hand_center(img)

center = np.unravel_index(np.argmax(img), img.shape)
plt.imshow(img)
plt.scatter([center[1]], [center[0]])
plt.show()