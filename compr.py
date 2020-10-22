import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def compr(A,r):
    u, s, v = np.linalg.svd(A)
    u = u[:,0:r]
    s = np.diag(s)
    s = s[0:r,0:r]
    v = v[0:r,:]
    return (u.dot(s)).dot(v)

img=mpimg.imread('1.jpg')
R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]
r = 500
R = compr(R,r)
G = compr(G,r)
B = compr(B,r)
img1 = np.array(img)
img1[:, :, 0] = R
img1[:, :, 1] = G
img1[:, :, 2] = B
plt.imshow(img1)
plt.show()





