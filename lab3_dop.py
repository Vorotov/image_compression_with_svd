import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
img=mpimg.imread('1.jpg')
RB = np.array(img)
RB[:, :, 1] = 0
plt.imshow(RB)
plt.show()

RB = np.array(img, dtype=int)
RB += -50
plt.imshow(RB)
plt.show()

gray = np.dot(img, [0.2989, 0.5870, 0.1440])
plt.imshow(gray)
plt.gray()
plt.show()

fig, ax = plt.subplots()
RB = np.array(img[:,:,1])
RB = RB.ravel()
ax.hist(RB, bins=256, color="green", range=(0, 256))
plt.show()


def compr(A,r):
    u, s, v = np.linalg.svd(A)
    u = u[:,0:r]
    s = np.diag(s)
    s = s[0:r,0:r]
    v = v[0:r,:]
    return (u.dot(s)).dot(v)

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





