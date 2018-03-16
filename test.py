# import the necessary packages
from skimage.exposure import rescale_intensity
from sklearn import preprocessing
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from PIL import Image
from resizeimage import resizeimage

np.set_printoptions(threshold=np.nan)


featureMaps = 48
r = 4
d = 32
size = (d-r+1)

def generateRandomWeights():
    matricesBank = []

    for x in range(featureMaps):
        m1 = np.random.randint(100, size=(4, 4))
        matricesBank.append(m1)
    return (matricesBank)

#print(generateRandomWeights())


'''
def orthognalize():
    orthArray = []
    bank = generateRandomWeights()
    for x in range(10):
        tempAr = bank[x]
        U, s, V = np.linalg.svd(tempAr)
        orthArray.append(U)
    return np.asarray(orthArray)

#print(orthognalize())
A = np.random.randint(100, size=(4, 4))
print(A)
U, s, V = np.linalg.svd(A)
print(U)
'''


'''with open('mycar.png', 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [32, 32])
        cover.save('mycar32.png', image.format)
        imageArr = cv2.imread("mycar32.png") #enter the image you would like to read
        iar = np.asarray(imageArr)'''




def SVD():
    m1 = np.random.randint(100, size=(4, 4))
    U, s, V = np.linalg.svd(m1)
    a = np.asarray([[[-0.57773964,  0.58086364,  0.50431857, -0.27290496],
 [-0.57079021, -0.52149295, -0.3422028,  -0.53398585],
 [-0.31149785,  0.49091473, -0.74364389,  0.33009934],
 [-0.49334018, -0.38684011,  0.27487048,  0.72898314]],
[[-0.57773964,  0.58086364,  0.50431857, -0.27290496],
 [-0.57079021, -0.52149295, -0.3422028,  -0.53398585],
 [-0.31149785, 0.49091473 ,-0.74364389 , 0.33009934],
 [-0.49334018,-0.38684011,  0.27487048 , 0.72898314]],
[[-0.57773964,  0.58086364,  0.50431857, -0.27290496],
 [-0.57079021, -0.52149295 ,-0.3422028 , -0.53398585],
 [-0.31149785,  0.49091473, -0.74364389,  0.33009934],
 [-0.49334018, -0.38684011,  0.27487048,  0.72898314]],
[[-0.57773964,  0.58086364,  0.50431857, -0.27290496],
 [-0.57079021, -0.52149295, -0.3422028 , -0.53398585],
 [-0.31149785,  0.49091473, -0.74364389,  0.33009934],
 [-0.49334018, -0.38684011,  0.27487048,  0.72898314]]])
    return U
#print(SVD())

def test():
    print(r)

#test()

def convolve(image, kernel):

    output = np.zeros((size, size), dtype="float32")
    Sum1 = 0.0
    Sum2 = 0.0
    j = 1
    i = 1

    r = 4
    for m in range(r):
        for n in range(r):
            Sum1 = Sum1 + (image[(i + m - 1), (j + n - 1)] * kernel[m][n])
        Sum2 = Sum2 + Sum1
        output[i , j] = Sum2
        i +=1
        j +=1
    return output



image11 = cv2.imread("mycar32.png")
iar=np.asarray(image11)
#print(iar[0][1])
gray = cv2.cvtColor(image11, cv2.COLOR_BGR2GRAY)
#print(gray[1+0-1, 1 + 0 -1]* -0.57773964 )
convoleOutput = convolve(gray, np.asarray(SVD()))
print(np.asarray(convoleOutput))
'''
print("Applying SVD on the image: ")
convoleOutput = convolve(gray, SVD())
cv2.imshow("feature", convoleOutput)
cv2.waitKey(0)
cv2.destroyAllWindows()






# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array((
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]), dtype="int")

# construct the Laplacian kernel used to detect edge-like
# regions of an image
laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")

# construct the Sobel x-axis kernel
sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")

# construct the Sobel y-axis kernel
sobelY = np.array((
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]), dtype="int")

# construct the kernel bank, a list of kernels we're going
# to apply using both our custom `convole` function and
# OpenCV's `filter2D` function
kernelBank = (
	("small_blur", smallBlur),
	("large_blur", largeBlur),
	("sharpen", sharpen),
	("laplacian", laplacian),
	("sobel_x", sobelX),
	("sobel_y", sobelY)
)
'''
