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
    return U


''''''

def convolve(image, kernel):

    output = np.zeros((size, size), dtype="float32")
    Sum = 0
    j = 0
    i = 0
    for m in range(size):
        for n in range(size):
            x = image[i] + m - 1
            y = j + n - 1 * kernel[m][n]
            coordinate = (x, y)
            Sum = coordinate.sum()
            output[i][j] = Sum
            i += 1
            j += 1
    return output


image11 = cv2.imread("mycar.png")
gray = cv2.cvtColor(image11, cv2.COLOR_BGR2GRAY)

convoleOutput = convolve(gray, SVD()[0])
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
