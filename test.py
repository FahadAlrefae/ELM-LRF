# import the necessary packages
from skimage.exposure import rescale_intensity
from sklearn import preprocessing
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from PIL import Image
from resizeimage import resizeimage

np.set_printoptions(threshold=np.nan)


'''
def generateRandomWeights():
    matricesBank = []

    for x in range(10):
        m1 = np.random.randint(100, size=(4, 4))
        matricesBank.append(m1)
    return (matricesBank)

#print(generateRandomWeights())


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


'''featureMaps = 48
r = 4
d = 32
size = (d-r+1)'''

def convolve(image, kernel):
	'''# grab the spatial dimensions of the image, along with
	# the spatial dimensions of the kernel'''
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]

	'''# allocate memory for the output image, taking care to
	# "pad" the borders of the input image so the spatial
	# size (i.e., width and height) are not reduced'''
	pad = int((kW - 1) / 2)
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")
    # loop over the input image "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to
	# bottom'''

	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			'''# extract the ROI of the image by extracting the
			# *center* region of the current (x, y)-coordinates
			# dimensions'''
			roi = image[y - pad:y + pad - 1, x - pad:x + pad - 1]

			'''# perform the actual convolution by taking the
			# element-wise multiplicate between the ROI and
			# the kernel, then summing the matrix'''
			k = (roi * kernel).sum()

			'''# store the convolved value in the output (x,y)-
			# coordinate of the output image'''
			output[y - pad, x - pad] = k
    # rescale the output image to be in the range [0, 255]
	output = rescale_intensity(output, in_range=(-1, 1))
	output = (output * 1).astype("uint8")

	# return the output image
	return output




image11 = cv2.imread("mycar32.png")
gray = cv2.cvtColor(image11, cv2.COLOR_BGR2GRAY)

print("Applying kernal: ")
convoleOutput = convolve(gray, SVD())
cv2.imshow("feature", convoleOutput)
cv2.waitKey(0)
cv2.destroyAllWindows()






'''

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
