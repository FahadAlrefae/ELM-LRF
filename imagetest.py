# import the necessary packages
from skimage.exposure import rescale_intensity
from sklearn import preprocessing
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from PIL import Image
from resizeimage import resizeimage

#np.set_printoptions(threshold=np.nan)

featureMaps = 24
r = 4
d = 32
size = (d-r+1)

def generateRandomWeights():
    matricesBank = []

    for x in range(featureMaps):
        m1 = np.random.randint(100, size=(4, 4))
        matricesBank.append(m1)
    return (matricesBank)

#print(np.asarray(generateRandomWeights()))#prints the 10 random kernels



def orthognalize():
    orthArray = []
    bank = generateRandomWeights()
    for x in range(featureMaps):
        tempAr = bank[x]
        U, s, V = np.linalg.svd(tempAr)
        orthArray.append(U)
    return np.asarray(orthArray)

#print(orthognalize())


def imageArr():
	image = cv2.imread("mycar.png") #enter the image you would like to read
	iar = np.asarray(image)
	return iar
#print(imageArr())

def resize():
    with open('mycar.png', 'r+b') as f:
        with Image.open(f) as image:
            cover = resizeimage.resize_cover(image, [d, d])
            cover.save('mycar32.png', image.format)
            imageArr = cv2.imread("mycar32.png") #enter the image you would like to read
            iar = np.asarray(imageArr)
            return iar

def normalize():
    iar = resize()
    iarmax, iarmin = iar.max(), iar.min()
    iar = (iar - iarmin)/(iarmax - iarmin)
    return iar
#print(normalize(resize())


def convolve(image, kernel):

    output = np.zeros((size, size), dtype="float32")
    Sum1 = 0.0
    Sum2 = 0.0
    j = 1
    i = 1

    r = 4
    for k in range(r):
        for m in range(r):
            for n in range(r):
                Sum1 = Sum1 + (float(image[(i + m - 1), (j + n - 1)]) * kernel[k][m][n])
            Sum2 = Sum2 + Sum1
            output[i , j] = Sum2
            i +=1
            j +=1
    return output





'''
max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(iar[8])
print(X_train_maxabs)  '''
# doctest +NORMALIZE_WHITESPACE^

#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# loop over the kernels
'''for (kernelName, kernel) in kernelBank:
	# apply the kernel to the grayscale image using both
	# our custom `convole` function and OpenCV's `filter2D`
	# function
	print("[INFO] applying {} kernel".format(kernelName))
	convoleOutput = convolve(gray, kernel)
	#opencvOutput = cv2.filter2D(gray, -1, kernel)

	# show the output images
	cv2.imshow("original", gray)
	cv2.imshow("{} - convole".format(kernelName), convoleOutput)
	#cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
'''
'''
print("Applying random weights")
convoleOutput = convolve(gray, sobelX)

cv2.imshow("sobel_x", convoleOutput)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


'''



print iar
plt.imshow(iar)
plt.show()









i = Image.open('dot.png')
iar = np.asarray(i)


print iar.shape

i2 = Image.open('numbers/y0.4.png')
iar2 = np.asarray(i2)


i3 = Image.open('numbers/y0.5.png')
iar3 = np.asarray(i3)


i4 = Image.open('sentdex.png')
iar4 = np.asarray(i4)



#threshold(iar3)


fig = plt.figure()
ax1 = plt.subplot2grid((2,2), (0,0))
ax2 = plt.subplot2grid((2,2), (0,1))
ax3 = plt.subplot2grid((2,2), (1,0))
ax4 = plt.subplot2grid((2,2), (1,1))

ax1.imshow(iar)
ax2.imshow(iar2)
ax3.imshow(iar3)
ax4.imshow(iar4)
plt.show()

'''
