import cv2
import matplotlib.pyplot as plt
import numpy as np
import gmpy2


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imga = cv2.imread('img/Fig0513(a)(ckt_gaussian_var_1000_mean_0).tif', cv2.IMREAD_GRAYSCALE)
imgb = cv2.imread('img/Fig0514(a)(ckt_saltpep_prob_pt25).tif', cv2.IMREAD_GRAYSCALE)


def average_variance(x,y,h,w,img,kernel_size):
    left = int(kernel_size / 2)
    right = kernel_size - left
    sum = 0
    v = 0
    for i in range(x - left, x + right):
        for j in range(y - left, y + right):
            if(i>=0 and i<h and j>=0 and j<w):
                sum = sum + img[i][j]
    average =  sum/(kernel_size*kernel_size)
    for i in range(x - left, x + right):
        for j in range(y - left, y + right):
            if(i>=0 and i<h and j>=0 and j<w):
                v = v + pow(img[i][j] - average, 2)
    variance = v/(kernel_size*kernel_size)
    return average,variance
def self_adapt_filtering(img,kernel_size,ver):
    h,w = img.shape
    filter = np.zeros((h,w),np.uint8)
    left = int(kernel_size/2)
    right = kernel_size-left
    for i in range(h):
        for j in range(w):
            average,variance = average_variance(i,j,h,w,img,kernel_size)
            if(variance < ver):
                t = img[i][j] - (variance / ver)*(img[i][j] - average)
                filter[i][j] = t
            if(variance > ver):
                filter[i][j] = average
    return filter

def median(list):
    list = sorted(list)
    n = len(list)
    if n % 2 == 0:
        return (list[int(n/2-1)] + list[int(n/2)]) / 2
    else:
        return list[int(n/2)]


def statistic_filter(x, y, h, w, img, kernel_size):
    left = int(kernel_size / 2)
    right = kernel_size - left
    list = []
    for i in range(x - left, x + right):
        for j in range(y - left, y + right):
            if (i >= 0 and i < h and j >= 0 and j < w):
                list.append(img[i][j])
    return min(list), int(median(list)), max(list)


def model_A(min, median, max,z,S,Smax):
    if min < median < max:
        return model_B(min, median, max,z)
    else:
        #卷积核大小为奇数
        S = S + 2
        if S < Smax:
            return  model_A(min, median, max,z,S,Smax)
        else:
            return median
def model_B(min, median, max,z):
    if min < z < max:
        return z
    else:
        return median

def self_median_filtering(img,kernel_size,Smax):
    h, w = img.shape
    filter = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            min, median, max = statistic_filter(i, j, h, w, img, kernel_size)
            z = img[i][j]
            a = model_A(min,median,max,z,kernel_size,Smax)
            filter[i][j] = a
    return filter

fig, axs = plt.subplots(1, 2, constrained_layout=True, dpi=300)
self_adapt = self_adapt_filtering(imga,7,1000)
plt.subplot(121), plt.imshow(imga, cmap='gray'), plt.title('gaussian_var_1000')
plt.subplot(122), plt.imshow(self_adapt, cmap='gray'),plt.title('self_adapt_filtering')
plt.show()

fig, axs = plt.subplots(1, 2, constrained_layout=True, dpi=300)
self_median = self_median_filtering(imgb,3,7)
plt.subplot(121), plt.imshow(imgb, cmap='gray'), plt.title('saltpep_prob')
plt.subplot(122), plt.imshow(self_median, cmap='gray'),plt.title('self_median_filtering')
plt.show()
