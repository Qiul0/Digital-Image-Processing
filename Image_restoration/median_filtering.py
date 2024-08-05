import cv2
import matplotlib.pyplot as plt
import numpy as np
import gmpy2


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imgb = cv2.imread('img/Fig0508(a)(circuit-board-pepper-prob-pt1).tif', cv2.IMREAD_GRAYSCALE)
imga = cv2.imread('img/Fig0508(b)(circuit-board-salt-prob-pt1).tif', cv2.IMREAD_GRAYSCALE)
imgc = cv2.imread('img/Fig0512(b)(ckt-uniform-plus-saltpepr-prob-pt1).tif', cv2.IMREAD_GRAYSCALE)


def median(list):
    list = sorted(list)
    n = len(list)
    if n % 2 == 0:
        return (list[int(n/2-1)] + list[int(n/2)]) / 2
    else:
        return list[int(n/2)]

def cut_either_end_average(list, d,kernel_size):
    n = len(list)
    sum = 0
    if n < kernel_size * kernel_size:
        rate = d / (kernel_size * kernel_size)
        d = int(n * rate)
        if d % 2 != 0:
            d = d + 1
    list = sorted(list)
    if d >= 0 and d % 2 == 0 and d < n:
        for i in range(int(d / 2),int(n - d / 2)):
            sum += list[i]
        return int(sum / (n - d))
    else:
        print("error")

# 统计滤波器(中值，最大值，最小值，中点，修正阿尔法)
def statistic_filter(x, y, h, w, img, kernel_size, d):
    left = int(kernel_size / 2)
    right = kernel_size - left
    list = []
    for i in range(x - left, x + right):
        for j in range(y - left, y + right):
            if (i >= 0 and i < h and j >= 0 and j < w):
                list.append(img[i][j])
    return int(median(list)), max(list), min(list),int(cut_either_end_average(list, d,kernel_size))

def median_filtering(img, kernel_size,d):
    h, w = img.shape
    median_filter = np.zeros((h, w), np.uint8)
    max_filter = np.zeros((h, w), np.uint8)
    min_filter = np.zeros((h, w), np.uint8)
    midpoint_filter = np.zeros((h, w), np.uint8)
    correct_alpha_filter = np.zeros((h, w), np.uint8)

    for i in range(h):
        for j in range(w):
            median,max,min,correct_alpha = statistic_filter(i, j, h, w, img, kernel_size, d)
            midpoint = (max + min)/2
            median_filter[i, j] = median
            max_filter[i, j] = max
            min_filter[i, j] = min
            correct_alpha_filter[i, j] = correct_alpha
            midpoint_filter[i, j] = midpoint
    return median_filter, min_filter, max_filter, midpoint_filter,correct_alpha_filter

median_filter, min_filter, max_filter, midpoint_filter,correct_alpha_filter = median_filtering(imga,3,0)
median_filter1,b,max_filter1,d,e = median_filtering(imgb,3,0)

fig,axs=plt.subplots(2,3,constrained_layout=True,dpi=300)
plt.subplot(231), plt.imshow(imga, cmap='gray'), plt.title('Salt')
plt.subplot(232), plt.imshow(median_filter, cmap='gray'),plt.title('Median Filtering')
plt.subplot(233), plt.imshow(min_filter, cmap='gray'),plt.title('Min Filtering')
plt.subplot(234), plt.imshow(imgb, cmap='gray'),plt.title('Pepper')
plt.subplot(236), plt.imshow(median_filter1, cmap='gray'),plt.title('Median Filtering')
plt.subplot(235), plt.imshow(max_filter1, cmap='gray'),plt.title('Max Filtering')
plt.show()


median_filter, min_filter, max_filter, midpoint_filter,correct_alpha_filter = median_filtering(imgc,3,6)
fig,axs=plt.subplots(1,2,constrained_layout=True,dpi=300)
plt.subplot(121), plt.imshow(imgc, cmap='gray'), plt.title('Salt and Pepper')
plt.subplot(122), plt.imshow(correct_alpha_filter, cmap='gray'),plt.title('correct_alpha_filter')
plt.show()
