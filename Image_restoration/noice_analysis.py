import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imga = cv2.imread('img/Fig0504(a)(gaussian-noise).tif', cv2.IMREAD_GRAYSCALE)
imgb = cv2.imread('img/Fig0504(b)(rayleigh-noise).tif', cv2.IMREAD_GRAYSCALE)
imgc = cv2.imread('img/Fig0504(c)(gamma-noise).tif', cv2.IMREAD_GRAYSCALE)
imgg = cv2.imread('img/Fig0504(g)(neg-exp-noise).tif', cv2.IMREAD_GRAYSCALE)
imgh = cv2.imread('img/Fig0504(h)(uniform-noise).tif', cv2.IMREAD_GRAYSCALE)
imgi = cv2.imread('img/Fig0504(i)(salt-pepper-noise).tif', cv2.IMREAD_GRAYSCALE)



def Grayscale_histogram(img):

    # 统计各灰度数量
    histogram_arr = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            histogram_arr[img[i, j]] += 1

    # 找到最大值
    max_value = np.max(histogram_arr)
    rate = 256 / max_value

    # 调整比例
    for i in range(256):
        histogram_arr[i] = histogram_arr[i] * rate

    #转化为直方图
    histogram = np.zeros((256,256), np.uint8)
    for i in range(0,255):
        for j in range(256-int(histogram_arr[i]),256):
            histogram[j,i] = 255
    return histogram

ha = Grayscale_histogram(imga)
hb = Grayscale_histogram(imgb)
hc = Grayscale_histogram(imgc)
hg = Grayscale_histogram(imgg)
hh = Grayscale_histogram(imgh)
hi = Grayscale_histogram(imgi)

fig, axs = plt.subplots(2, 3, constrained_layout=True, dpi=300)
plt.subplot(231), plt.imshow(imga, cmap='gray'), plt.title('gaussian')
plt.subplot(232), plt.imshow(imgb,cmap='gray'),plt.title('rayleigh')
plt.subplot(233), plt.imshow(imgc,cmap='gray'),plt.title('gamma')
plt.subplot(234), plt.imshow(ha, cmap='gray')
plt.subplot(235), plt.imshow(hb,cmap='gray')
plt.subplot(236), plt.imshow(hc,cmap='gray')
plt.show()

fig, axs = plt.subplots(2, 3, constrained_layout=True, dpi=300)
plt.subplot(231), plt.imshow(imgg, cmap='gray')
plt.subplot(232), plt.imshow(imgh,cmap='gray')
plt.subplot(233), plt.imshow(imgi,cmap='gray')
plt.subplot(234), plt.imshow(hg, cmap='gray')
plt.subplot(235), plt.imshow(hh,cmap='gray')
plt.subplot(236), plt.imshow(hi,cmap='gray')
plt.show()











