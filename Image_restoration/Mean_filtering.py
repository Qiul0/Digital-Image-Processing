import cv2
import matplotlib.pyplot as plt
import numpy as np
import gmpy2

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imge = cv2.imread('img/Fig0507(b)(ckt-board-gauss-var-400).tif', cv2.IMREAD_GRAYSCALE)
imga = cv2.imread('img/Fig0508(a)(circuit-board-pepper-prob-pt1).tif', cv2.IMREAD_GRAYSCALE)
imgb = cv2.imread('img/Fig0508(b)(circuit-board-salt-prob-pt1).tif', cv2.IMREAD_GRAYSCALE)

exp = 1e-7

def median(list):
    list = sorted(list)
    n = len(list)
    if n % 2 == 0:
        return (list[n // 2] + list[n // 2 - 1])/2
    else:
        return list[n // 2]


#算术均值滤波器
def sum_average_filter(x,y,h,w,img,kernel_size):
    left = int(kernel_size / 2)
    right = kernel_size - left
    sum = 0
    for i in range(x - left, x + right):
        for j in range(y - left, y + right):
            if(i>=0 and i<h and j>=0 and j<w):
                sum = sum + img[i][j]
    return int(sum/(kernel_size*kernel_size))


# 几何均值值滤波器
def geometry_average_filter(x,y,h,w,img,kernel_size):
    left = int(kernel_size / 2)
    right = kernel_size - left
    # 大数处理
    geo = gmpy2.mpz(1)
    for i in range(x - left, x + right):
        for j in range(y - left, y + right):
            if(i>=0 and i<h and j>=0 and j<w):
                geo = geo * img[i][j]
    return pow(geo,1/(kernel_size * kernel_size))

#谐波均值滤波器
def harmonic_average_filter(x,y,h,w,img,kernel_size,Q):

    left = int(kernel_size / 2)
    right = kernel_size - left
    sum = 0.0
    for i in range(x - left, x + right):
        for j in range(y - left, y + right):
            if(i>=0 and i<h and j>=0 and j<w):
                sum = sum + 1/(img[i][j]+exp)

    return (kernel_size * kernel_size)/sum

#反谐波均值滤波器
def inharmonic_average_filter(x,y,h,w,img,kernel_size,Q):
    left = int(kernel_size / 2)
    right = kernel_size - left
    sum1 = 0
    sum2 = 0

    for i in range(x - left, x + right):
        for j in range(y - left, y + right):
            if(i>=0 and i<h and j>=0 and j<w):
                sum1 = sum1 + pow(img[i][j]+exp,Q+1)
                sum2 = sum2 + pow(img[i][j]+exp,Q)
    return sum1/(sum2 + exp)
def average_filtering(img,kernel_size,filter_f,Q):
    h,w = img.shape
    filter = np.zeros((h,w),np.uint8)
    left = int(kernel_size/2)
    right = kernel_size-left
    for i in range(h):
        for j in range(w):
            filter[i][j] = filter_f(i,j,h,w,img,kernel_size,Q)
    return filter


fig,axs=plt.subplots(3,2,constrained_layout=True,dpi=300)
imgc = average_filtering(imga,3,inharmonic_average_filter,1.5)
imgd = average_filtering(imgb,3,inharmonic_average_filter,-1.5)
imgf = average_filtering(imge,3,harmonic_average_filter,0)
plt.subplot(321), plt.imshow(imga, cmap='gray'),plt.title('pepper-prob')
plt.subplot(322), plt.imshow(imgb,cmap='gray'),plt.title('salt-prob')
plt.subplot(323), plt.imshow(imgc,cmap='gray'),plt.title('inharmonic_average Q=1.5')
plt.subplot(324), plt.imshow(imgd,cmap='gray'),plt.title('inharmonic_average Q=-1.5')
plt.subplot(325), plt.imshow(imge,cmap='gray'),plt.title('gauss-var')
plt.subplot(326), plt.imshow(imgf,cmap='gray'),plt.title('harmonic-filter')

plt.show()
