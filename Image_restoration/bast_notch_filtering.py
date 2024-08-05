import cv2
import matplotlib.pyplot as plt
import numpy as np

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('img/Fig0520(a)(NASA_Mariner6_Mars).tif', cv2.IMREAD_GRAYSCALE)

# 放大并填充图片
def amplify_fill(img):
    M, N = img.shape
    P = 2 * M
    Q = 2 * N
    amplify_img = np.zeros((P, Q), np.uint8)
    for i in range(0, M):
        for j in range(0, N):
            amplify_img[i, j] = img[i, j]

    return amplify_img

# 矩阵相乘
def multiplication(img1, img2):
    M, N = img1.shape
    multiplication_img = np.zeros((M, N),dtype=complex)
    for i in range(M):
        for j in range(N):
            multiplication_img[i, j] = img1[i, j] * img2[i, j]
    return multiplication_img

def multiplication1(img1, img2):
    M, N = img1.shape
    multiplication_img = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            multiplication_img[i, j] = img1[i, j] * img2[i, j]
    return multiplication_img

# 计算邻域内均值
def average(x,y,img,kernel_size):
    h,w = img.shape
    left = int(kernel_size / 2)
    right = kernel_size - left
    sum = 0
    for i in range(x - left, x + right):
        for j in range(y - left, y + right):
            if(i>=0 and i<h and j>=0 and j<w):
                sum = sum + img[i][j]
    return sum/(kernel_size*kernel_size)

#计算噪声权重函数
def weight_f(g,n,kernel_size):
    h,w = g.shape
    weight = np.zeros((h, w))

    mult_img = multiplication1(g,n)
    mult_n2 = multiplication1(n,n)
    for i in range(h):
        for j in range(w):
            A = average(i,j,mult_img,kernel_size)
            B = average(i,j,g,kernel_size)*average(i,j,n,kernel_size)
            C = average(i,j,mult_n2,kernel_size)
            D = pow((average(i,j,n,kernel_size)),2)
            weight[i][j] = (A-B)/(C-D)

    return weight

def Frequent_img(img):
    aimg = amplify_fill(img)        # 扩大填充
    cimg = np.fft.fftshift(aimg)    # 中心化

    f = np.fft.fft2(cimg)  #傅里叶
    fshift = np.fft.fftshift(f) #傅里叶
    magnitude_spectrum = 20 * np.log(np.abs(fshift)) #强化显示频率域图

    return magnitude_spectrum


#观察并设置矩形滤波器
def filter(img,flag):
    h,w = img.shape
    h = h * 2
    w = w * 2
    f = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if(i>=34 and i<40) or (i>=15 and i<21) or (j>=34 and j<40) or (j>=15 and j<21)\
                    or (j<(w/2-14) and j>=(w/2-24)) or (j>=(w/2+14) and j<(w/2+24))\
                    or (i<=h-34 and i>h-40) or (i<h-15 and i>=h-21) or (j<=w-34 and j>w-40) or (j<w-15 and j>=w-21):
                f[i][j] = flag
            else:
                f[i][j] = abs(flag - 1)
    return f


def resume(img,noise,kernel_size):
    h,w = img.shape
    resume_img = np.zeros((h, w),dtype=np.uint8)
    weight = weight_f(img,noise,kernel_size)
    for i in range(h):
        for j in range(w):
            t = img[i][j] -weight[i][j]*noise[i][j]
            resume_img[i][j] = t
            
    return resume_img

def filtering(img,H):
    aimg = amplify_fill(img)  # 扩大填充
    cimg = np.fft.fftshift(aimg)  # 中心化

    f = np.fft.fft2(cimg)  # 傅里叶
    fshift = np.fft.fftshift(f)  # 傅里叶
    magnitude_spectrum = 20 * np.log(np.abs(fshift))  # 强化显示频率域图
    mimg = multiplication(H, fshift)   #滤波

    Gshift = np.fft.ifft2(mimg)  #反傅里叶
    iG = np.fft.ifftshift(Gshift) #反中心化回去

    G = np.abs(iG) #去除虚部
    result = G[0:G.shape[0] // 2, 0:G.shape[1] // 2]

    return result


Frequent_IMG = Frequent_img(img)
Filter_Pass = filter(img,1)
Filter_Reject = filter(img,0)
b = multiplication1(Frequent_IMG,Filter_Reject)
test = filtering(img,Filter_Reject)

noise = filtering(img,Filter_Pass)
resume_img = resume(img,noise,3)

fig,axs=plt.subplots(2,3,constrained_layout=True,dpi=1200)
plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title('Pollute')
plt.subplot(232), plt.imshow(Frequent_IMG, cmap='gray'), plt.title('Frequent_IMG')
plt.subplot(234), plt.imshow(b, cmap='gray'), plt.title('Frequent_IMG+BPF')
plt.subplot(233), plt.imshow(Filter_Pass, cmap='gray'), plt.title('BRF')
plt.subplot(235), plt.imshow(noise, cmap='gray'), plt.title('Noise')
plt.subplot(236), plt.imshow(test, cmap='gray'), plt.title('Resume')
plt.savefig('sy2.png')
plt.show()