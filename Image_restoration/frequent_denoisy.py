import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('img/Fig0424(a)(rectangle).tif', cv2.IMREAD_GRAYSCALE)

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


# 带复数的矩阵相乘
def multiplication(img1, img2):
    M, N = img1.shape
    multiplication_img = np.zeros((M, N), dtype=complex)
    for i in range(M):
        for j in range(N):
            multiplication_img[i, j] = img1[i, j] * img2[i, j]
    return multiplication_img


# 理想陷波带阻滤波器
def INBSF(img, r, u, v):
    M,N = img.shape
    P = 2 * M
    Q = 2 * N
    filter = np.zeros((P, Q))
    for i in range(P):
        for j in range(Q):
            D1 = math.sqrt(pow(i - M - u, 2) + pow(j - N - v, 2))
            D2 = math.sqrt(pow(i - M + u, 2) + pow(j - N + v, 2))
            if D1 <= r or D2 <= r:
                filter[i, j] = 0
            else:
                filter[i, j] = 1
    return filter

# 巴特沃斯陷波带阻滤波器
def BNBSF(img, r, n, u, v):
    M, N = img.shape
    P = 2 * M
    Q = 2 * N
    filter = np.zeros((P, Q))
    for i in range(P):
        for j in range(Q):
            D1 = math.sqrt(pow(i - M - u, 2) + pow(j - N - v, 2)) + 0.000001
            D2 = math.sqrt(pow(i - M + u, 2) + pow(j - N + v, 2)) + 0.000001
            H1 = 1 / pow(1 + r / D1, 2 * n)
            H2 = 1 / pow(1 + r / D2, 2 * n)
            filter[i, j] = H1 * H2
    return filter


# 高斯陷波带阻滤波器
def GNBSF(img, r, u, v):
    M, N = img.shape
    P = 2 * M
    Q = 2 * N
    filter = np.zeros((P, Q))
    for i in range(P):
        for j in range(Q):
            D1 = math.sqrt(pow(i - M - u, 2) + pow(j - N - v, 2))
            D2 = math.sqrt(pow(i - M + u, 2) + pow(j - N + v, 2))
            H1 = 1 - pow(math.e,-1*(D1*D1)/(2*r*r))
            H2 = 1 - pow(math.e,-1*(D2*D2)/(2*r*r))
            filter[i, j] =  H1 * H2

    return filter

# 理想高通滤波
def INBPF(img, r, u, v):
    return 1 - INBSF(img,r, u, v)

# 巴特沃斯高通滤波器
def BNBPF(img, r, n, u, v):
    return 1 - BNBSF(img,r,n,u,v)


# 高斯高通滤波器
def GNBPF(img, r, u, v):
    return 1-GNBSF(img, r, u, v)

def filtering(H,img):
    aimg = amplify_fill(img)        # 扩大填充
    cimg = np.fft.fftshift(img)    # 中心化

    f = np.fft.fft2(cimg)  #傅里叶
    fshift = np.fft.fftshift(f) #傅里叶
    magnitude_spectrum = 20 * np.log(np.abs(fshift)) #强化显示频率域图
    mimg = multiplication(H, fshift)   #滤波

    Gshift = np.fft.ifft2(mimg)  #反傅里叶
    iG = np.fft.ifftshift(Gshift) #反中心化回去

    G = np.abs(iG) #去除虚部
    result = G[0:G.shape[0] // 2, 0:G.shape[1] // 2]
    return result


I_H = INBSF(img,20,500,500)
H_H = BNBSF(img,20,3,500,500)
G_H = GNBSF(img,20,500,500)

P_H = INBPF(img,10,30,30)
P_img = filtering(P_H,img)
Re_img = filtering(I_H,img)

fig,axs=plt.subplots(2,3,constrained_layout=True,dpi=300)
plt.subplot(231), plt.imshow(I_H, cmap='gray'), plt.title('INBSF')
plt.subplot(232), plt.imshow(H_H, cmap='gray'), plt.title('BNBSF')
plt.subplot(233), plt.imshow(G_H, cmap='gray'), plt.title('GNBSF')
plt.subplot(234), plt.imshow(img, cmap='gray'), plt.title('raw')
plt.subplot(235), plt.imshow(P_img, cmap='gray'), plt.title('Pollute')
plt.subplot(236), plt.imshow(Re_img, cmap='gray'), plt.title('Resume')
# plt.savefig('output.png')
plt.show()