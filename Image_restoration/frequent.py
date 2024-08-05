import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('img/Fig0441(a)(characters_test_pattern).tif', cv2.IMREAD_GRAYSCALE)


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

# 二维离散积分,取样间隔为T
def discrete_integral_Tow(img, u, v, T):
    M, N = img.shape
    for i in range(0, M, T):
        for j in range(0, N, T):
            F = img[i, j] * pow(math.e, complex(0, 2 * math.pi * (u * i / M + v * j / N)))
    return F


def multiplication(img1, img2):
    M, N = img1.shape
    multiplication_img = np.zeros((M, N), dtype=complex)
    for i in range(M):
        for j in range(N):
            multiplication_img[i, j] = img1[i, j] * img2[i, j]
    return multiplication_img


# 理想低通滤波
def idea_low_pass_filter(img, r):
    M,N = img.shape
    P = 2 * M
    Q = 2 * N
    filter = np.zeros((P, Q))
    for i in range(P):
        for j in range(Q):
            D = math.sqrt(pow(i - M, 2) + pow(j - N, 2))
            if D <= r:
                filter[i, j] = 1
            else:
                filter[i, j] = 0
    return filter


# 巴特沃斯低通滤波器
def BLPF(img, r, n):
    M, N = img.shape
    P = 2 * M
    Q = 2 * N
    filter = np.zeros((P, Q))
    for i in range(P):
        for j in range(Q):
            D = math.sqrt(pow(i - M, 2) + pow(j - N, 2))
            filter[i, j] = 1 / pow(1 + D / r, 2 * n)

    return filter


# 高斯低通滤波器
def CLPF(img, r):
    M, N = img.shape
    P = 2 * M
    Q = 2 * N
    filter = np.zeros((P, Q))
    for i in range(P):
        for j in range(Q):
            D = math.sqrt(pow(i - M, 2) + pow(j - N, 2))
            filter[i, j] =  pow(math.e,-1*(D*D)/(2*r*r))

    return filter

# 理想高通滤波
def idea_high_pass_filter(img, r):
    M,N = img.shape
    P = 2 * M
    Q = 2 * N
    filter = np.zeros((P, Q))
    for i in range(P):
        for j in range(Q):
            D = math.sqrt(pow(i - M, 2) + pow(j - N, 2))
            if D <= r:
                filter[i, j] = 0
            else:
                filter[i, j] = 1
    return filter


# 巴特沃斯高通滤波器
def BHPF(img, r, n):
    M, N = img.shape
    P = 2 * M
    Q = 2 * N
    filter = np.zeros((P, Q))
    for i in range(P):
        for j in range(Q):
            D = math.sqrt(pow(i - M, 2) + pow(j - N, 2))
            filter[i, j] = 1 / pow(1 + r / (D+0.0001), 2 * n)

    return filter


# 高斯高通滤波器
def CHPF(img, r):
    M, N = img.shape
    P = 2 * M
    Q = 2 * N
    filter = np.zeros((P, Q))
    for i in range(P):
        for j in range(Q):
            D = math.sqrt(pow(i - M, 2) + pow(j - N, 2))
            filter[i, j] =1 - pow(math.e,-1*(D*D)/(2*r*r))

    return filter

def filtering(H,img):
    aimg = amplify_fill(img)        # 扩大填充
    cimg = np.fft.fftshift(aimg)    # 中心化

    f = np.fft.fft2(cimg)  #傅里叶
    fshift = np.fft.fftshift(f) #傅里叶
    magnitude_spectrum = 20 * np.log(np.abs(fshift)) #强化显示频率域图
    mimg = multiplication(H, fshift)   #滤波

    Gshift = np.fft.ifft2(mimg)  #反傅里叶
    iG = np.fft.ifftshift(Gshift) #反中心化回去
    G = np.abs(iG) #去除虚部
    result = G[0:G.shape[0] // 2, 0:G.shape[1] // 2]
    return result,magnitude_spectrum

H_idea1 = idea_high_pass_filter(img, 10)
H_idea2 = idea_high_pass_filter(img, 30)
result1,magnitude_spectrum1 = filtering(H_idea1, img)
result2,magnitude_spectrum2 = filtering(H_idea2, img)

H_BHPF1 = BHPF(img,10,1)
H_BHPF2 = BHPF(img,30,1)
result3,magnitude_spectrum3 = filtering(H_BHPF1, img)
result4,magnitude_spectrum4 = filtering(H_BHPF2, img)
H_CHPF1 = CHPF(img,10)
H_CHPF2 = CHPF(img,20)
result5,magnitude_spectrum5 = filtering(H_CHPF1, img)
result6,magnitude_spectrum6 = filtering(H_CHPF2, img)

fig, axs = plt.subplots(4, 2, constrained_layout=True, dpi=300)
plt.subplot(421), plt.imshow(img, cmap='gray'), plt.title('raw')
plt.subplot(422),plt.imshow(result1,cmap='gray'),plt.title('IHPF,D=10')
plt.subplot(423),plt.imshow(result2,cmap='gray'),plt.title('IHPF,D=30')
plt.subplot(424),plt.imshow(result3,cmap='gray'),plt.title('BHPF,D=10')
plt.subplot(425), plt.imshow(result4, cmap='gray'), plt.title('BHPF,D=30')
plt.subplot(426),plt.imshow(result5,cmap='gray'),plt.title('CHPF,D=10')
plt.subplot(427),plt.imshow(result6,cmap='gray'),plt.title('CHPF,D=20')
plt.show()

H_CLPF1 = CLPF(img,10)
H_CLPF2 = CLPF(img,20)
H_CLPF3 = CLPF(img,60)

result1,magnitude_spectrum1 = filtering(H_CLPF1, img)
result2,magnitude_spectrum2 = filtering(H_CLPF2, img)
result3,magnitude_spectrum3 = filtering(H_CLPF3, img)

fig, axs = plt.subplots(2, 2, constrained_layout=True, dpi=300)
plt.subplot(221), plt.imshow(img, cmap='gray')
plt.subplot(222),plt.imshow(result1,cmap='gray')
plt.subplot(223),plt.imshow(result2,cmap='gray')
plt.subplot(224),plt.imshow(result3,cmap='gray')
plt.show()

H_BLPF1 = BLPF(img,10,1)
H_BLPF2 = BLPF(img,30,1)
H_BLPF3 = BLPF(img,100,1)

result1,magnitude_spectrum1 = filtering(H_BLPF1, img)
result2,magnitude_spectrum2 = filtering(H_BLPF2, img)
result3,magnitude_spectrum3 = filtering(H_BLPF3, img)

fig, axs = plt.subplots(2, 2, constrained_layout=True, dpi=300)
plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('raw')
plt.subplot(222),plt.imshow(result1,cmap='gray'),plt.title('D=10')
plt.subplot(223),plt.imshow(result2,cmap='gray'),plt.title('D=20')
plt.subplot(224),plt.imshow(result3,cmap='gray'),plt.title('D=60')
plt.show()

H_idea1 = idea_low_pass_filter(img, 10)
H_idea2 = idea_low_pass_filter(img, 30)
H_idea3 = idea_low_pass_filter(img, 100)

result1,magnitude_spectrum1 = filtering(H_idea1, img)
result2,magnitude_spectrum2 = filtering(H_idea2, img)
result3,magnitude_spectrum3 = filtering(H_idea3, img)


fig, axs = plt.subplots(2, 2, constrained_layout=True, dpi=300)
plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('raw')
plt.subplot(222),plt.imshow(result1,cmap='gray'),plt.title('D=10')
plt.subplot(223),plt.imshow(result2,cmap='gray'),plt.title('D=20')
plt.subplot(224),plt.imshow(result3,cmap='gray'),plt.title('D=60')
plt.show()