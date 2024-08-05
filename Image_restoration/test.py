import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('img/Fig0441(a)(characters_test_pattern).tif',cv2.IMREAD_GRAYSCALE)


# 放大并填充图片
def amplify_fill(img):
    M, N = img.shape
    P = 2 * M
    Q = 2 * N
    amplify_img = np.zeros((P,Q), np.uint8)
    for i in range(0,M):
        for j in range(0,N):
            amplify_img[i,j] = img[i,j]

    return amplify_img

# 中心化图谱
def centerlization(img):
    M,N = img.shape
    center_img = np.zeros(img.shape, np.uint8)
    for i in range(M):
        for j in range(N):
            center_img[i,j] = img[i,j] * pow(-1,i+j)
    return center_img


#二维离散积分,取样间隔为T
def discrete_integral_Tow(img,u,v,T):
    M, N = img.shape
    for i in range(0,M,T):
        for j in range(0,N,T):
            F = img[i,j] * pow(math.e, complex(0,2*math.pi*(u*i/M+v*j/N)))
    return F

# 二维离散傅里叶变换,
def IDFT(img,T):

    M, N = img.shape
    frequency_img =  np.zeros((M, N), dtype=complex)
    for i in range(M):
        for j in range(N):
            frequency_img[i,j] = discrete_integral_Tow(img,i,j,T)
        print(i)
    return frequency_img

def deal_negative(img):
    real_fft = np.real(img)
    deal_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i,j]<0):
                deal_img[i,j] = 0
            else:
                deal_img[i,j] = real_fft[i,j]
    return deal_img

def multiplication(img1,img2):
    M, N = img1.shape
    multiplication_img = np.zeros((M, N),dtype=complex)
    for i in range(M):
        for j in range(N):
            multiplication_img[i, j] = img1[i, j] * img2[i, j]
    return multiplication_img

# 理想低通滤波
def idea_low_pass_filter(img, r):
    P = img.shape[0]
    Q = img.shape[1]
    filter = np.zeros((P, Q), np.uint8)
    ch = P/2
    cw = Q/2
    for i in range(P):
        for j in range(Q):
            D = math.sqrt(pow(i-ch, 2)+pow(j-cw, 2))
            if D <= r:
                filter[i, j] = 1
            else:
                filter[i, j] = 0
    return filter

# 巴特沃斯低通滤波器
def BLPF(img,r,n):
    P, Q = img.shape
    filter = np.zeros((P, Q), np.uint8)
    ch = P / 2
    cw = Q / 2
    for i in range(P):
        for j in range(Q):
            D = math.sqrt(pow(i-ch, 2)+pow(j-cw, 2))
            filter[i, j] = 1/pow(1+D/r,2*n)

    return filter



aimg = amplify_fill(img)
cimg = np.fft.fftshift(aimg)
H = idea_low_pass_filter(cimg,30)

f = np.fft.fft2(cimg)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
mimg = multiplication(H,fshift)

# iG = np.fft.ifftshift(mimg)
Gshift = np.fft.ifft2(mimg)
iG = np.fft.fftshift(Gshift)
G = np.abs(iG)

result = G[0:G.shape[0]//2,0:G.shape[1]//2]

plt.subplot(331),plt.imshow(img,cmap='gray'),plt.title('raw')
# plt.subplot(332),plt.imshow(aimg,cmap='gray'),plt.title('amplify')
# plt.subplot(333),plt.imshow(cimg,cmap='gray'),plt.title('center')
# plt.subplot(334),plt.imshow(H,cmap='gray'),plt.title('filter')
# plt.subplot(335),plt.imshow(magnitude_spectrum,cmap='gray'),plt.title('DFT')
# plt.subplot(336),plt.imshow(G,cmap='gray'),plt.title('t')
# plt.subplot(337),plt.imshow(result,cmap='gray'),plt.title('result')

plt.show()