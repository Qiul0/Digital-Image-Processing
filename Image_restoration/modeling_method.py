import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('img/Fig0526(a)(original_DIP).tif', cv2.IMREAD_GRAYSCALE)

# 放大并填充边缘
def adjust_fill(img):
    M, N = img.shape
    adjust_img = np.zeros((M+2, N+2), np.uint8)
    for i in range(1, M+1):
        for j in range(1, N+1):
            adjust_img[i, j] = img[i, j]

    return adjust_img

# 裁剪边缘
def adjust_cut(img):
    h, w = img.shape
    adjust_img = img[1:h-1, 1:w-1]
    return adjust_img

# 矩阵相乘
def multiplication(img1, img2):
    M, N = img1.shape
    multiplication_img = np.zeros((M, N), dtype=complex)
    for i in range(M):
        for j in range(N):
            multiplication_img[i, j] = img1[i, j] * img2[i, j]
    return multiplication_img



def Power(img):

    h,w = img.shape
    power = np.zeros((h,w))
    f = np.fft.fft2(img)  # 傅里叶
    Frequent = np.fft.fftshift(f)  # 傅里叶
    real_arr = Frequent.real
    imag_arr = Frequent.imag
    for i in range(h):
        for j in range(w):
            power[i, j] = real_arr[i, j]**2 + imag_arr[i, j]**2
    return power

# 湍流退化函数
def Turbulence(k,img):
    h,w = img.shape
    P = h * 2
    Q = w * 2
    H = np.zeros((P,Q),dtype=np.float32)
    for i in range(P):
        for j in range(Q):
            t = -1 * k * pow(((i-h)**2 + (j-w)**2), 5 / 6)
            H[i, j] = pow(math.e, t)
    return H


def Wiener(img,H,K):
    h,w = img.shape
    P = h * 2
    Q = w * 2
    wiener = np.zeros((P,Q),dtype=np.float32)
    for i in range(P):
        for j in range(Q):
            real = H[i, j].real
            imag = H[i, j].imag
            h = real**2 + imag**2
            wiener[i, j] = (h/(h+K))/(H[i, j]+0.000001)

    return wiener


# 运动模糊
def Move_Average_Noise(a,b,T,img):
    P, Q = img.shape
    h = int(P / 2)
    w = int(Q / 2)
    cimg = np.fft.fftshift(img)  # 中心化
    f = np.fft.fft2(cimg)  # 傅里叶
    fshift = np.fft.fftshift(f)  # 傅里叶

    pollute_img = np.zeros((P, Q),dtype=complex)
    H = np.zeros((P, Q), dtype=complex)

    for i in range(P):
        for j in range(Q):
            u = i - h + 0.0000001
            v = j - w + 0.0000001
            t = complex(0, (-1 * math.pi * (u * a + v * b)))
            if i == 0 or j == 0 or i == P or j == P:
                H[i, j] = 1
            else:
                H[i,j] = T/(math.pi * (u * a + v * b)) * math.sin(math.pi * (u * a + v * b)) * pow(math.e, t)
            pollute_img[i, j] = H[i,j] * fshift[i, j]

    Gshift = np.fft.ifft2(pollute_img)  # 反傅里叶
    iG = np.fft.ifftshift(Gshift)  # 反中心化回去
    result = np.abs(iG)  # 去除虚部

    return result,H



#最小乘方滤波
def Minimum_power(pult,H,k,p):
    P, Q = pult.shape
    h =int(P / 2)
    w =int(Q / 2)


    cimg = np.fft.fftshift(pult)  # 中心化
    f = np.fft.fft2(cimg)  # 傅里叶
    pultshift = np.fft.fftshift(f)  # 傅里叶

    Hconj = np.conj(H)
    minimum_power = np.zeros((P, Q), dtype=complex)
    Pxy = np.zeros((P, Q), dtype=int)

    # 扩大并移到中心
    t = int((p.shape[0]-1)/2)
    for x in range(h-t,h+t+1):
        for y in range(w-t,w+t+1):
            Pxy[x,y] = p[x-h+t,y-w+t]

    cimg = np.fft.fftshift(Pxy)  # 中心化
    f = np.fft.fft2(cimg)  # 傅里叶
    Puv = np.fft.fftshift(f)  # 傅里叶

    # 计算最小乘方滤波器
    for i in range(P):
        for j in range(Q):
            minimum_power[i,j] = pultshift[i, j] * Hconj[i,j] / (H[i,j]*Hconj[i,j] + k * Puv[i,j])

    Gshift = np.fft.ifft2(minimum_power)  # 反傅里叶
    iG = np.fft.ifftshift(Gshift)  # 反中心化回去
    result = np.abs(iG)  # 去除虚部

    return result


pxy = np.array([[0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]])

Pollute_img,H = Move_Average_Noise(0.1,0.1,1,img)
resume = Minimum_power(Pollute_img,H,0.001,pxy)

fig, axs = plt.subplots(1, 3, constrained_layout=True, dpi=300)
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Raw')
plt.subplot(132), plt.imshow(Pollute_img, cmap='gray'), plt.title('Pollute_img')
plt.subplot(133), plt.imshow(resume, cmap='gray'), plt.title('Resume')
plt.savefig('sy4.png')
plt.show()