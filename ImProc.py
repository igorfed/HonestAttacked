
import os
import functInColor
import functInColor
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt

import re



class ImProcess(object):
    def __init__(self, folder1, folder2, output, Frame):
        self.dirName1 = folder1
        self.dirName2 = folder2
        self.frame = Frame
        self.dirNOutput = output
        self.ext = '.png'
        self.fig = plt.figure()
        self.im_Nam = []
        self.im_Num = []
        self.im_Cor = []
        self.new_height, self.new_width = 400, 400
        self.ifFolderExist(True)
        #self.ifFolderExist(False)

    def ifFolderExist(self, Folder_select):
        print ("----------------")
        if Folder_select == True:  folder = self.dirName1
        else:
            folder = self.dirName2

        if os.path.exists(folder) and os.path.isdir(folder):
            if not os.listdir(self.dirName1):
                print(functInColor.color.RED + functInColor.color.BOLD + "Directory is empty " + functInColor.color.RED)
            else:
                print(functInColor.color.GREEN + functInColor.color.BOLD + "Directory is not empty " + functInColor.color.END)
                cnt = self.directory(folder)
        else:
            print(functInColor.color.RED + functInColor.color.BOLD + "Given Directory does not exists" + functInColor.color.END)


    def directory(self,folder):

        list_dir = os.listdir(folder)
        cnt = 0
        for file in list_dir:

            if file.endswith(self.ext):
                print(re.findall('\d+', file)[0])
                if re.findall('\d+', file)[0] == str(self.frame):
                    cnt = cnt + 1
                    image_name = folder+ "/"+file
                    print(image_name)
                    self.im_Nam = re.findall('\d+', file)[0]
                    self.im_Num = re.findall('\d+', file)[1]
                    self.im_Cor = []
                    self.FFT2D(image_name)
                else:
                    print(functInColor.color.RED + functInColor.color.BOLD + "frame_"+str(self.frame)+ " not found" + functInColor.color.END)

        cv2.destroyAllWindows()
        plt.show()
        print(functInColor.color.BLUE + functInColor.color.BOLD + str(cnt),  ": " + self.ext + ": files " + functInColor.color.END)
        return cnt
    def centeredCrop(self, im, shiftx, shifty):
        width = np.size(im, 1)
        height = np.size(im, 0)
        left = int(np.ceil((width - self.new_width) / 2.))
        top = int(np.ceil((height - self.new_height) / 2.))
        right = int(np.floor((width + self.new_width) / 2.))
        bottom = int(np.floor((height + self.new_height) / 2.))
        cImg = im[top+shiftx:bottom+shiftx, left+shifty:right+shifty]
        return cImg

    def im_show(self):
        H1 = cv2.hconcat([self.img,self.imgfft,self.imjcorr ])
        H2 = cv2.hconcat([self.der2img, self.derfft, self.imjcorr])
        C = cv2.vconcat([H1, H2])
        cv2.imshow('image', C)
        cv2.imwrite(self.dirNOutput + "/" + 'frame_'+self.im_Nam + '_'+self.im_Num + ".jpeg", C)
        cv2.waitKey(100)

    def FFT2D (self, image_name):
        def magnitude(Im):
            fShift= np.fft.fftshift(np.fft.fft2(Im))
            magnitudeFShift = np.array(20 * np.log(np.abs(fShift)), dtype=np.uint8)
            return fShift, magnitudeFShift
        def readim(image_name):
            img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            if img.shape[0]== img.shape[1]:
                im =copy.deepcopy(img)
            else:
                im = img[0:0+min(img.shape[0],img.shape[1]), 0:0+min(img.shape[0],img.shape[1])]
            return im
        self.source_image = readim(image_name)
        self.img =  self.centeredCrop(self.source_image, 0, 0)

        self.Derivative2()
        self.imgfftShift, self.imgfft, = magnitude(self.img)
        self.devfftShift,self.derfft = magnitude(self.der2img)
        self.fig.clear()
        ax1 = self.fig.add_subplot(2, 1, 1)
        ax2 = self.fig.add_subplot(2, 1, 2)

        for shift_x in range(0, 7):

            Cor, cc_image = self.Correlation_Shift(shift_x)

            ax1.plot(cc_image.real[int(cc_image.real.shape[1] / 2), :],label=shift_x)
            ax2.plot(cc_image.real[int(cc_image.real.shape[1] / 2), :],label=shift_x)
            ax1.legend()
            ax2.legend()
            ax1.set_title(self.im_Nam)
            ax2.set_title(self.im_Nam)
            self.fig.savefig(self.dirNOutput + "/" + 'frame_' + self.im_Nam + '_' + self.im_Num +"_"+str(shift_x) + ".jpeg")
            self.im_Cor.append(Cor)
            ax1.cla()
        ax1.set_title(self.im_Nam)
        ax2.plot(self.im_Cor, label=self.im_Num)
        ax2.set_title(self.im_Nam)
        plt.draw()
        plt.pause(1)
        self.im_show()

    def Correlation(self):
        imgcj = np.conjugate(self.devfftShift)
        prod = copy.deepcopy(imgcj)
        for x in range (imgcj.shape[0]):
            for y in range (imgcj.shape[1]):
                prod[x][y] = self.imgfftShift[x][y] * self.imgfftShift[x][y]
        cc = (np.real(np.fft.ifft2(np.fft.fftshift(prod))))  # real image of the correlation
        cv2.imshow('correlation',(np.real(prod)))
        cc = prod
        return cc

    def Correlation_Shift(self, shiftX):
        cv2.Laplacian(self.img, cv2.CV_64F)
        image_product = np.fft.fft2(cv2.Laplacian(self.img, cv2.CV_64F))
        image_offset = self.centeredCrop(cv2.Laplacian(self.source_image, cv2.CV_64F), shiftX, 0)
        image_offset = np.fft.fft2(image_offset).conj()
        cc_image = np.abs(np.fft.fftshift(np.fft.ifft2(image_product * image_offset)))
        normalizedImg = np.zeros((self.img.shape[0], self.img.shape[1]))
        self.imjcorr = np.array(cv2.normalize(cc_image.real, normalizedImg, 0, 255, cv2.NORM_MINMAX), dtype=np.uint8)
        Cor = np.max(cc_image.real)

        return Cor, cc_image




#        self.imjcorr = np.array(cv2.normalize(cc_image.real, normalizedImg, 0, 255, cv2.NORM_MINMAX), dtype=np.uint8)

#    def Correlation1(self):
#        image_product = np.fft.fft2(self.img) * np.fft.fft2(self.img).conj()
#        cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
#        normalizedImg = np.zeros((self.img.shape[0], self.img.shape[1]))
#        self.im_Cor = np.max(cc_image.real)
#        self.imjcorr = np.array(cv2.normalize(cc_image.real, normalizedImg, 0, 255, cv2.NORM_MINMAX), dtype=np.uint8)
        #cc_image = np.array((np.abs(cc_image)), dtype=np.uint8)
#        self.fig.clear()
#        ax1 = self.fig.add_subplot(1, 2, 1)
#        ax2 = self.fig.add_subplot(1, 2, 2)

        #cv2.normalize(cc_image.real, np.zeros(self.img.shape[0],self.img.shape[1]), 0, 255, cv2.NORM_MINMAX)

        #image = np.array(image, dtype=np.uint8)
        #cv2.imshow('cor',image)
        #ax1.imshow(cc_image.real, cmap='gray')
        #ax1.imshow(im, cmap='gray')

        #ax2.plot(cc_image.real[:, int(self.img.shape[1] / 2)])
        #ax2.plot(im[:, int(self.img.shape[1] / 2)])

    def Derivative2(self):
        self.der2img = cv2.Laplacian(self.img, cv2.CV_64F)
        self.der2img = np.asarray(self.der2img, dtype=np.uint8)


def mkDir(dirName="Output"):
    try:
        os.mkdir(dirName)
        print(functInColor.color.GREEN + functInColor.color.BOLD + "Directory {}: Created".format(
            dirName) + functInColor.color.END)
    except FileExistsError:
        print(functInColor.color.BLUE + "Directory {}: ALREADY EXIST".format(dirName) + functInColor.color.END)
    return dirName