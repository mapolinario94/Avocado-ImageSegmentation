import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.filters import threshold_otsu
import cv2
from scipy import ndimage
from PIL import Image


class ImageSegmentation:

    def rgb_image(self, path="../data/test/RGB/IMG_00123.jpg", plot=False):
        """

        :param path:
        :param plot:
        :return:
        """
        img_rgb = io.imread(path)
        img_lab = color.rgb2lab(img_rgb)

        a_comp = img_lab[:, :, 1]
        a_comp = 255*((a_comp-a_comp.min())/(a_comp.max()-a_comp.min()))

        img_ls = self.__linear_stretching(a_comp, lim_sup_p=70)

        mask = self.__masking(img_ls)

        if plot:
            self.__plot_segmentation("RGB Segmentation: {0}".format(path), img_rgb, mask)

        return mask

    def nir_image(self, path, plot=False):
        img_nir = io.imread(path)
        img = img_nir[:, :, 0]
        img = 255 * (img / img.max())

        img_ls = self.__linear_stretching(img, lim_sup_p=97)

        mask = self.__masking(255-img_ls, dilatation_factor=35, erosion_factor=15, min_size=100000)

        if plot:
            self.__plot_segmentation("NIR Segmentation: {0}".format(path), img, mask)

        return mask

    def thermal_image(self, path, plot=False):
        img = Image.open(path)
        img = np.array(img)
        img = img * 0.040 - 273.15

        img_ls = self.__linear_stretching(img, lim_sup_p=70)

        mask = self.__masking(img_ls, erosion_factor=0, dilatation_factor=0, min_size=8000)

        if plot:
            self.__plot_segmentation("Thermal Segmentation: {0}".format(path), img_ls, mask)

        return mask

    @staticmethod
    def __plot_segmentation(title, img, mask):
        plt.figure(title)
        plt.subplot(121)
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        plt.axis(False)
        plt.title("Original Image")
        plt.subplot(122)
        plt.imshow(mask, cmap="gray", vmin=0, vmax=255)
        plt.axis(False)
        plt.title("Result")
        plt.show()

    @staticmethod
    def __linear_stretching(img, lim_inf_p=3, lim_sup_p=97):
        """

        :param img:
        :param lim_inf_p:
        :param lim_sup_p:
        :return:
        """
        hist, bin_edges = np.histogram(img, bins=256, range=[0, 255])
        cdf = hist.cumsum()

        lim_inf = cdf.max() * lim_inf_p / 100
        lim_sup = cdf.max() * lim_sup_p / 100

        img_min = np.argmax(np.where(cdf < lim_inf))
        img_max = np.argmax(np.where(cdf < lim_sup))

        img_clip = np.clip(img, img_min, img_max)

        img_ls = 255 * (img_clip - img_clip.min()) / (img_clip.max() - img_clip.min())

        return img_ls

    @staticmethod
    def __masking(img, dilatation_factor=15, erosion_factor=15, min_size=30000):
        """

        :param img:
        :param dilatation_factor:
        :param min_size:
        :return:
        """

        thresh = threshold_otsu(img)
        binary = np.uint8((img < thresh) * 1)
        mask = np.zeros_like(binary, dtype=np.uint8)

        binary = np.uint8(ndimage.binary_fill_holes(binary))

        n_label, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        for l in range(1, n_label):
            if stats[l, cv2.CC_STAT_AREA] >= min_size:
                mask[labels == l] = 255

        mask = (mask == 255)
        for k in range(dilatation_factor):
            mask = ndimage.binary_dilation(mask).astype(int)
        mask = ndimage.binary_fill_holes(mask).astype(int)
        mask = np.uint8(mask * 255)

        mask = (mask == 255)
        for k in range(erosion_factor):
            mask = ndimage.binary_erosion(mask).astype(int)
        mask = np.uint8(mask * 255)

        return mask


if __name__ == "__main__":
    img_obj = ImageSegmentation()
    _ = img_obj.rgb_image("../data/RGB/Original/20200213_090858_240_8b.JPG", plot=True)
    _ = img_obj.nir_image("../data/NIR/Original/IMG_00050.jpg", plot=True)
    _ = img_obj.thermal_image("../data/FLIR/Original/20200217_105542_349_IR.TIFF", plot=True)
