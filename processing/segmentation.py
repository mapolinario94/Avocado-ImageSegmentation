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
        This method performs the segmentation of RGB (in jpg or png format) images acquired by a RGB camera

        :param path: PATH of the original RGB image
        :param plot:It is a bool parameter (True or False) to show  the mask automatically segmented and the
        original image. Default: False
        :return: Return a 2D numpy array. It is the mask automatically segmented as a binary image
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
        """
        This method performs the segmentation of NIR (Near Infrared) images acquired by Sentera camera

        :param path: PATH of the original NIR image
        :param plot:It is a bool parameter (True or False) to show  the mask automatically segmented and the
        original image. Default: False
        :return: Return a 2D numpy array. It is the mask automatically segmented as a binary image
        """
        img_nir = io.imread(path)
        img = img_nir[:, :, 0]
        img = 255 * (img / img.max())

        img_ls = self.__linear_stretching(img, lim_sup_p=97)

        mask = self.__masking(255-img_ls, dilatation_factor=35, erosion_factor=15, min_size=100000)

        if plot:
            self.__plot_segmentation("NIR Segmentation: {0}".format(path), img, mask)

        return mask

    def thermal_image(self, path, plot=False):
        """
        This method performs the segmentation of thermal images acquired by FLIR camera

        :param path: PATH of the original Thermal image
        :param plot: It is a bool parameter (True or False) to show  the mask automatically segmented and the
        original image. Default: False
        :return: Return a 2D numpy array. It is the mask automatically segmented as a binary image
        """
        img = Image.open(path)
        img = np.array(img)
        img = img * 0.040 - 273.15

        img_ls = self.__linear_stretching(img, lim_sup_p=70)

        mask = self.__masking(img_ls, erosion_factor=0, dilatation_factor=0, min_size=8000)

        if plot:
            self.__plot_segmentation("Thermal Segmentation: {0}".format(path), img_ls, mask)

        return mask

    def eval_performance(self, path_img, path_target, type_img, plot=False):
        """
        This method eval the performance of the segmentation methods

        :param path_img: PATH of the original image (RGB, NIR or Thermal image)
        :param path_target: PATH of the ground truth mask (black and white)
        :param type_img: There are three types of images: RGB, NIR and Thermal. It is a string parameter
        :param plot: It is a bool parameter (True or False) to show the superposition of the mask automatically
        segmented and the ground truth mask. Default: False
        :return: Return two parameters, stats and img_res. stats is a dictionary with some metrics like accuracy,
        precision and more. img_res is a three-channel image composed by the mask automatically segmented in the first
        channel, the ground truth mask in the second channel and the intersection of both mask in the last channel.
        """
        if type_img == "RGB":
            mask = self.rgb_image(path_img)

        elif type_img == "NIR":
            mask = self.nir_image(path_img)
        else:
            mask = self.thermal_image(path_img)

        target = io.imread(path_target)
        if target.ndim == 3:
            target = target[:, :, 0]

        target = np.uint8((target > 0)*255)

        img_res = np.zeros([mask.shape[0], mask.shape[1], 3], dtype=np.uint8)
        img_res[:, :, 0] = target
        img_res[:, :, 1] = mask
        img_res[:, :, 2] = np.uint8((np.logical_and(target > 0, mask > 0)) * 255)

        fn = np.sum((target - img_res[:, :, 2]) > 0)
        fp = np.sum((mask - img_res[:, :, 2]) > 0)
        tp = np.sum(img_res[:, :, 2] > 0)
        tn = np.sum(np.sum(img_res, axis=-1) == 0)
        iou = np.sum(img_res[:, :, 2] > 0) / np.sum(np.logical_or(target > 0, mask > 0))
        accuracy = (tp + tn) / (tp + tn + fn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        results = dict()
        results["iou"] = iou
        results["accuracy"] = accuracy
        results["precision"] = precision
        results["recall"] = recall
        results["fn"] = fn
        results["fp"] = fp
        results["tn"] = tn
        results["tp"] = tp

        if plot:
            plt.figure("Evaluation {0}".format(type_img))
            plt.imshow(img_res)
            plt.axis(False)
            plt.title("Result")
            plt.show()

        return results, img_res

    @staticmethod
    def __plot_segmentation(title, img, mask):
        """
        Method to plot results of the image segmentation

        :param title: Title of the plot
        :param img: Original image. Numpy array
        :param mask: Mask automatically segmented. Numpy array
        :return:
        """
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
        Image transformation method to improve contrast

        :param img: Grayscale image
        :param lim_inf_p: Number between 0 and 100
        :param lim_sup_p: Number between 0 and 100
        :return: Grayscale image
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
        This method generate a binary mask based on a thresholding approach

        :param img: Grayscale image. 2D numpy array
        :param dilatation_factor:
        :param erosion_factor:
        :param min_size: Minimum number of pixels of the avocado trees
        :return: The mask automatically segmented as a binary image. 2D numpy array
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
    #_ = img_obj.rgb_image("../data/RGB/Original/20200213_090858_240_8b.JPG", plot=True)
    #_ = img_obj.nir_image("../data/NIR/Original/IMG_00050.jpg", plot=True)
    #_ = img_obj.thermal_image("../data/FLIR/Original/20200217_105542_349_IR.TIFF", plot=True)
    stats, img_res = img_obj.eval_performance(
        "../data/NIR/Original/IMG_00050.jpg",
        "../data/NIR/Mask/IMG_00050.jpg",
        type_img="NIR",
        plot=True)
