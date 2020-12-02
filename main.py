from processing.segmentation import ImageSegmentation

# First, create a ImageSegmentation object

img_obj = ImageSegmentation()

# Then, for RGB segmentation use the following method:

_ = img_obj.rgb_image("./data/RGB/Original/20200213_090858_240_8b.JPG", plot=False)

# For NIR segmentation use:

_ = img_obj.nir_image("./data/NIR/Original/IMG_00050.jpg", plot=False)

# For thermal image segmentation use:

_ = img_obj.thermal_image("./data/FLIR/Original/20200217_105542_349_IR.TIFF", plot=False)

# To eval the performance of the segmentation method use:

stats, img_res = img_obj.eval_performance(
    "./data/NIR/Original/IMG_00050.jpg",
    "./data/NIR/Mask/IMG_00050.jpg",
    type_img="NIR",
    plot=True)

print("Results: ", stats)
