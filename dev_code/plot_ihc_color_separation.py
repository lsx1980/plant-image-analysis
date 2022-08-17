"""
==============================================
Immunohistochemical staining colors separation
==============================================

Color deconvolution consists of the separation of features by their colors.

In this example we separate the immunohistochemical (IHC) staining from the
hematoxylin counterstaining. The separation is achieved with the method
described in [1]_, known as "color deconvolution".

The IHC staining expression of the FHL2 protein is here revealed with
Diaminobenzidine (DAB) which gives a brown color.


.. [1] A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical
       staining by color deconvolution.," Analytical and quantitative
       cytology and histology / the International Academy of Cytology [and]
       American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001.
       
"""
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave, imread

from skimage import data
from skimage.color import rgb2hed
from skimage import img_as_ubyte, img_as_float32

from matplotlib.colors import LinearSegmentedColormap
import cv2
# Create an artificial color close to the orginal one
cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['white', 'navy'])
cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white',
                                             'saddlebrown'])
cmap_eosin = LinearSegmentedColormap.from_list('mycmap', ['darkviolet',
                                               'white'])

'''
# load the input image 
image = cv2.imread("/home/suxing/example/plant_test/seeds/test/EX12.png")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

(H, W) = image.shape[:2]
'''

image = imread("/home/suxing/example/plant_test/seeds/test/EX12_masked.png")

(H, W, N) = image.shape

ihc_rgb = image

blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),swapRB=False, crop=False)

#ihc_rgb = data.immunohistochemistry()
ihc_hed = rgb2hed(ihc_rgb)

#img_uint8 = img.astype(np.uint8)

result_file = ("/home/suxing/example/plant_test/seeds/test/hed.png")
imsave(result_file, img_as_float32(ihc_hed))

#cv2.imwrite(result_file, ihc_hed.astype(np.uint8))

'''
fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(ihc_rgb)
ax[0].set_title("Original image")

ax[1].imshow(ihc_hed[:, :, 0], cmap=cmap_hema)
ax[1].set_title("Hematoxylin")

ax[2].imshow(ihc_hed[:, :, 1], cmap=cmap_eosin)
ax[2].set_title("Eosin")

ax[3].imshow(ihc_hed[:, :, 2], cmap=cmap_dab)
ax[3].set_title("DAB")

for a in ax.ravel():
    a.axis('off')

fig.tight_layout()
'''

######################################################################
# Now we can easily manipulate the hematoxylin and DAB "channels":

import numpy as np

from skimage.exposure import rescale_intensity

# Rescale hematoxylin and DAB signals and give them a fluorescence look
h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))
d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1))
zdh = np.dstack((np.zeros_like(h), d, h))
'''
fig = plt.figure()
axis = plt.subplot(1, 1, 1, sharex=ax[0], sharey=ax[0])
axis.imshow(zdh)
axis.set_title("Stain separated image (rescaled)")
axis.axis('off')
plt.show()
'''
'''
result_file = ("/home/suxing/example/plant_test/seeds/test/zdh.png")
imsave(result_file, img_as_float32(zdh))

result_file = ("/home/suxing/example/plant_test/seeds/test/hed_0.png")
imsave(result_file, img_as_float32(ihc_hed[:, :, 0]))

result_file = ("/home/suxing/example/plant_test/seeds/test/hed_1.png")
imsave(result_file, img_as_float32(ihc_hed[:, :, 1]))

result_file = ("/home/suxing/example/plant_test/seeds/test/hed_2.png")
imsave(result_file, img_as_float32(ihc_hed[:, :, 2]))
'''

# convert the image to grayscale and blur it slightly
gray = cv2.cvtColor(ihc_rgb, cv2.COLOR_BGR2GRAY)


blurred = cv2.GaussianBlur(gray, (7, 7), 0)


thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 81, 10)

result_file = ("/home/suxing/example/plant_test/seeds/test/zdh_2_thresh.png")
imsave(result_file, thresh)


