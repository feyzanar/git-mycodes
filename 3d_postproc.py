# napari-skimage-regionprops
# or go to Tools>Measurement tables>regionprops or object features/properties(scikit-image, nsr)
from skimage.measure import regionprops_table
from napari_skimage_regionprops import visualize_measurement_on_labels, regionprops_table, add_table, get_table

regionprops_table(mask,
                  image[:,0,:,:],
                  size = True,
                  intensity = True,
                  napari_viewer = viewer)



