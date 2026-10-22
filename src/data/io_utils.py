"""
io_utils.py
Functions for reading and writing 3D TIFF images using SimpleITK
"""


import numpy as np
import SimpleITK as sitk
import os
import sys
import gc
import tqdm
#import napari
import matplotlib.pyplot as plt

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm


def read_tif_image(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    reader = sitk.ImageFileReader()
    reader.SetImageIO("TIFFImageIO")
    reader.SetFileName(file_path)
    return reader.Execute()

def read_nrrd_image(chemin):
    if not os.path.exists(chemin):
        raise FileNotFoundError(f"Le fichier {chemin} n'existe pas")
    reader=sitk.ImageFileReader()
    reader.SetImageIO("NrrdImageIO")
    reader.SetFileName(chemin)
    return reader.Execute()

def save_image(image, file_path):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(file_path)
    writer.Execute(image)
    return


def show_image_info(image):
    if isinstance(image, sitk.Image):
        print("Image Information:")
        print(f"   - Dimensions (X,Y,Z): {image.GetSize()}")
        print(f"   - Spacing: {image.GetSpacing()}")
        print(f"   - Pixel Type: {image.GetPixelIDTypeAsString()}")
        print(f"   - Direction: {image.GetDirection()}")
        print(f"   - Origin: {image.GetOrigin()}")
    else:
        print("Error: This is not a SimpleITK image")
    return


def show_array_info(array):
    if isinstance(array, np.ndarray):
        print("Array Information:")
        print(f"   - Shape: {array.shape}")
        print(f"   - Number of dimensions: {array.ndim}")
        print(f"   - Data type: {array.dtype}")
    else:
        print("Error: This is not a numpy array")
    return


def get_image_info_dictionary(image):
    if isinstance(image, sitk.Image):
        info_dict = {
            "dimensions": image.GetSize(),
            "spacing": image.GetSpacing(),
            "direction": image.GetDirection(),
            "origin": image.GetOrigin()
        }
    elif isinstance(image, np.ndarray):
        info_dict = {
            "dimensions": image.ndim,
            "shape": image.shape,
        }
    else:
        info_dict = None
    return info_dict


def convert_image_to_array_xyz(image):
    return sitk.GetArrayFromImage(image).transpose((2, 1, 0))


def convert_image_to_array(image):
    return sitk.GetArrayFromImage(image)


def convert_array_to_image(array):
    return sitk.GetImageFromArray(array)


def convert_array_to_image_with_reference(array, image_ref):
    if isinstance(array, np.ndarray) and isinstance(image_ref, sitk.Image):
        return sitk.GetImageFromArray(array).CopyInformation(image_ref)
    elif isinstance(array, np.ndarray):
        print("Reference image invalid. Returning array as image without reference information")
        return sitk.GetImageFromArray(array)
    elif isinstance(image_ref, sitk.Image):
        print("Array to convert is invalid. Returning reference image")
        return image_ref
    else:
        print("Both array and reference image are invalid")
        return None


def save_array(array, file_path):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(file_path)
    writer.Execute(convert_array_to_image(array))
    return


def get_volume_from_image(image, z_start, z_end, y_start=None, y_end=None, x_start=None, x_end=None):
    if isinstance(image, sitk.Image):
        array = sitk.GetArrayFromImage(image)
    else:
        array = image

    if y_start is None:
        y_start = 0
    if y_end is None:
        y_end = array.shape[1]
    if x_start is None:
        x_start = 0
    if x_end is None:
        x_end = array.shape[2]

    volume_array = array[z_start:z_end, y_start:y_end, x_start:x_end]
    volume = sitk.GetImageFromArray(volume_array)

    if isinstance(image, sitk.Image):
        volume.SetSpacing(image.GetSpacing())
        volume.SetDirection(image.GetDirection())

        new_origin = (
            image.GetOrigin()[0] + x_start * image.GetSpacing()[0],
            image.GetOrigin()[1] + y_start * image.GetSpacing()[1],
            image.GetOrigin()[2] + z_start * image.GetSpacing()[2]
        )
        volume.SetOrigin(new_origin)

    return volume


def get_volume_from_array(array, z_start, z_end, y_start=None, y_end=None, x_start=None, x_end=None):
    if y_start is None:
        y_start = 0
    if y_end is None:
        y_end = array.shape[1]
    if x_start is None:
        x_start = 0
    if x_end is None:
        x_end = array.shape[2]
    return array[z_start:z_end, y_start:y_end, x_start:x_end]


def get_slices_from_image(image, z_start, z_end):
    return get_volume_from_image(image, z_start, z_end)


def get_slices_from_array(array, z_start, z_end):
    return get_volume_from_array(array, z_start, z_end)


def get_one_slice_from_image(image, position):
    if isinstance(image, sitk.Image):
        array = convert_image_to_array(image)
    else:
        array = image
    return sitk.GetImageFromArray(array[position, :, :])


def get_one_slice_from_array(array, position):
    if isinstance(array, sitk.Image):
        array = convert_image_to_array(array)
    return sitk.GetImageFromArray(array[position, :, :])


def show_one_slice_2d(image, position, title="Slice", cmap='gray', figsize=(10, 10)):
    slice_img = get_one_slice_from_image(image, position)
    if isinstance(slice_img, sitk.Image):
        slice_img = convert_image_to_array(slice_img)
    plt.figure(figsize=figsize)
    plt.imshow(slice_img, cmap=cmap)
    plt.colorbar(label='Intensity')
    plt.title(f"{title} number {position}")
    plt.xlabel("X (column)")
    plt.ylabel("Y (row)")
    plt.show()
    return


def progress_bar(iterable, description='Progress'):
    return tqdm(iterable, desc=description)


def check_image_mask_consistency(image_path, mask_path):
    image = read_tif_image(image_path)
    mask = read_tif_image(mask_path)
    
    image_size = image.GetSize()
    mask_size = mask.GetSize()
    
    if image_size != mask_size:
        print(f"WARNING: Dimensions mismatch!")
        print(f"  Image dimensions (X,Y,Z): {image_size}")
        print(f"  Mask dimensions (X,Y,Z): {mask_size}")
        return False
    else:
        print(f"Dimensions match: {image_size}")
        return True


def get_array_shape_from_sitk(image):
    return convert_image_to_array(image).shape