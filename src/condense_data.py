'''Resaves the TIF files of the light field images or birefringent objects.'''
import tifffile
import os
import numpy as np

class ImageTransformer:
    def __init__(self, src_dir, dest_dir):
        self.src_dir = src_dir
        self.src_im_dir = os.path.join(src_dir, 'images')
        self.src_ob_dir = os.path.join(src_dir, 'objects')
        self.dest_dir = dest_dir
        self.dest_im_dir = os.path.join(dest_dir, 'images')
        self.dest_ob_dir = os.path.join(dest_dir, 'objects')

        # Ensure destination directory exists or create it
        if not os.path.exists(self.dest_im_dir):
            os.makedirs(self.dest_im_dir)
        if not os.path.exists(self.dest_ob_dir):
            os.makedirs(self.dest_ob_dir)

    def crop_object(self, crop_box):
        """
        Crop images in the last two dimensions based on the provided crop_box and save to dest_ob_dir.

        Parameters:
        - crop_box: tuple of four integers (left, upper, right, lower) defining the box to crop.
        """
        # List all TIFF files in the source directory of objects
        files = sorted(os.listdir(self.src_ob_dir))
        for file in files:
            # Construct full file paths
            src_path = os.path.join(self.src_ob_dir, file)
            dest_path = os.path.join(self.dest_ob_dir, file)

            # Read the TIFF file
            image = tifffile.imread(src_path)

            # Validate image shape
            if image.shape != (4, 8, 32, 32):
                print(f"Skipping {file} due to unexpected shape {image.shape}.")
                continue

            # Crop images in the last two dimensions
            left, upper, right, lower = crop_box
            cropped_image = image[:, :, upper:lower, left:right]

            # Save the cropped TIFF file to the destination directory
            tifffile.imwrite(dest_path, cropped_image)

        print(f"Successfully transformed and saved TIFF images from {self.src_ob_dir} to {self.dest_ob_dir}.")

    def resave_image(self):
        """
        Resave the light field images in a new directory.
        """
        # List all TIFF files in the source directory of images
        files = sorted(os.listdir(self.src_im_dir))
        for file in files:
            # Construct full file paths
            src_path = os.path.join(self.src_im_dir, file)
            dest_path = os.path.join(self.dest_im_dir, file)

            # Read the TIFF file
            image = tifffile.imread(src_path)

            # Save the TIFF file to the destination directory
            tifffile.imwrite(dest_path, image)

        print(f"Successfully transformed and saved TIFF images from {self.src_im_dir} to {self.dest_im_dir}.")

    def verify_crop_volume_empty(self, crop_box):
        """
        Crop images in the last two dimensions based on the provided crop_box.
        Then, check that the cropped region only contains zero birefringence values.

        Parameters:
        - crop_box: tuple of four integers (left, upper, right, lower) defining the box to crop.
        """
        num_nonempty = 0
        files = sorted(os.listdir(self.src_ob_dir))
        for file in files:
            # Construct full file paths
            src_path = os.path.join(self.src_ob_dir, file)

            # Read the TIFF file
            image = tifffile.imread(src_path)

            # Crop images in the last two dimensions
            left, upper, right, lower = crop_box
            cropped_image = image[:, :, upper:lower, left:right]
            
            # Check for nonzero birefringence values
            delta_n = cropped_image[0, ...]
            nonzero_bool = np.any(delta_n != 0)

            if nonzero_bool == True:
                count = np.count_nonzero(delta_n)
                num_nonempty += 1
                print(f"Volume {file} has {count} nonzero birefringence values.")
                # break

        print(f"Successfully checked that the cropped region {crop_box} from the objects in" +
              f"{self.src_ob_dir} have only zero birefringence values, except for {num_nonempty} objects.")

    def verify_volume_empty_after_zero_fill(self, crop_box):
        """
        Crop images in the last two dimensions based on the provided crop_box.
        Then, replace that region of the original volume with zero birefringence.
        Lastly, verify that the volume only contains zero birefringence values.

        Parameters:
        - crop_box: tuple of four integers (left, upper, right, lower)
                    defining the region to replace with zeros.
        """
        num_nonempty = 0
        files = sorted(os.listdir(self.src_ob_dir))
        for file in files:
            # Construct full file paths
            src_path = os.path.join(self.src_ob_dir, file)

            # Read the TIFF file
            image = tifffile.imread(src_path)

            delta_n_fill_zero = image[0, ...].copy()
            left, upper, right, lower = crop_box
            delta_n_fill_zero[:, upper:lower, left:right] = 0
            cropped_image = image[:, :, upper:lower, left:right]
            
            # Check for nonzero birefringence values
            nonzero_bool = np.any(delta_n_fill_zero != 0)

            if nonzero_bool == True:
                count = np.count_nonzero(delta_n_fill_zero)
                num_nonempty += 1
                print(f"Volume {file} has {count} nonzero birefringence values.")
                # break

        print(f"Successfully checked that the cropped region {crop_box} from the objects in" +
              f"{self.src_ob_dir} have only zero birefringence values, except for {num_nonempty} objects.")

    def verify_zdepth_empty(self):
        """
        Check if the top of bottom z depth layer has nonzero values.

        Parameters:
        - crop_box: tuple of four integers (left, upper, right, lower)
                    defining the region to replace with zeros.
        """
        num_nonempty = 0
        files = sorted(os.listdir(self.src_ob_dir))
        for file in files:
            # Construct full file paths
            src_path = os.path.join(self.src_ob_dir, file)

            # Read the TIFF file
            image = tifffile.imread(src_path)

            delta_n_top = image[0, 0, :, :]
            delta_n_bottom = image[0, -1, :, :]
            
            # Check for nonzero birefringence values
            nonzero_bool_top = np.any(delta_n_top != 0)

            if nonzero_bool_top == True:
                count = np.count_nonzero(delta_n_top)
                num_nonempty += 1
                print(f"Volume {file} top layer has {count} nonzero birefringence values.")
                # break

            # Check for nonzero birefringence values
            nonzero_bool_bottom = np.any(delta_n_bottom != 0)

            if nonzero_bool_bottom == True:
                count = np.count_nonzero(delta_n_bottom)
                num_nonempty += 1
                print(f"Volume {file} bottom layer has {count} nonzero birefringence values.")
                # break

        print(f"Successfully checked that the objects in" +
              f"{self.src_ob_dir} have only zero birefringence values, " +
              f"except for {num_nonempty} top or bottom layers of objects.")

if __name__ == '__main__':
    transformer = ImageTransformer("/mnt/efs/shared_data/restorators/spheres", "/mnt/efs/shared_data/restorators/spheres_11by11")
    object_region = (10, 10, 21, 21)
    transformer.verify_volume_empty_after_zero_fill(object_region)
    transformer.crop_object(object_region)
    transformer.resave_image()