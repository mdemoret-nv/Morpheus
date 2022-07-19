import os

import morpheus


def get_data_file_path(data_filename: str):

    # First check if the path is relative
    if (os.path.isabs(data_filename)):
        # Already absolute, nothing to do
        return data_filename

    # See if the file exists.
    does_exist = os.path.exists(data_filename)

    if (not does_exist):
        # If it doesnt exist, then try to make it relative to the morpheus library root
        morpheus_root = os.path.dirname(morpheus.__file__)

        value_abs_to_root = os.path.join(morpheus_root, data_filename)

        # If the file relative to our package exists, use that instead
        if (os.path.exists(value_abs_to_root)):

            return value_abs_to_root

    return data_filename


def load_labels_file(labels_filename: str):
    with open(labels_filename, "r") as lf:
        return [x.strip() for x in lf.readlines()]
