# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib

matplotlib.use('TkAgg')

import numpy as np
NoneType = type(None)

# <editor-fold desc="Exercise 2">
# You can copy this code to your personal pipeline project or execute it here.
def swap(coords: np.ndarray):
    """
    This method will flip the x and y coordinates in the coords array.

    :param coords: A numpy array of bounding box coordinates with shape [n,5] in format:
        ::

            [[x11, y11, x12, y12, classid1],
             [x21, y21, x22, y22, classid2],
             ...
             [xn1, yn1, xn2, yn2, classid3]]

    :return: The new numpy array where the x and y coordinates are flipped.

    **This method is part of a series of debugging exercises.**
    **Each Python method of this series contains bug that needs to be found.**

    | ``1   Can you spot the obvious error?``
    | ``2   After fixing the obvious error it is still wrong, how can this be fixed?``

    >>> import numpy as np
    >>> coords = np.array([[10, 5, 15, 6, 0],
    ...                    [11, 3, 13, 6, 0],
    ...                    [5, 3, 13, 6, 1],
    ...                    [4, 4, 13, 6, 1],
    ...                    [6, 5, 13, 16, 1]])
    >>> swapped_coords = swap(coords)

    The example demonstrates the issue. The returned swapped_coords are expected to have swapped
    x and y coordinates in each of the rows.
    """
    # coords_copy = coords.copy()
    # coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3], = coords_copy[:, 1], coords_copy[:, 0], coords_copy[:, 3], coords_copy[:, 2]

    for row in coords:
        row[0] , row[1] = row[1], row[0]
        row[2] , row[3] = row[3], row[2]

    return coords
if __name__ == "__main__":
    coords = np.array([[10, 5, 15, 6, 0],
                       [11, 3, 13, 6, 0],
                       [5, 3, 13, 6, 1],
                       [4, 4, 13, 6, 1],
                       [6, 5, 13, 16, 1]])
    swapped_coords = swap(coords)

    print(f"{swapped_coords}")
# </editor-fold>
