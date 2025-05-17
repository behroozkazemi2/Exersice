# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import csv
import matplotlib

import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
from typing import Set, Tuple, List

NoneType = type(None)


# <editor-fold desc="Exercise 1">
# You can copy this code to your personal pipeline project or execute it here.
def id_to_fruit(fruit_id: int, fruitsArray: List[str]) -> str:
    """
    This method returns the fruit name by getting the string at a specific index of the set.

    :param fruit_id: The id of the fruit to get
    :param fruits: The set of fruits to choose the id from
    :return: The string corrosponding to the index ``fruit_id``

    **This method is part of a series of debugging exercises.**
    **Each Python method of this series contains bug that needs to be found.**

    | ``1   It does not print the fruit at the correct index, why is the returned result wrong?`` // sort will be change
    | ``2   How could this be fixed?`` // use sort first to sort the set then find the index or use array with index for loop

    This example demonstrates the issue:
    name1, name3 and name4 are expected to correspond to the strings at the indices 1, 3, and 4:
    'orange', 'kiwi' and 'strawberry'..

    >>> name1 = id_to_fruit(1, {"apple", "orange", "melon", "kiwi", "strawberry"})
    >>> name3 = id_to_fruit(3, {"apple", "orange", "melon", "kiwi", "strawberry"})
    >>> name4 = id_to_fruit(4, {"apple", "orange", "melon", "kiwi", "strawberry"})
    """
    # for i in range(len(fruitsArray)):
    #     if i == fruit_id:
    #         return fruitsArray[i]
    return fruitsArray[fruit_id]



if __name__ == "__main__":
    name1 = id_to_fruit(
        1,
        ["apple", "orange", "melon", "kiwi", "strawberry"]
    )
    print(f" name1 {name1}")

    name3 = id_to_fruit(3,
                        ["apple", "orange", "melon", "kiwi", "strawberry"]

                        )
    print(f" name3 {name3}")

    name4 = id_to_fruit(
        4,
        ["apple", "orange", "melon", "kiwi", "strawberry"]
    )
    print(f" name4 {name4}")

    print("OFF")
# </editor-fold>
