import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import LabelEncoder


def string_to_list(s: str) -> list:
    """
    split string to list by "_"
    """
    split_str = s.split('_')

    int_list = list(map(int, split_str))

    return int_list