# Byte-swapping #######################################################################################################
# Introduction to byte ordering and ndarrays ##########################################################################
import numpy as np
import os, sys

big_end_str = chr(0) + chr(1) + chr(3) + chr(2)
big_end_str


big_end_arr = np.ndarray(shape=(2,),dtype='>i2', buffer=big_end_str)
big_end_arr[0]

big_end_arr[1]

little_end_u4 = np.ndarray(shape=(1,),dtype='<u4', buffer=big_end_str)
little_end_u4[0] == 1 * 256**1 + 3 * 256**2 + 2 * 256**3


# Changing byte ordering ##############################################################################################
# Data and dtype endianness don’t match, change dtype to match data ###################################################

wrong_end_dtype_arr = np.ndarray(shape=(2,),dtype='<i2', buffer=big_end_str)
wrong_end_dtype_arr[0]

fixed_end_dtype_arr = wrong_end_dtype_arr.newbyteorder()
fixed_end_dtype_arr[0]

fixed_end_dtype_arr.tobytes() == big_end_str

# Data and type endianness don’t match, change data to match dtype ####################################################
fixed_end_mem_arr = wrong_end_dtype_arr.byteswap()
fixed_end_mem_arr[0]

fixed_end_mem_arr.tobytes() == big_end_str


# Data and dtype endianness match, swap data and dtype ################################################################
swapped_end_arr = big_end_arr.byteswap().newbyteorder()
swapped_end_arr[0]

swapped_end_arr.tobytes() == big_end_str

swapped_end_arr = big_end_arr.astype('<i2')
swapped_end_arr[0]

swapped_end_arr.tobytes() == big_end_str