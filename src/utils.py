import jax.numpy as np
import jax
import numpy as onp
import orix
import meshio
import pickle
import time
import os
import matplotlib.pyplot as plt


def walltime(func):
    def wrapper(*list_args, **keyword_wargs):
        start_time = time.time()
        func(*list_args, **keyword_wargs)
        end_time = time.time()
        time_elapsed = end_time - start_time
        print(f"Time elapsed {time_elapsed}") 
        return time_elapsed
    return wrapper
