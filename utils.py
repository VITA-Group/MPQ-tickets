import torch
import numpy as np
import os

class safe_save:
    @classmethod
    def check_path(cls, file_path, log=None):
        # get directory of file in file_path
        dir_path = cls.get_path_of_file(file_path)
        if not os.path.exists(dir_path):
            message = f'A folder named {dir_path} is created to avoid save error'
            if log is not None:
                log(message)
            else:
                print(message)
            # create directory named dir_path
            os.makedirs(dir_path)

    @classmethod
    def get_path_of_file(cls, file_path):
        return os.path.join(os.getcwd(), file_path.rsplit("/", 1)[0])

    @classmethod
    def torch_save(cls, data, path, log=None):
        cls.check_path(path, log)
        torch.save(data, path)

    @classmethod
    def np_save(cls, data, path, log=None):
        cls.check_path(path, log)
        np.save(path, data)

    @classmethod
    def np_load(cls, path):
        return np.load(path, allow_pickle=True).item()


def stats_mask(mask):
    bits = [32, 16, 8, 4, 2, 1, 0]
    result = {}
    for bit in bits:
        result[bit] = 0
    for key, value in mask.items():
        for bit in bits:
            count = torch.sum(value == bit).item()
            if count > 0:
                result[bit] += count
    for key, value in result.items():
        if value > 0:
            print(f'{key}:{value}')
