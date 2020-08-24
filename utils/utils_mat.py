import os
import json
import scipy.io as spio
import pandas as pd


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return dict_to_nonedict(_check_keys(data))

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


def mat2json(mat_path=None, filepath = None):
    """
    Converts .mat file to .json and writes new file
    Parameters
    ----------
    mat_path: Str
        path/filename .mat存放路径
    filepath: Str
        如果需要保存成json, 添加这一路径. 否则不保存
    Returns
        返回转化的字典
    -------
    None
    Examples
    --------
    >>> mat2json(blah blah)
    """

    matlabFile = loadmat(mat_path)
    #pop all those dumb fields that don't let you jsonize file
    matlabFile.pop('__header__')
    matlabFile.pop('__version__')
    matlabFile.pop('__globals__')
    #jsonize the file - orientation is 'index'
    matlabFile = pd.Series(matlabFile).to_json()

    if filepath:
        json_path = os.path.splitext(os.path.split(mat_path)[1])[0] + '.json'
        with open(json_path, 'w') as f:
                f.write(matlabFile)
    return matlabFile