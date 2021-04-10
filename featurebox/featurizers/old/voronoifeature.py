# -*- coding: utf-8 -*-

# @Time   : 2019/5/23 20:40
# @Author : Administrator
# @Project : feature_preparation
# @FileName: backforward.py
# @Software: PyCharm

""""""

import numpy as np
from pymatgen.analysis.local_env import VoronoiNN


def _count_i(vo_data, site_ele, mess='face_dist'):
    nei_all_ele = []
    for elei, ele_name in zip(vo_data, site_ele):
        nei_all = []
        for nei in elei:
            nei_name = nei['site'].specie.name
            data_ = nei['poly_info'][mess]
            nei_all.append([nei_name, data_])
        nei_all = np.array(nei_all)
        nei_all_ele.append((ele_name, nei_all))
    return nei_all_ele


def _count_voronoinn(s1, mess='face_dist'):
    """

    :type s1: pymatgen structre type
    """
    vo = VoronoiNN(tol=0.01,
                   allow_pathological=False, weight='solid_angle',
                   extra_nn_info=True, compute_adj_neighbors=True)
    vo_data = vo.get_all_nn_info(s1)
    site_ele = [i.specie.name for i in s1.sites]
    nei_all_ele = _count_i(vo_data, site_ele, mess=mess)
    a_dict = {}
    for i in s1.types_of_specie:
        a_list = []
        for j in nei_all_ele:
            if j[0] == i.name:
                a_list.extend(j[1])
        a_list = np.array(a_list)
        a_dict[i] = a_list
    result_0 = a_dict[s1.types_of_specie[0]]
    number = []
    for ele in s1.types_of_specie:
        b_index = np.where(result_0[:, 0] == ele.name)[0]
        int_ = result_0[b_index, 1:].astype(float)
        number.append(np.sum(int_) / s1.num_sites)
    return number


def count_voronoinn(structures, mess='area'):
    result = [_count_voronoinn(s1, mess=mess) for i, s1 in enumerate(structures)]
    data = np.array(result)
    return data
