# -*- coding: utf-8 -*-
"""
Created on Thu May 27 10:20:52 2021

@author: GL
"""

import os

import numpy as np
import pandas as pd
import torch

from featurebox.featurizers.generator import GraphGenerator, MGEDataLoader
from featurebox.models.cgcnn import CrystalGraphConvNet
from featurebox.models.flow import BaseLearning
from featurebox.utils.general import train_test

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    # PATH = "/home/iap13/wcx/featurebox/test/test_gl"
    # # PATH = os.getcwd()
    # from mgetool.imports import BatchFile
    #
    # bf = BatchFile(os.path.join(PATH, "data"), suffix='cif')
    # f = bf.merge()
    # os.chdir(PATH)
    #
    # df_atomic_attributes = pd.read_csv(r'df_atomic_attributes_remake.csv', index_col=0)
    # atom_feature = df_atomic_attributes.T
    # atom_feature = atom_feature.drop(['Name', 'Symbol'], axis=1)
    #
    # atom_feature = atom_feature.fillna(0)
    # icsdid = pd.read_csv('caogao.csv', index_col=0)
    #
    # index = set(icsdid.index)
    #
    # f_new = []  # 有在icsdid里找不到的，删去二十几个##
    # for fi in f:
    #     if int(re.findall(r'\d+', os.path.split(fi)[-1])[0]) in index:
    #         f_new.append(fi)
    #
    # data = [Structure.from_file(i) for i in f_new]
    # marks = [int(re.findall(r'\d+', os.path.split(i)[-1])[0]) for i in f_new]
    #
    # y = [icsdid.loc[b, "Tc"] for b in marks]
    #
    # tmps = AtomTableMap(search_tp="name", tablename=atom_feature)
    # ce = CheckElements.from_pymatgen_structures()
    # checked_data = ce.check(data)
    # y = np.array(y)[ce.passed_idx()]
    # gt = CrystalGraph(n_jobs=1, atom_converter=tmps)
    # in_data = gt.transform(checked_data)
    #
    # pd.to_pickle((in_data, y), "in_data.pkl_pd")
    ###########
    in_data, y = pd.read_pickle("in_data.pkl_pd")
    #
    y = y.astype(np.float32)
    y = torch.from_numpy(y)
    X_train, y_train, X_test, y_test = train_test(*in_data, y, random_state=0)
    gen = GraphGenerator(*X_train, targets=y_train, print_data_size=True)
    test_gen = GraphGenerator(*X_test, targets=y_test)

    loader1 = MGEDataLoader(
        dataset=gen,
        batch_size=2000,
        shuffle=True,
        num_workers=0, )
    loader2 = MGEDataLoader(
        dataset=test_gen,
        batch_size=2000,
        shuffle=True,
        num_workers=0, )

    model = CrystalGraphConvNet(atom_fea_len=58, nbr_fea_len=4,
                                # state_fea_len=2,
                                inner_atom_fea_len=300, n_conv=3, h_fea_len=(256, 128, 64), n_h=2, )
    bl = BaseLearning(model, loader1, test_loader=loader2, device="cuda:0",
                      clf=False, loss_threshold=2.0,
                      milestones=[50, 70, 80]
                      )

    bl.run(100)
    torch.save(bl.model.state_dict(), './parameter_n_sgt.pkl')
