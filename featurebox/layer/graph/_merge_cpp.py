import imp
import os
import platform
import sys
from importlib import util
from pathlib import Path

import importlib
import torch
import torch.utils.cpp_extension


def import_module_from_library(module_name, path, is_python_module):
    file, path, description = imp.find_module(module_name, [path])
    # Close the .so file after load.
    with file:
        if is_python_module:
            return imp.load_module(module_name, file, path, description)
        else:
            torch.ops.load_library(path)

source = """
#include <iostream>
#include <torch/extension.h>


using namespace torch::indexing;
using namespace std;

inline torch::Tensor merge_idxi_mean(const torch::Tensor &x, const torch::Tensor &idxi) {

    return at::mean(x.index({idxi}), 0, true);
}

inline torch::Tensor merge_idxi_sum(const torch::Tensor &x, const torch::Tensor &idxi) {

    return at::sum(x.index({idxi}), 0, true);

}

inline torch::Tensor merge_idxi_max(const torch::Tensor &x, const torch::Tensor &idxi) {

    return std::get<0>(at::max(x.index({idxi}), 0, true));

}

inline torch::Tensor merge_idxi_min(const torch::Tensor &x, const torch::Tensor &idxi) {

    return std::get<0>(at::min(x.index({idxi}), 0, true));

}


torch::Tensor merge_idx_mean(const torch::Tensor &x, const std::vector<torch::Tensor> &idx) {
    std::vector<torch::Tensor> rx;
    for (const torch::Tensor &item : idx) {
        torch::Tensor rxi = at::mean(x.index({item}), 0, true);
        rx.push_back(rxi);
    }
    return at::cat(rx, 0);}

torch::Tensor merge_idx_sum(const torch::Tensor &x, const std::vector<torch::Tensor> &idx) {
    std::vector<torch::Tensor> rx;
    for (const torch::Tensor &item : idx) {
        torch::Tensor rxi = at::sum(x.index({item}), 0, true);
        rx.push_back(rxi);
    }
    return at::cat(rx, 0);}

torch::Tensor merge_idx_max(const torch::Tensor &x, const std::vector<torch::Tensor> &idx) {
    std::vector<torch::Tensor> rx;
    for (const torch::Tensor &item : idx) {
        torch::Tensor rxi = std::get<0>(at::max(x.index({item}), 0, true));
        rx.push_back(rxi);
    }
    return at::cat(rx, 0);}

torch::Tensor merge_idx_min(const torch::Tensor &x, const std::vector<torch::Tensor> &idx) {
    std::vector<torch::Tensor> rx;
    for (const torch::Tensor &item : idx) {
        torch::Tensor rxi = std::get<0>(at::min(x.index({item}), 0, true));
        rx.push_back(rxi);
    }
    return at::cat(rx, 0);}

torch::Tensor merge_idx(const torch::Tensor &x, const std::vector<torch::Tensor> &idx, const string& merge_type="mean") {
    std::vector<torch::Tensor> rx;
    if (merge_type == "mean"){
    for (const torch::Tensor &item : idx) {
        torch::Tensor rxi = at::mean(x.index({item}), 0, true);
        rx.push_back(rxi);
    }}
    else if (merge_type == "sum"){
        for (const torch::Tensor &item : idx) {
            torch::Tensor rxi = at::sum(x.index({item}), 0, true);
            rx.push_back(rxi);
        }}

    else if (merge_type == "max"){
        for (const torch::Tensor &item : idx) {
            torch::Tensor rxi = std::get<0>(at::max(x.index({item}), 0, true));
            rx.push_back(rxi);
        }}
    else {
        for (const torch::Tensor &item : idx) {
            torch::Tensor rxi = std::get<0>(at::min(x.index({item}), 0, true));
            rx.push_back(rxi);
        }}

//    auto rxx = torch::TensorList(rx);
    return at::cat(rx, 0);
}

"""

name = "segment_method"

if platform.system() == "Windows":
    ext = "dll"
else:
    ext = "so"

name_dir = "cache_" + name
MODULE_DIR = Path(__file__).parent.absolute()
MODULE_DIR_NAME_DIR = MODULE_DIR / name_dir

if os.path.isdir(MODULE_DIR_NAME_DIR) and os.path.isfile(MODULE_DIR_NAME_DIR / "{}.{}".format(name, ext))\
        :
    mod = import_module_from_library(name, MODULE_DIR_NAME_DIR, True)

else:
    if not os.path.isdir(MODULE_DIR_NAME_DIR):
        os.mkdir(MODULE_DIR_NAME_DIR)
    mod = torch.utils.cpp_extension.load_inline(
        name=name,
        cpp_sources=source,
        verbose=True,
        build_directory=MODULE_DIR_NAME_DIR,
        is_python_module=True,
        functions=["merge_idx", "merge_idx_mean", "merge_idx_max", "merge_idx_sum", "merge_idx_min"]
    )

if __name__ == '__main__':
    inputs = torch.tensor([1.2, 3, 4, 5, 6, 0.7, 9], requires_grad=True, device="cpu")
    node_bond_idx = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    # b = mod.merge_idx(inputs, node_bond_idx, "mean")
