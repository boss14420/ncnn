// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef LAYER_CONVOLUTION1D_VULKAN_H
#define LAYER_CONVOLUTION1D_VULKAN_H

#include "convolution1d.h"

namespace ncnn {

class Convolution1D_vulkan : virtual public Convolution1D
{
public:
    Convolution1D_vulkan();

    virtual int create_pipeline(const Option& opt) override;
    virtual int destroy_pipeline(const Option& opt) override;

    virtual int upload_model(VkTransfer& cmd, const Option& opt) override;

    using Convolution1D::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob,
                        VkCompute& cmd, const Option& opt) const override;
    virtual int forward(const VkImageMat& bottom_blob, VkImageMat& top_blob,
                        VkCompute& cmd, const Option& opt) const override;

public:
    ncnn::Layer* padding;

    Mat weight_data_packed;
    Mat bias_data_packed;

    VkMat weight_data_gpu;
    VkMat bias_data_gpu;

    VkImageMat weight_data_gpu_image;
    VkImageMat bias_data_gpu_image;

    Pipeline* pipeline_convolution;
    // using share memory, generic 
    Pipeline* pipeline_convolution_local_memory;
    // using share memory, stride = 1, dilation = 1
    Pipeline* pipeline_convolution_s1d1;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION1D_VULKAN_H
