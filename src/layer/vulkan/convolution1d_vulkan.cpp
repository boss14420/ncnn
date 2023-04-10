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

#include "convolution1d_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

static constexpr int GROUP_SIZE_X_LOCAL_MEMORY = 8;
static constexpr int GROUP_SIZE_Y_LOCAL_MEMORY = 8;

namespace ncnn {

Convolution1D_vulkan::Convolution1D_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    padding = nullptr;

    pipeline_convolution = nullptr;
    pipeline_convolution_s1d1 = nullptr;
    pipeline_convolution_local_memory = nullptr;
}

int Convolution1D_vulkan::create_pipeline(const Option& _opt)
{
    if (dynamic_weight)
    {
        support_vulkan = false;
        support_image_storage = false;
        return 0;
    }

    Option opt = _opt;
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int num_input = weight_data_size / kernel_w / num_output;

    // the shape after padding
    Mat shape_bordered;
    if (shape.dims != 0)
    {
        if (pad_left > 0 || pad_right > 0)
        {
            shape_bordered = Mat(shape.w + pad_left + pad_right,
                                 shape.h, shape.c, nullptr);
        }
        else if ((pad_left == -233 && pad_right == -233)
                 || (pad_left == -234 && pad_right == -234))
        {
            const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

            int wpad = kernel_extent_w + (shape.w - 1) / stride_w * stride_w - shape.w;
            if (wpad > 0)
            {
                shape_bordered = Mat(shape.w + wpad, shape.h, shape.c, nullptr);
            }
        }
        else
        {
            shape_bordered = shape;
        }
        shape_bordered.dims = 2 + (shape.c > 1);
    }

    int elempack = opt.use_shader_pack8 && num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;

    size_t elemsize;
    size_t out_elemsize;
    if (opt.use_fp16_storage)
    {
        elemsize = elempack * 2u;
        out_elemsize = out_elempack * 2u;
    }
    else if (opt.use_fp16_packed)
    {
        elemsize = elempack == 1 ? 4u : elempack * 2u;
        out_elemsize = out_elempack == 1 ? 4u : out_elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
        out_elemsize = out_elempack * 4u;
    }

    Mat shape_bordered_packed;
    if (shape_bordered.dims == 2)
        shape_bordered_packed = Mat(shape_bordered.w, num_input / elempack,
                                    nullptr, elemsize, elempack);

    Mat out_shape_packed;
    if (out_shape.dims == 2)
        out_shape_packed = Mat(out_shape.w, num_output / out_elempack,
                               nullptr, out_elemsize, out_elempack);
    else if (out_shape.dims == 3 && out_shape.c == 1)
        out_shape_packed = Mat(out_shape.w, num_output / out_elempack,
                               1, nullptr, out_elemsize, out_elempack);

    {
        padding = ncnn::create_layer(ncnn::LayerType::Padding);
        padding->vkdev = vkdev;

        padding->bottom_shapes.resize(1);
        padding->bottom_shapes[0] = shape;
        padding->top_shapes.resize(1);
        padding->top_shapes[0] = shape_bordered;

        ncnn::ParamDict pd;
        pd.set(0, 0);
        pd.set(1, 0);
        pd.set(2, pad_left);
        pd.set(3, pad_right);
        pd.set(4, 0);
        pd.set(5, pad_value);

        padding->load_param(pd);

        padding->create_pipeline(opt);
    }

    ////////////////////////////////////////////////////////////////////////
    /// Packing

    Mat weight_data_r2 = weight_data.reshape(kernel_w, num_input, num_output);

    weight_data_packed.create(kernel_w, num_input / elempack,
                              num_output / out_elempack,
                              (size_t)4 * elempack * out_elempack,
                              elempack * out_elempack);

    for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
    {
        float* g00 = weight_data_packed.channel(q / out_elempack);

        for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
        {
            for (int k = 0; k < kernel_w; k++)
            {
                for (int i = 0; i < out_elempack; i++)
                {
                    const Mat k0 = weight_data_r2.channel(q + i);

                    for (int j = 0; j < elempack; j++)
                    {
                        const float* k00 = k0.row(p + j);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }
    if (bias_term)
    {
        convert_packing(bias_data, bias_data_packed, out_elempack, opt);
    }

    ////////////////////////////////////////////////////////////////////////
    /// Vk pipeline setup


    // when s == d == 1
    if (stride_w == 1 && dilation_w == 1 && opt.use_shader_local_memory
        && elempack == 4 && out_elempack <= 4)
    {
        std::vector<vk_specialization_type> specializations(5 + 8);
        specializations[0].i = kernel_w;
        specializations[1].i = bias_term;
        specializations[2].i = activation_type;
        specializations[3].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
        specializations[4].f = activation_params.w == 2 ? activation_params[1] : 0.f;
        specializations[4 + 0].i = shape_bordered_packed.w;
        specializations[4 + 1].i = shape_bordered_packed.h;
        specializations[4 + 2].i = shape_bordered_packed.c;
        specializations[4 + 3].i = shape_bordered_packed.cstep;
        specializations[4 + 4].i = out_shape_packed.w;
        specializations[4 + 5].i = out_shape_packed.h;
        specializations[4 + 6].i = out_shape_packed.c;
        specializations[4 + 7].i = out_shape_packed.cstep;

        int shader_type_index = -1;
        //if (elempack == 1 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_1x1s1d1;
        if (elempack == 4 && out_elempack == 4)
            shader_type_index = LayerShaderType::convolution1d_pack4_s1d1;
        //if (elempack == 1 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack1to4_1x1s1d1;
        if (elempack == 4 && out_elempack == 1)
            shader_type_index = LayerShaderType::convolution1d_pack4to1_s1d1;
        //if (elempack == 8 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack8_1x1s1d1;
        //if (elempack == 1 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack1to8_1x1s1d1;
        //if (elempack == 8 && out_elempack == 1) shader_type_index = LayerShaderType::convolution_pack8to1_1x1s1d1;
        //if (elempack == 4 && out_elempack == 8) shader_type_index = LayerShaderType::convolution_pack4to8_1x1s1d1;
        //if (elempack == 8 && out_elempack == 4) shader_type_index = LayerShaderType::convolution_pack8to4_1x1s1d1;

        pipeline_convolution_s1d1 = new Pipeline(vkdev);
        pipeline_convolution_s1d1->set_local_size_xyz(
                    GROUP_SIZE_X_LOCAL_MEMORY, GROUP_SIZE_Y_LOCAL_MEMORY, 1);
        pipeline_convolution_s1d1->create(
                                    shader_type_index, opt, specializations);
    }

    // generic case
    std::vector<vk_specialization_type> specializations(7 + 10);
    specializations[0].i = kernel_w;
    specializations[1].i = dilation_w;
    specializations[2].i = stride_w;
    specializations[3].i = bias_term;
    specializations[4].i = activation_type;
    specializations[5].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
    specializations[6].f = activation_params.w == 2 ? activation_params[1] : 0.f;
    specializations[7 + 0].i = shape_bordered_packed.dims;
    specializations[7 + 1].i = shape_bordered_packed.w;
    specializations[7 + 2].i = shape_bordered_packed.h;
    specializations[7 + 3].i = shape_bordered_packed.c;
    specializations[7 + 4].i = shape_bordered_packed.cstep;
    specializations[7 + 5].i = out_shape_packed.dims;
    specializations[7 + 6].i = out_shape_packed.w;
    specializations[7 + 7].i = out_shape_packed.h;
    specializations[7 + 8].i = out_shape_packed.c;
    specializations[7 + 9].i = out_shape_packed.cstep;

    Mat local_size_xyz(8, std::min(4, (num_output / out_elempack + 1) / 2), nullptr);
    if (out_shape_packed.dims != 0)
    {
        local_size_xyz.w = std::min(8, out_shape_packed.w);
        local_size_xyz.h = std::min(4, (out_shape_packed.h + 1) / 2);
        local_size_xyz.c = 1;
    }

    int shader_type_index = -1;
    if (elempack == 1 && out_elempack == 1)
        shader_type_index = LayerShaderType::convolution1d;
    else if (elempack == 4 && out_elempack == 4)
        shader_type_index = LayerShaderType::convolution1d_pack4;
    else if (elempack == 1 && out_elempack == 4)
        shader_type_index = LayerShaderType::convolution1d_pack1to4;
    else if (elempack == 4 && out_elempack == 1)
        shader_type_index = LayerShaderType::convolution1d_pack4to1;
    else if (elempack == 8 && out_elempack == 8)
        shader_type_index = LayerShaderType::convolution1d_pack8;
    else if (elempack == 1 && out_elempack == 8)
        shader_type_index = LayerShaderType::convolution1d_pack1to8;
    else if (elempack == 8 && out_elempack == 1)
        shader_type_index = LayerShaderType::convolution1d_pack8to1;
    else if (elempack == 4 && out_elempack == 8)
        shader_type_index = LayerShaderType::convolution1d_pack4to8;
    else if (elempack == 8 && out_elempack == 4)
        shader_type_index = LayerShaderType::convolution1d_pack8to4;

    pipeline_convolution = new Pipeline(vkdev);
    pipeline_convolution->set_optimal_local_size_xyz(local_size_xyz);
    pipeline_convolution->create(shader_type_index, opt, specializations);

    if ((stride_w != 1 || kernel_w != 1) && opt.use_shader_local_memory
    //if (opt.use_shader_local_memory
        && elempack == 4 && out_elempack == 4)
    {
        if (elempack == 4 && out_elempack == 4)
            shader_type_index = LayerShaderType::convolution1d_pack4_local;

        pipeline_convolution_local_memory = new Pipeline(vkdev);
        pipeline_convolution_local_memory->set_local_size_xyz(
                GROUP_SIZE_X_LOCAL_MEMORY, GROUP_SIZE_Y_LOCAL_MEMORY, 1);
        pipeline_convolution_local_memory->create(
                                    shader_type_index, opt, specializations);
    }

    return 0;
}

int Convolution1D_vulkan::destroy_pipeline(const Option& opt)
{
    if (padding)
    {
        padding->destroy_pipeline(opt);
        delete padding;
        padding = nullptr;
    }

    delete pipeline_convolution;
    pipeline_convolution = nullptr;

    if (pipeline_convolution_s1d1)
    {
        delete pipeline_convolution_s1d1;
        pipeline_convolution_s1d1 = nullptr;
    }

    if (pipeline_convolution_local_memory)
    {
        delete pipeline_convolution_local_memory;
        pipeline_convolution_local_memory = nullptr;
    }

    return 0;
}

int Convolution1D_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (padding)
    {
        padding->upload_model(cmd, opt);
    }

    if (support_image_storage && opt.use_image_storage)
    {
        cmd.record_upload(weight_data_packed, weight_data_gpu_image, opt);
    }
    else
    {
        cmd.record_upload(weight_data_packed, weight_data_gpu, opt);
    }

    weight_data_packed.release();

    if (bias_term)
    {
        if (support_image_storage && opt.use_image_storage)
        {
            cmd.record_upload(bias_data_packed, bias_data_gpu_image, opt);
        }
        else
        {
            cmd.record_upload(bias_data_packed, bias_data_gpu, opt);
        }

        bias_data_packed.release();
    }

    return 0;
}

int Convolution1D_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    // int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    VkMat bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0)
    {
        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->forward(bottom_blob, bottom_blob_bordered, cmd, opt_pad);
    }
    else if (pad_left == -233 && pad_right == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = 0;
        if (wpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad / 2;
            padding_params[1] = hpad - hpad / 2;
            padding_params[2] = wpad / 2;
            padding_params[3] = wpad - wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;
            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }
    }
    else if (pad_left == -234 && pad_right == -234)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = 0;
        if (wpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad - hpad / 2;
            padding_params[1] = hpad / 2;
            padding_params[2] = wpad - wpad / 2;
            padding_params[3] = wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;

            std::vector<VkMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }
    }

    w = bottom_blob_bordered.w;
    // h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    auto out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    top_blob.create(outw, num_output / out_elempack, out_elemsize,
                    out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;


    // pipeline binding
    std::vector<VkMat> bindings(4);
    bindings[0] = bottom_blob_bordered;
    bindings[1] = top_blob;
    bindings[2] = weight_data_gpu;
    bindings[3] = bias_data_gpu;

    if (pipeline_convolution_s1d1 && opt.use_shader_local_memory
        && top_blob.w >= GROUP_SIZE_X_LOCAL_MEMORY + kernel_w - 1
        && bottom_blob_bordered.h >= 4)
    {
        std::vector<vk_constant_type> constants(8);
        constants[0].i = bottom_blob_bordered.w;
        constants[1].i = bottom_blob_bordered.h;
        constants[2].i = bottom_blob_bordered.c;
        constants[3].i = bottom_blob_bordered.cstep;
        constants[4].i = top_blob.w;
        constants[5].i = top_blob.h;
        constants[6].i = top_blob.c;
        constants[7].i = top_blob.cstep;

        VkMat dispatcher;
        dispatcher.w = (top_blob.w + 3) / 4;
        dispatcher.h = top_blob.h;
        dispatcher.c = 1;

        cmd.record_pipeline(pipeline_convolution_s1d1, bindings,
                            constants, dispatcher);
    }
    else if (pipeline_convolution_local_memory && opt.use_shader_local_memory)
    {
        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_blob_bordered.dims;
        constants[1].i = bottom_blob_bordered.w;
        constants[2].i = bottom_blob_bordered.h;
        constants[3].i = bottom_blob_bordered.c;
        constants[4].i = bottom_blob_bordered.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = top_blob.cstep;

        VkMat dispatcher;
        dispatcher.w = (top_blob.w + 3) / 4;
        dispatcher.h = top_blob.h;
        dispatcher.c = top_blob.c;

	    NCNN_LOGE("\t=> use generic local memory convolution1d:"
                  "stride %d, dilation %d, dw %d, dh %d",
                  stride_w, dilation_w,
                  dispatcher.w, dispatcher.h);
        cmd.record_pipeline(pipeline_convolution_local_memory, bindings,
                            constants, dispatcher);
    }
    else
    {
        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_blob_bordered.dims;
        constants[1].i = bottom_blob_bordered.w;
        constants[2].i = bottom_blob_bordered.h;
        constants[3].i = bottom_blob_bordered.c;
        constants[4].i = bottom_blob_bordered.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = top_blob.cstep;

        VkMat dispatcher;
        dispatcher.w = (top_blob.w + 1) / 2;
        dispatcher.h = (top_blob.h + 1) / 2;
        dispatcher.c = top_blob.c;

        cmd.record_pipeline(pipeline_convolution, bindings,
                            constants, dispatcher);
    }
    return 0;
}

int Convolution1D_vulkan::forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    VkImageMat bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0)
    {
        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->forward(bottom_blob, bottom_blob_bordered, cmd, opt_pad);
    }
    else if (pad_left == -233 && pad_right == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = 0;
        if (wpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkImageMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad / 2;
            padding_params[1] = hpad - hpad / 2;
            padding_params[2] = wpad / 2;
            padding_params[3] = wpad - wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;

            std::vector<VkImageMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkImageMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }
    }
    else if (pad_left == -234 && pad_right == -234)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = 0;
        if (wpad > 0)
        {
            Option opt_pad = opt;
            opt_pad.blob_vkallocator = opt.workspace_vkallocator;

            VkImageMat padding_param_blob(6, (size_t)4u, 1, opt.staging_vkallocator);
            int* padding_params = padding_param_blob.mapped();

            padding_params[0] = hpad - hpad / 2;
            padding_params[1] = hpad / 2;
            padding_params[2] = wpad - wpad / 2;
            padding_params[3] = wpad / 2;
            padding_params[4] = 0;
            padding_params[5] = 0;

            std::vector<VkImageMat> padding_inputs(2);
            padding_inputs[0] = bottom_blob;
            padding_inputs[1] = padding_param_blob;

            std::vector<VkImageMat> padding_outputs(1);
            padding->forward(padding_inputs, padding_outputs, cmd, opt_pad);
            bottom_blob_bordered = padding_outputs[0];
        }
    }

    w = bottom_blob_bordered.w;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    top_blob.create(outw, num_output / out_elempack,
                    out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkImageMat> bindings(4);
    bindings[0] = bottom_blob_bordered;
    bindings[1] = top_blob;
    bindings[2] = weight_data_gpu_image;
    bindings[3] = bias_data_gpu_image;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob_bordered.dims;
    constants[1].i = bottom_blob_bordered.w;
    constants[2].i = bottom_blob_bordered.h;
    constants[3].i = bottom_blob_bordered.c;
    constants[4].i = 0; //bottom_blob_bordered.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = 0; //top_blob.cstep;

    VkImageMat dispatcher;
    dispatcher.w = (top_blob.w + 1) / 2;
    dispatcher.h = (top_blob.h + 1) / 2;
    dispatcher.c = top_blob.c;

    cmd.record_pipeline(pipeline_convolution, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
