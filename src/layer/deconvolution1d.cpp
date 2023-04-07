// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "deconvolution1d.h"

#include "fused_activation.h"

namespace ncnn {

Deconvolution1D::Deconvolution1D()
{
    one_blob_only = true;
    support_inplace = false;
}

int Deconvolution1D::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    dilation_w = pd.get(2, 1);
    stride_w = pd.get(3, 1);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    output_pad_right = pd.get(18, 0);
    output_w = pd.get(20, 0);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    return 0;
}

int Deconvolution1D::load_model(const ModelBin& mb)
{
    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}

int Deconvolution1D::create_pipeline(const Option &)
{
    if (dilation_w != 1)
        return -100;

    int inh = weight_data_size / num_output / kernel_w;
    weight_data_transposed.create(
            inh, kernel_w, num_output,
            weight_data.elemsize, weight_data.elempack, weight_data.allocator);

    float const *ink = (float const *)weight_data;
    for (int p = 0; p < num_output; ++p)
    {
        float *outk = weight_data_transposed.channel(p);
        for (int r = 0; r < inh; ++r)
        {
            for (int s = 0, c = 0; s < stride_w; ++s)
            {
                for (int origc = s; origc < kernel_w; origc += stride_w, ++c)
                {
                    outk[c * inh + r] = ink[r * kernel_w + origc];
                }
            }
        }
        ink += kernel_w * inh;
    }

    return 0;
}

static int deconvolution1d(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int kernel_w, int stride_w, int dilation_w, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;

    const int outw = top_blob.w;
    const int outh = top_blob.h;

    const int bias_term = bias_data.empty() ? 0 : 1;

    #define NEW
    #ifdef NEW
    Mat bottom_blob_transposed(h, w);
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c)
            ((float*)bottom_blob_transposed)[c*h+r]
                                        = ((float const*)bottom_blob)[r*w+c];


    // maximum num of strides per kernel
    int n_stride = (kernel_w + stride_w - 1) / stride_w;
    #endif

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outh; p++)
    {
        const float bias = bias_term ? bias_data[p] : 0.f;

        #ifdef NEW
        float *out = top_blob.row(p);
        #else
        Mat out = top_blob.row_range(p, 1);

        out.fill(bias);
        #endif

        #ifdef NEW
        for (int j = 0; j < outw; ++j)
        {
            float res = bias;
            //float sum0 = bias, sum1 = bias, sum2 = bias, sum3 = bias;

            // calculate input index
            int jj = j / stride_w, kidx = j - jj * stride_w;
            auto divmod = std::div(kidx, stride_w);
            // kernel index after re-arange by prduce at ::create_pipeline
            int new_kidx = n_stride * divmod.rem + divmod.quot;

            const float* kptr = (const float*)weight_data
                                + kernel_w * h * p + new_kidx * h;
            const float* sptr = (const float*)bottom_blob_transposed + h * jj;

            for (; jj >= w; kidx += stride_w, --jj, kptr += h, sptr -= h);

            for (; (kidx < kernel_w) & (jj >= 0); kidx += stride_w, --jj)
            {
                for (int q = 0; q < h; ++q)
                    res += kptr[q] * sptr[q];
                #if 0
                int q = 0;
                for (q = 0; q + 4 <= h; q += 4)
                {
                    sum0 += kptr[q] * sptr[q];
                    sum1 += kptr[q+1] * sptr[q+1];
                    sum2 += kptr[q+2] * sptr[q+2];
                    sum3 += kptr[q+3] * sptr[q+3];
                }
                for (; q < h; ++q)
                    sum0 += kptr[q] * sptr[q];
                #endif
                kptr += h;
                sptr -= h;
            }

            // res = sum0 + sum1 + sum2 + sum3;
            out[j] = activation_ss(res, activation_type, activation_params);
        }


        #else
        for (int j = 0; j < w; j++)
        {
            float* outptr = (float*)out + j * stride_w;

            const float* kptr = (const float*)weight_data + kernel_w * h * p;

            for (int q = 0; q < h; q++)
            {
                const float val = bottom_blob.row(q)[j];

                for (int k = 0; k < kernel_w; k++)
                {
                    float w = kptr[k];
                    outptr[k * dilation_w] += val * w;
                }

                kptr += kernel_w;
            }
        }

        {
            float* outptr = out;

            for (int i = 0; i < outw; i++)
            {
                outptr[i] = activation_ss(outptr[i], activation_type, activation_params);
            }
        }
        #endif
    }

    return 0;
}

int Deconvolution1D::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w + output_pad_right;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || output_w > 0)
    {
        top_blob_bordered.create(outw, num_output, elemsize, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, num_output, elemsize, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    #ifdef NEW
    int ret = deconvolution1d(bottom_blob, top_blob_bordered, weight_data_transposed, bias_data, kernel_w, stride_w, dilation_w, activation_type, activation_params, opt);
    #else
    int ret = deconvolution1d(bottom_blob, top_blob_bordered, weight_data, bias_data, kernel_w, stride_w, dilation_w, activation_type, activation_params, opt);
    #endif
    if (ret != 0)
        return ret;

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

void Deconvolution1D::cut_padding(const Mat& top_blob_bordered, Mat& top_blob, const Option& opt) const
{
    if (pad_left > 0 || pad_right > 0)
    {
        copy_cut_border(top_blob_bordered, top_blob, 0, 0, pad_left, pad_right, opt);
    }
    else if (output_w > 0)
    {
        int wcut = top_blob_bordered.w - output_w;

        if (pad_left == -233 || pad_right == -233)
        {
            // onnx padding=SAME_UPPER
            copy_cut_border(top_blob_bordered, top_blob, 0, 0, wcut / 2, wcut - wcut / 2, opt);
        }
        else if (pad_left == -234 || pad_right == -234)
        {
            // onnx padding=SAME_LOWER
            copy_cut_border(top_blob_bordered, top_blob, 0, 0, wcut - wcut / 2, wcut / 2, opt);
        }
    }
    else
    {
        top_blob = top_blob_bordered;
    }
}

} // namespace ncnn
