#include <iostream>
#include <fstream>
#include <chrono>
#include <cassert>

#include "net.h"

struct Config {
    bool use_vulkan;
    bool use_shader_local_memory;
    bool use_fp16;
    bool use_bf16;
    int debug_stage;
};

namespace chrono = std::chrono;

static void output(const ncnn::Mat &m, int maxR, int maxC)
{
    for (int r = 0; r < std::min(maxR, m.h); ++r) {
        auto *row = m.row(r);
        for (int c = 0; c < std::min<int>(maxC, m.w); ++c)
            std::fprintf(stdout, "%8.4f ", row[c]);
        std::fputc('\n', stdout);
    }
}

static void export_matrix(const ncnn::Mat &m, char const *filename)
{
    std::fstream f2(filename, std::ios::out);
    for (int r = 0; r < m.h; ++r) {
        auto row = m.row(r);
        for (int c = 0; c < m.w; ++c)
            f2 << row[c] << ' ';
        f2 << '\n';
    }
    f2.close();
}

static void initNet(ncnn::Net &net, char const *param, char const *model,
                    Config const &config)
{
    //net.opt.num_threads = 6;
    net.opt.use_vulkan_compute = config.use_vulkan;
    net.opt.use_shader_pack8 = false;
    net.opt.use_shader_local_memory = config.use_shader_local_memory;

    net.opt.use_fp16_packed = config.use_fp16;
    net.opt.use_fp16_arithmetic = config.use_fp16;
    net.opt.use_fp16_storage = config.use_fp16;

    net.opt.use_bf16_storage = config.use_bf16;

    net.opt.use_image_storage = false;

    net.opt.use_packing_layout = true;

    if (net.load_param(param))
        exit(-1);
    if (net.load_model(model))
        exit(-1);
}

static ncnn::Mat length_regulator(ncnn::Mat const &input, ncnn::Mat const &dur)
{
    // assume input is 2D, dur is 1D

    assert(input.h == dur.w);

    std::uint32_t total = 0;
    float const *dur_ptr= dur;
    for (std::size_t x = 0; x < static_cast<std::uint32_t>(dur.w); ++x) {
        auto d = static_cast<std::uint32_t>(std::max(0.f, dur_ptr[x]) + 0.5f);
        total += d;
    }

    ncnn::Mat res(input.w, static_cast<int>(total));

    float *dptr = res.row(0);
    for (std::size_t y = 0; y < static_cast<std::size_t>(input.h); ++y) {
        auto d = static_cast<std::size_t>(std::max(0.f, dur_ptr[y]) + 0.5f);

        auto srow = input.row(y);
        // repeate row d times
        for (std::size_t yi = 0; yi < d; ++yi)
        {
            for (std::size_t x = 0; x < static_cast<std::size_t>(input.w); ++x)
                *dptr++ = srow[x];
        }
    }
    return res;
}

static ncnn::Mat transpose(ncnn::Mat const &input)
{
    assert(input.c == 1 && input.elempack == 1);
    ncnn::Mat res(input.h, input.w);

    for (int y = 0; y < input.h; ++y){
        float const *row = input.row(y);
        for (int x = 0; x < input.w; ++x)
            res.row(x)[y] = row[x];
    }

    return res;
}

static void test(const ncnn::Mat &in0, ncnn::Mat &out0, ncnn::Mat &out1,
                 char const *output_name, Config const &config)
{
    ncnn::Net net0, net2, net3;

    initNet(net0, "tts-stage0.ncnn.param", "tts-stage0.ncnn.bin", config);
    if (config.debug_stage > 0) {
        initNet(net2, "tts-stage2.ncnn.param", "tts-stage2.ncnn.bin", config);
        if (config.debug_stage > 2) {
            initNet(net3, "tts-stage3.ncnn.param", "tts-stage3.ncnn.bin", config);
        }
    }


    // output some data
    std::fprintf(stdout, "Input: %d, %d\n", in0.h, in0.w);
    output(in0, 8, 8);

    ncnn::Mat out_st0, dur_st0, out_st2;

    ///////////////////////////////////////////////////////////
    /// INFERENCE
    auto start = std::chrono::system_clock::now();
    decltype(start) time0, time1, time2, time3;
    decltype(time0 - time1) elapsed;

    // stage 0
    std::fprintf(stdout, "\n\n----------------------------------");
    std::fprintf(stdout, "Stage 0\n");
    auto ex0 = net0.create_extractor();
    ex0.input("in0", in0);
    if (config.debug_stage == 0) {
        ex0.extract(output_name, out_st0);
        out0 = out_st0;
        out1 = out_st0;
    } else {
        ex0.extract("out1", dur_st0);
        ex0.extract("out0", out_st0);

        time0 = std::chrono::system_clock::now();

        // stage 1
        std::fprintf(stdout, "\n\n----------------------------------");
        std::fprintf(stdout, "Stage 1\n");
        ncnn::Mat in2 = length_regulator(out_st0, dur_st0);

        time1 = std::chrono::system_clock::now();

        // stage 2
        std::fprintf(stdout, "\n\n----------------------------------");
        std::fprintf(stdout, "Stage 2\n");
        auto ex2 = net2.create_extractor();
        ex2.input("in0", in2);
        if (config.debug_stage == 2) {
            ex2.extract(output_name, out_st2);
            out0 = out_st2;
            out1 = out_st2;
        } else {
            ex2.extract("out0", out_st2);

            time2 = std::chrono::system_clock::now();

            // stage 3

            std::fprintf(stdout, "\n\n----------------------------------");
            std::fprintf(stdout, "Stage 3\n");
            // because ncnn eliminated the last transpose layer of 2nd stage 
            ncnn::Mat in3 = transpose(out_st2);
            auto ex3 = net3.create_extractor();
            ex3.input("in0", in3);

            if (config.debug_stage == 3) {
                ex3.extract(output_name, out0);
            } else {
                ex3.extract("out0", out0);

                time3 = std::chrono::system_clock::now();
                elapsed = time3 - start;

            } // debug_stage == 3
        } // debug_stage == 2
    } // debug_stage == 0

    if (config.debug_stage > 3) {
        // output some data
        std::fprintf(stdout, "\n\nTime: total %lld ms"
                     "\n\tStage0: %lld ms"
                     "\n\tStage1: %lld ms"
                     "\n\tStage2: %lld ms"
                     "\n\tStage3: %lld ms\n\n",
                     chrono::duration_cast<chrono::milliseconds>(elapsed).count(),
                     chrono::duration_cast<chrono::milliseconds>(time0 - start).count(),
                     chrono::duration_cast<chrono::milliseconds>(time1 - time0).count(),
                     chrono::duration_cast<chrono::milliseconds>(time2 - time1).count(),
                     chrono::duration_cast<chrono::milliseconds>(time3 - time2).count());
    }
    std::fprintf(stdout, "Output: %d, %d\n", out0.h, out0.w);
    output(out0, 8, 8);
    std::fprintf(stdout, "Output: %d, %d\n", out1.h, out1.w);
    output(out1, 8, 8);

#if 0
    // check
    int w = in0.w, h = in0.h;

    float maxDiff = 0.0f;
    for (int r = 0; r < h; ++r) {
        float const *inRow = in0.row(r), *outRow = out0.row(r);
        for (int c = 0; c < w; ++c)
            maxDiff = std::max(maxDiff,
                               std::fabs((inRow[c] + 5.f) / 5.f - outRow[c]));
    }

    std::fprintf(stdout, "\n\nMax diff: %f\n", maxDiff);
#endif
}

int main(int argc, char *argv[])
{
    if (argc != 9) {
        std::fputs("Usage: binaryop DEBUG_STAGE INPUT_MATRIX OUTPUT_MATRIX OUTPUT_NAME\n", stderr);
        return EXIT_FAILURE;
    }

    int debug_stage = argv[1][0] - '0';

    std::vector<int> vi;
    std::fstream f(argv[2]);
    int tmp;

    for(; f >> tmp; vi.emplace_back(tmp));

    ncnn::Mat in0(vi.size(), 1), out0, dur0;
    int *row = reinterpret_cast<int*>(in0.row(0));
    for (auto x : vi)
        *row++ = x;

    int c = 5;

    Config config {
        .use_vulkan = argv[c][0] == '1',
        .use_shader_local_memory = argv[c+1][0] == '1',
        .use_fp16 = argv[c+2][0] == '1',
        .use_bf16 = argv[c+3][0] == '1',
        .debug_stage = debug_stage
    };

    test(in0, out0, dur0, argv[4], config);

    export_matrix(out0, argv[3]);
    return 0;
}
