#include <iostream>
#include <fstream>
#include <chrono>

#include "net.h"

namespace chrono = std::chrono;

static void output(const ncnn::Mat &m, int maxR, int maxC)
{
    for (int r = 0; r < std::min(maxR, m.h); ++r) {
        auto *row = m.row(r);
        for (int c = 0; c < std::min<int>(maxC, m.w); ++c)
            std::fprintf(stdout, "%.04f ", row[c]);
        std::fputc('\n', stdout);
    }
}

static void test(const ncnn::Mat &in0, ncnn::Mat &out0)
{
    ncnn::Net net;

    // net.opt.use_fp16_packed = false;
    // net.opt.use_fp16_arithmetic = false;
    // net.opt.use_fp16_storage = false;

    if (net.load_param("/tmp/ncnn/tts-stage3.ncnn.param"))
        exit(-1);
    if (net.load_model("/tmp/ncnn/tts-stage3.ncnn.bin"))
        exit(-1);

    // net.opt.use_fp16_packed = false;
    // net.opt.use_fp16_arithmetic = false;
    // net.opt.use_fp16_storage = false;

    // output some data
    std::fprintf(stdout, "Input: %d, %d\n", in0.h, in0.w);
    output(in0, 5, 5);

    auto start = std::chrono::system_clock::now();
    auto ex = net.create_extractor();
    ex.input("in0", in0);

    ex.extract("out0", out0);
    auto elapsed = std::chrono::system_clock::now() - start;

    // output some data
    std::fprintf(stdout, "\n\nTime: %ld ms\n",
                 chrono::duration_cast<chrono::milliseconds>(elapsed).count());
    std::fprintf(stdout, "Output: %d, %d\n", out0.h, out0.w);
    output(out0, 5, 5);

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
    if (argc < 3) {
        std::fputs("Usage: binaryop INPUT_MATRIX OUTPUT_MATRIX\n", stderr);
        return EXIT_FAILURE;
    }
    int w = 1601, h = 80;
    std::fstream f(argv[1]);

    ncnn::Mat in0(w, h, 1), out0;
    for (int r = 0; r < h; ++r) {
        auto row = in0.row(r);
        for (int c = 0; c < w; ++c)
            f >> row[c];
    }

    test(in0, out0);

    std::fstream f2(argv[2], std::ios::out);
    for (int r = 0; r < out0.h; ++r) {
        auto row = out0.row(r);
        for (int c = 0; c < out0.w; ++c)
            f2 << row[c] << ' ';
        f2 << '\n';
    }
    f2.close();
    return 0;
}
