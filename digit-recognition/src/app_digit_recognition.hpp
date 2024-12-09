#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <format>
#include <array>
#include <vector>
#include <span>
#include <functional>
#include <random>
#include <stdexcept>
#include <cmath>
#include <cstdint>

#include "neural.hpp"
#include "endian.hpp"
#include "stream.hpp"

namespace digitrec
{

    static constexpr size_t DIGIT_WIDTH = 28;
    static constexpr size_t DIGIT_HEIGHT = 28;
    static constexpr size_t N_DIGIT_VALUES = DIGIT_WIDTH * DIGIT_HEIGHT;

    static constexpr auto TRAIN_IMAGES_PATH =
        "./MNIST/train-images.idx3-ubyte";
    static constexpr auto TRAIN_LABELS_PATH =
        "./MNIST/train-labels.idx1-ubyte";
    static constexpr auto TEST_IMAGES_PATH = "./MNIST/t10k-images.idx3-ubyte";
    static constexpr auto TEST_LABELS_PATH = "./MNIST/t10k-labels.idx1-ubyte";

    struct DigitSample
    {
        // pixel values for a digit stored in a row major format
        std::array<uint8_t, N_DIGIT_VALUES> values;

        // digit label from 0 to 9. I didn't use uint8_t for better alignment.
        uint32_t label;
    };

    class AppDigitRecognition
    {
    public:
        AppDigitRecognition();

        void run();

    private:
        std::vector<DigitSample> train_samples;
        std::vector<DigitSample> test_samples;

        void load_digit_samples(
            std::string_view images_path,
            std::string_view labels_path,
            std::vector<DigitSample>& out_samples
        );

    };

}
