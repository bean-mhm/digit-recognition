#pragma once

#include <iostream>
#include <fstream>
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

namespace digitrec
{

    static constexpr size_t DIGIT_WIDTH = 28;
    static constexpr size_t DIGIT_HEIGHT = 28;
    static constexpr size_t N_DIGIT_VALUES = DIGIT_WIDTH * DIGIT_HEIGHT;

    struct DigitSample
    {
        // pixel values for a digit stored in a row major format
        std::array<float, N_DIGIT_VALUES> values;

        // digit label from 0 to 9. I didn't use uint8_t for better alignment.
        uint32_t label;
    };

    class AppDigitRecognition
    {
    public:
        AppDigitRecognition();

        void run();

    private:
        std::vector<DigitSample> training_samples;
        std::vector<DigitSample> test_samples;

        void load_digit_samples();

    };

}
