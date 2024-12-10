#pragma once

#include <iostream>
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

namespace curve_fitting
{

    class App
    {
    public:
        App();

        void run();

    private:
        static constexpr uint32_t SEED = 2727272u;
        static inline const auto ACTIVATION_FN =
            neural::leaky_relu<float, .01f>;
        static inline const auto ACTIVATION_DERIV =
            neural::leaky_relu_deriv<float, .01f>;

        std::mt19937 rng;
        neural::Network<float, true> net;

        float test_cost();

    };

}
