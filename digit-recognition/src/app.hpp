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

namespace digitrec
{

    class App
    {
    public:
        App();

        void run();

    private:
        static constexpr uint32_t seed = 14124;
        static inline const auto activation_fn = neural::tanh<float>;
        static inline const auto activation_deriv = neural::tanh_deriv<float>;

        std::mt19937 rng;
        neural::Network<float, 4> net;

        float test_cost();

    };

}
