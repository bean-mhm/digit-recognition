#include "app.hpp"

#include <iostream>
#include <string>
#include <format>
#include <array>
#include <vector>
#include <span>
#include <functional>
#include <random>
#include <stdexcept>
#include <cstdint>

namespace digitrec
{

    App::App()
        : net(
            { 1, 16, 16, 1 },
            { neural::tanh<float>, neural::tanh<float>, neural::tanh<float> },
        {
            neural::tanh_deriv<float>,
            neural::tanh_deriv<float>,
            neural::tanh_deriv<float>
        }
        )
    {}

    void App::run()
    {
        // randomize neural network values
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (auto& v : net.input_values())
        {
            v = dist(rng);
        }
        for (size_t i = 1; i < net.num_layers(); i++)
        {
            for (auto& v : net.values(i))
            {
                v = dist(rng);
            }
            for (auto& v : net.biases(i))
            {
                v = .01f * dist(rng);
            }
            for (size_t j = 0; j < net.layer_sizes()[i]; j++)
            {
                for (auto& v : net.weights(i, j))
                {
                    v = dist(rng);
                }
            }
        }

        // evaluate network for a few values
        for (float v = -6.f; v < 6.01f; v += .05f)
        {
            net.input_values()[0] = v;
            net.eval();
            std::cout << std::format(
                "{:.3f}, {:.3f}\n",
                v,
                net.output_values()[0]
            );
        }
    }

}
