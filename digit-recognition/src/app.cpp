#include "app.hpp"

namespace digitrec
{

    static float target_fn(float v)
    {
        return std::fmod(v, 1.f) > .5f ? 1.f : 0.f;
    }

    App::App()
        : rng(seed),
        net(
            { 1, 16, 16, 1 },
            { activation_fn, activation_fn, activation_fn },
            { activation_deriv, activation_deriv, activation_deriv }
        )
    {}

    void App::run()
    {
        // randomize weights and biases
        net.randomize_xavier_normal(rng, -.01f, .01f);

        // evaluate network for a few values
        for (float v = -6.f; v < 6.01f; v += .05f)
        {
            net.input_values()[0] = v;
            net.forward_pass();
            std::cout << std::format(
                "{:.3f}, {:.3f}\n",
                v,
                net.output_values()[0]
            );
        }

        std::cout << std::format("test cost: {:.4f}\n", test_cost());
    }

    float App::test_cost()
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-10.f, 10.f);

        std::vector<float> data_points(2000);
        for (size_t i = 0; i < data_points.size(); i += 2)
        {
            data_points[i] = dist(rng);
            data_points[i + 1] = target_fn(data_points[i]);
        }

        return net.average_cost(
            std::span<float>(data_points.data(), data_points.size())
        );
    }

}
