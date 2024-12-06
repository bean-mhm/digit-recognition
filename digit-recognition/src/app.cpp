#include "app.hpp"

namespace digitrec
{

    static float target_fn(float v)
    {
        return std::fmod(v, 2.f) > 1.f ? 1.f : 0.f;
    }

    template<typename RandomEngine>
    static void generate_random_training_data(
        RandomEngine& engine,
        size_t n_data_points,
        std::vector<float>& out_data,
        std::vector<std::span<float>>& out_spans
    )
    {
        std::uniform_real_distribution<float> dist(-6.f, 6.f);

        out_data.resize(n_data_points * 2u);
        for (size_t i = 0u; i < n_data_points; i++)
        {
            out_data[i * 2u] = dist(engine);
            out_data[i * 2u + 1u] = target_fn(out_data[i * 2u]);
        }

        out_spans.resize(n_data_points);
        for (size_t i = 0u; i < n_data_points; i++)
        {
            out_spans[i] = std::span<float>(out_data.data() + i * 2u, 2u);
        }
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

        // train
        std::uniform_real_distribution<float> dist(-6.f, 6.f);
        for (size_t i = 0; i < 200; i++)
        {
            std::vector<float> data;
            std::vector<std::span<float>> spans;
            generate_random_training_data(rng, 200u, data, spans);

            for (size_t i = 0; i < 100; i++)
            {
                net.train(spans, .01f);
            }
        }

        // evaluate network for a few values
        for (float v = -10.f; v < 10.01f; v += .05f)
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
        std::vector<float> data;
        std::vector<std::span<float>> spans;
        generate_random_training_data(rng, 500u, data, spans);

        return net.average_cost(spans);
    }

}
