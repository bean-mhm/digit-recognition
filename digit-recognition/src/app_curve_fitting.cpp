#include "app_curve_fitting.hpp"

namespace curve_fitting
{

    static float target_fn(float v)
    {
        //return std::fmod(v, .5f) > .25f ? 1.f : 0.f;
        return neural::gaussian_distribution<float>(.5f, .1f, v) * .2f;
    }

    template<typename RandomEngine>
    static void generate_random_training_data(
        RandomEngine& engine,
        size_t n_data_points,
        std::vector<float>& out_data,
        std::vector<std::span<float>>& out_spans
    )
    {
        std::uniform_real_distribution<float> dist(0.f, 1.f);

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
        : rng(SEED),
        net(
            { 1, 16, 16, 1 },
            { ACTIVATION_FN, ACTIVATION_FN, ACTIVATION_FN },
            { ACTIVATION_DERIV, ACTIVATION_DERIV, ACTIVATION_DERIV }
        )
    {}

    void App::run()
    {
        // randomize weights and biases
        net.randomize_xavier_normal(rng, -.01f, .01f);

        // train
        static constexpr size_t N_STEPS = 100000u;
        for (size_t i = 0u; i < N_STEPS; i++)
        {
            std::vector<float> data;
            std::vector<std::span<float>> spans;
            generate_random_training_data(rng, 10u, data, spans);

            net.train(spans, .01f);

            if (i % 1000 == 0)
            {
                std::cout << std::format("training step {} / {}\n", i, N_STEPS);
            }
        }

        // evaluate the network
        for (float v = -1.f; v < 2.0001f; v += .005f)
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
