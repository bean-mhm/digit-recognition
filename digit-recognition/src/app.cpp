#include "app.hpp"

namespace digitrec
{

    App::App()
        : rng(seed),
        net(
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
        // randomize weights and biases
        net.randomize_xavier(rng, -.01f, .01f);

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
