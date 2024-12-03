#pragma once

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

        neural::Network<float, 4> net;

    };

}
