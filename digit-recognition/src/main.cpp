#include <iostream>

#include "app_1d_function.hpp"

int main()
{
    try
    {
        digitrec::App1dFunction app;
        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
