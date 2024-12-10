#include <iostream>

#include "app_curve_fitting.hpp"
#include "app_digit_rec.hpp"

int main()
{
    try
    {
        digit_rec::App app;
        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        std::cin.get();
        return 1;
    }
}
