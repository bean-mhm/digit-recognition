#include <iostream>

#include "app.hpp"

int main()
{
    try
    {
        digitrec::App app;
        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
