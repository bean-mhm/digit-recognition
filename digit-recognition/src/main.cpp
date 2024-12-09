#include <iostream>

#include "app_1d_function.hpp"
#include "app_digit_recognition.hpp"

int main()
{
    try
    {
        digitrec::AppDigitRecognition app;
        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
