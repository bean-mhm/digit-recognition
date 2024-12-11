#pragma once

#include <vector>
#include <string>
#include <algorithm> 
#include <cctype>
#include <locale>

namespace str
{

    // trim from start (in place)
    static void ltrim_inplace(std::string& s)
    {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch)
            {
                return !std::isspace(ch);
            }));
    }

    // trim from end (in place)
    static void rtrim_inplace(std::string& s)
    {
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch)
            {
                return !std::isspace(ch);
            }).base(), s.end());
    }

    // trim from both ends (in place)
    static void trim_inplace(std::string& s)
    {
        rtrim_inplace(s);
        ltrim_inplace(s);
    }

    // trim from start (copying)
    static std::string ltrim_copy(std::string s)
    {
        ltrim_inplace(s);
        return s;
    }

    // trim from end (copying)
    static std::string rtrim_copy(std::string s)
    {
        rtrim_inplace(s);
        return s;
    }

    // trim from both ends (copying)
    static std::string trim_copy(std::string s)
    {
        trim_inplace(s);
        return s;
    }

    // https://www.geeksforgeeks.org/how-to-split-string-by-delimiter-in-cpp/
    static std::vector<std::string> split(
        std::string s, // must be copied
        std::string_view delimiter
    )
    {
        if (s.empty())
        {
            return {};
        }

        std::vector<std::string> elem;

        // find the first occurrence of the delimiter
        auto pos = s.find(delimiter);

        // while there are still delimiters in the string
        while (pos != std::string::npos)
        {
            // extract substring up to the delimiter
            elem.push_back(std::string{ s.substr(0, pos) });

            // erase the extracted part from the original string
            s.erase(0, pos + delimiter.length());

            // find the next occurrence of the delimiter
            pos = s.find(delimiter);
        }

        // output the last substring (after the last delimiter)
        if (!elem.empty() && !s.empty())
        {
            elem.push_back(s);
        }

        return elem;
    }

}
