#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>

#include "endian.hpp"

namespace stream
{

    static std::fstream open_binary_file(const std::filesystem::path& path)
    {
        if (!std::filesystem::exists(path))
        {
            throw std::runtime_error("can't open a non-existent file");
        }
        if (std::filesystem::is_directory(path))
        {
            throw std::runtime_error("can't open a directory as a binary file");
        }
        return std::fstream(path, std::ios::in | std::ios::binary);
    }

    template <class Elem = char, class Traits = std::char_traits<char>>
    void ensure(const std::basic_ios<Elem, Traits>& s)
    {
        if (s.fail())
        {
            throw std::runtime_error("stream in failure state");
        }
    }

    template <
        typename T,
        class Elem = char,
        class Traits = std::char_traits<char>
    >
    T read(std::basic_istream<Elem, Traits>& s)
    {
        Elem buf[sizeof(T)];
        s.read(buf, sizeof(T) / sizeof(Elem));

        ensure(s);

        return *reinterpret_cast<T*>(buf);
    }

    template <
        typename T,
        class Elem = char,
        class Traits = std::char_traits<char>
    >
    void read(
        std::basic_istream<Elem, Traits>& s,
        T* target,
        size_t count
    )
    {
        s.read(
            reinterpret_cast<Elem*>(target),
            count * sizeof(T) / sizeof(Elem)
        );
        ensure(s);
    }

    template <
        typename T,
        class Elem = char,
        class Traits = std::char_traits<char>
    >
    T read_bigend(std::basic_istream<Elem, Traits>& s)
    {
        return endian::big2host(read<T, Elem, Traits>(s));
    }

    template <
        typename T,
        class Elem = char,
        class Traits = std::char_traits<char>
    >
    T read_littleend(std::basic_istream<Elem, Traits>& s)
    {
        return endian::little2host(read<T, Elem, Traits>(s));
    }

    template <
        typename T,
        class Elem = char,
        class Traits = std::char_traits<char>
    >
    T read_bigend(
        std::basic_istream<Elem, Traits>& s,
        T* target,
        size_t count
    )
    {
        read<T, Elem, Traits>(s, target, count);
        for (size_t i = 0; i < count; i++)
        {
            target[i] = endian::big2host(target[i]);
        }
    }

    template <
        typename T,
        class Elem = char,
        class Traits = std::char_traits<char>
    >
    T read_littleend(
        std::basic_istream<Elem, Traits>& s,
        T* target,
        size_t count
    )
    {
        read<T, Elem, Traits>(s, target, count);
        for (size_t i = 0; i < count; i++)
        {
            target[i] = endian::little2host(target[i]);
        }
    }

}
