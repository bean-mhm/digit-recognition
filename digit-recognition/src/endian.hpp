#pragma once

#include <bit>
#include <climits>

namespace endian
{

    template<typename T>
    T swap(T u)
    {
        static_assert (CHAR_BIT == 8, "CHAR_BIT != 8");

        union
        {
            T u;
            unsigned char u8[sizeof(T)];
        } source, dest;

        source.u = u;

        for (size_t k = 0; k < sizeof(T); k++)
            dest.u8[k] = source.u8[sizeof(T) - k - 1];

        return dest.u;
    }

    template<typename T>
    inline T host2big(T u)
    {
        if constexpr (std::endian::native == std::endian::little)
        {
            return swap(u);
        }
        return u;
    }

    template<typename T>
    inline T host2little(T u)
    {
        if constexpr (std::endian::native == std::endian::big)
        {
            return swap(u);
        }
        return u;
    }

    template<typename T>
    inline T big2host(T u)
    {
        return host2big(u);
    }

    template<typename T>
    inline T little2host(T u)
    {
        return host2little(u);
    }

}
