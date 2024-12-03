#pragma once

#include <tuple>
#include <array>
#include <vector>
#include <functional>
#include <type_traits>
#include <utility>
#include <cstdint>

namespace neural
{

    template<
        template<typename, size_t> typename TFixedArray,
        typename TNumber,
        size_t prev_layer_size
    >
    class Node
    {
    public:
        TFixedArray<TNumber, prev_layer_size> weights;
        TNumber bias;
        TNumber value;

    };

    template<
        template<typename, size_t> typename TFixedArray,
        typename TNumber,
        size_t size,
        size_t prev_layer_size
    >
    class Layer
    {
    public:
        TFixedArray<
            Node<TFixedArray, TNumber, prev_layer_size>,
            size
        > nodes;

    };

    namespace detail
    {

        template<auto a, auto... b>
        constexpr auto first = a;

        template<template<size_t...> typename T, size_t First, size_t... Rest>
        struct RemoveFirst
        {
            using Result = T<Rest...>;
        };

        template <auto a, auto... b>
        constexpr auto remove_first = std::make_tuple(b...);

        template<typename T, size_t size>
        consteval std::array<T, size - (size_t)1> remove_last(
            std::array<T, size> arr
        )
        {
            std::array<T, size - (size_t)1> r{ 0 };
            for (size_t i = 0; i < size - (size_t)1; i++)
            {
                r[i] = arr[i];
            }
            return r;
        }

        template<typename T, T... values>
        constexpr auto RemoveLast = std::array<T, sizeof...(values)>(values...)[
            std::make_index_sequence<sizeof...(values) - (size_t)1>
        ]...;

        template<
            template<typename, size_t> typename TFixedArray,
            typename TNumber,
            size_t... layer_sizes
        > requires (sizeof...(layer_sizes) >= (size_t)2)
            class Network
        {
        public:
            // the input layer which only contains values and no weights, biases, or
            // activation functions.
            TFixedArray<TNumber, first<layer_sizes...>> input;

            // the hidden layers and the output layer
            Layer<
                TFixedArray,
                TNumber,
                remove_first<layer_sizes...>...,
                RemoveLast<size_t, layer_sizes...>...
            > layer;

            // activation functions for the hidden layers and the output layer
            TFixedArray<
                std::function<TNumber(TNumber)>,
                sizeof...(layer_sizes) - (size_t)1
            > activation_f;

            // derivatives for the activation functions
            TFixedArray<
                std::function<TNumber(TNumber)>,
                sizeof...(layer_sizes) - (size_t)1
            > activation_fp;

        };

    }

    // TFixedArray must be a templated array type that takes in a typename
    // parameter to represent its element type and a size_t parameter for the
    // number of elements.
    // 
    // TNumber is the type used to store numerical values. A typical value may
    // be `float`.
    //
    // layer_sizes is a parameter pack to represent the sizes of each layer.
    // there should be at least 2 values to represent the sizes of the input and
    // output layers.
    template<
        template<typename, size_t> typename TFixedArray,
        typename TNumber,
        size_t... layer_sizes
    > requires (sizeof...(layer_sizes) >= (size_t)2)
        class Network
    {
    public:
        // the input layer which only contains values and no weights, biases, or
        // activation functions.
        TFixedArray<TNumber, detail::first<layer_sizes...>> input;

        // the hidden layers and the output layer
        Layer<
            TFixedArray,
            TNumber,
            detail::remove_first<layer_sizes...>...,
            detail::RemoveLast<size_t, layer_sizes...>...
        > layer;

        // activation functions for the hidden layers and the output layer
        TFixedArray<
            std::function<TNumber(TNumber)>,
            sizeof...(layer_sizes) - (size_t)1
        > activation_f;

        // derivatives for the activation functions
        TFixedArray<
            std::function<TNumber(TNumber)>,
            sizeof...(layer_sizes) - (size_t)1
        > activation_fp;

    };

}
