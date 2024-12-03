#pragma once

#include <array>
#include <vector>
#include <span>
#include <functional>
#include <concepts>
#include <stdexcept>
#include <cstdint>

namespace neural
{

    template<typename T>
    T relu(T v)
    {
        if (v < (T)0)
            return (T)0;
        return v;
    }

    template<typename T>
    T relu_deriv(T v)
    {
        if (v < (T)0)
            return (T)0;
        return (T)1;
    }

    template<std::floating_point T>
    T tanh(T v)
    {
        return std::tanh(v);
    }

    template<std::floating_point T>
    T tanh_deriv(T v)
    {
        T th = std::tanh(v);
        return (T)1 - (th * th);
    }

    // T is the type used to store numerical values. A typical value may
    // be `float`.
    // n_layers is the number of layers. it should be at least 2 to represent an
    // input and an output layer.
    template<typename T, size_t n_layers, bool strict_checks = true>
        requires(n_layers >= (size_t)2)
    class Network
    {
    public:
        Network(
            const std::array<size_t, n_layers>& layer_sizes,

            const std::array<std::function<T(T)>, n_layers - (size_t)1>&
            activation_fns,

            const std::array<std::function<T(T)>, n_layers - (size_t)1>&
            activation_derivs
        )
            : _layer_sizes(layer_sizes),
            _activation_fns(activation_fns),
            _activation_derivs(activation_derivs)
        {
            if (n_layers < (size_t)2)
            {
                throw std::invalid_argument(
                    "n_layers should be at least 2 to represent an input and "
                    "an output layer."
                );
            }

            for (const size_t layer_size : _layer_sizes)
            {
                if (layer_size < (size_t)1)
                {
                    throw std::invalid_argument(
                        "a layer must contain at least 1 node"
                    );
                }
            }

            size_t n_data = _layer_sizes[0];
            for (size_t i = 1; i < n_layers; i++)
            {
                size_t n_nodes = _layer_sizes[i];
                n_data += n_nodes // values
                    + n_nodes // biases
                    + (n_nodes * _layer_sizes[i - (size_t)1]); // weights
            }
            data.resize(n_data, (T)0);
        }

        constexpr size_t num_layers() const
        {
            return n_layers;
        }

        constexpr const std::array<size_t, n_layers>& layer_sizes() const
        {
            return _layer_sizes;
        }

        constexpr const size_t input_layer_size() const
        {
            return layer_sizes()[0];
        }

        constexpr const size_t output_layer_size() const
        {
            return layer_sizes()[n_layers - (size_t)1];
        }

        // activation function for a hidden layer or the output layer
        constexpr const std::function<T(T)>& activation_fn(
            size_t layer_idx
        ) const
        {
            if (strict_checks
                && (layer_idx < (size_t)1 || layer_idx >= n_layers))
            {
                throw std::invalid_argument("invalid layer index");
            }
            return _activation_fns[layer_idx - (size_t)1];
        }

        // derivative of the activation function for a hidden layer or the
        // output layer.
        constexpr const std::function<T(T)>& activation_deriv(
            size_t layer_idx
        ) const
        {
            if (strict_checks
                && (layer_idx < (size_t)1 || layer_idx >= n_layers))
            {
                throw std::invalid_argument("invalid layer index");
            }
            return _activation_derivs[layer_idx - (size_t)1];
        }

        // node values in a layer
        constexpr std::span<T> values(size_t layer_idx)
        {
            if (strict_checks && layer_idx >= n_layers)
            {
                throw std::invalid_argument("invalid layer index");
            }

            if (layer_idx == 0)
            {
                return std::span<T>(data.data(), input_layer_size());
            }

            size_t data_idx = input_layer_size();
            for (size_t i = 1; i < layer_idx; i++)
            {
                size_t n_nodes = layer_sizes()[i];
                data_idx += n_nodes // values
                    + n_nodes // biases
                    + (n_nodes * layer_sizes()[i - (size_t)1]); // weights
            }

            return std::span<T>(
                data.data() + data_idx,
                layer_sizes()[layer_idx]
            );
        }

        // node values in the first layer
        constexpr std::span<T> input_values()
        {
            return values(0);
        }

        // node values in the last layer
        constexpr std::span<T> output_values()
        {
            return values(n_layers - 1);
        }

        // node biases in a layer
        constexpr std::span<T> biases(size_t layer_idx)
        {
            if (strict_checks
                && (layer_idx < (size_t)1 || layer_idx >= n_layers))
            {
                throw std::invalid_argument("invalid layer index");
            }

            size_t data_idx = input_layer_size();
            for (size_t i = 1; i < layer_idx; i++)
            {
                size_t n_nodes = layer_sizes()[i];
                data_idx += n_nodes // node values
                    + n_nodes // biases
                    + (n_nodes * layer_sizes()[i - (size_t)1]); // weights
            }
            data_idx += layer_sizes()[layer_idx]; // this layer's values

            return std::span<T>(
                data.data() + data_idx,
                layer_sizes()[layer_idx]
            );
        }

        // weights for a specific node in a layer
        constexpr std::span<T> weights(size_t layer_idx, size_t node_idx)
        {
            if (strict_checks
                && (layer_idx < (size_t)1 || layer_idx >= n_layers))
            {
                throw std::invalid_argument("invalid layer index");
            }

            if (strict_checks && node_idx >= layer_sizes()[layer_idx])
            {
                throw std::invalid_argument("invalid node index");
            }

            size_t data_idx = input_layer_size();
            for (size_t i = 1; i < layer_idx; i++)
            {
                size_t n_nodes = layer_sizes()[i];
                data_idx += n_nodes // node values
                    + n_nodes // biases
                    + (n_nodes * layer_sizes()[i - (size_t)1]); // weights
            }
            data_idx += layer_sizes()[layer_idx] * ((size_t)2 + node_idx);

            return std::span<T>(
                data.data() + data_idx,
                layer_sizes()[layer_idx - (size_t)1]
            );
        }

        // evaluate the model. this function will modify every value in every
        // layer except the input layer.
        void eval()
        {
            for (size_t layer_idx = 1; layer_idx < n_layers; layer_idx++)
            {
                auto prev_layer_values = values(layer_idx - (size_t)1);
                auto this_layer_values = values(layer_idx);
                auto this_layer_biases = biases(layer_idx);
                const auto& activ = activation_fn(layer_idx);

                size_t n_nodes = layer_sizes()[layer_idx];
                for (size_t node_idx = 0; node_idx < n_nodes; node_idx++)
                {
                    auto w = weights(layer_idx, node_idx);

                    T sum = (T)0;
                    for (size_t i = 0; i < w.size(); i++)
                    {
                        sum += w[i] * prev_layer_values[i];
                    }
                    sum += this_layer_biases[node_idx];

                    this_layer_values[node_idx] = activ(sum);
                }
            }
        }

    private:
        std::array<size_t, n_layers> _layer_sizes;
        std::array<std::function<T(T)>, n_layers - (size_t)1> _activation_fns;
        std::array<std::function<T(T)>, n_layers - (size_t)1>
            _activation_derivs;

        std::vector<T> data;

    };

}
