#pragma once

#include <array>
#include <vector>
#include <span>
#include <functional>
#include <random>
#include <stdexcept>
#include <cmath>
#include <cstdint>

namespace neural
{

    template<typename T, T(*exp_fn)(T) = std::exp>
    T gaussian_distribution(T mean, T standard_deviation, T x)
    {
        static constexpr T inv_sqrt_2pi = (T)0.39894228040143267793994605993439;

        T a = (x - mean) / standard_deviation;
        return exp_fn((T)(-.5) * a * a) * inv_sqrt_2pi / standard_deviation;
    }

    template<typename T, T(*exp_fn)(T) = std::exp>
    T gaussian_distribution(T standard_deviation, T x)
    {
        static constexpr T inv_sqrt_2pi = (T)0.39894228040143267793994605993439;

        T a = x / standard_deviation;
        return exp_fn((T)(-.5) * a * a) * inv_sqrt_2pi / standard_deviation;
    }

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

    template<typename T, T(*tanh_fn)(T) = std::tanh>
    T tanh(T v)
    {
        return tanh_fn(v);
    }

    template<typename T, T(*tanh_fn)(T) = std::tanh>
    T tanh_deriv(T v)
    {
        T th = tanh_fn(v);
        return (T)1 - (th * th);
    }

    template<typename T, T(*exp_fn)(T) = std::exp>
    T logistic(T v)
    {
        return (T)1 / ((T)1 + exp_fn(-v));
    }

    template<typename T, T(*exp_fn)(T) = std::exp>
    T logistic_deriv(T v)
    {
        T a = exp_fn(-v);
        T b = (T)1 + a;
        return a / (b * b);
    }

    // T is the type used to store numerical values. A typical value may
    // be `float`.
    // n_layers is the number of layers. it should be at least 2 to represent an
    // input and an output layer.
    template<typename T, size_t n_layers>
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

        constexpr const size_t input_size() const
        {
            return layer_sizes()[0];
        }

        constexpr const size_t output_size() const
        {
            return layer_sizes()[n_layers - (size_t)1];
        }

        // activation function for a hidden layer or the output layer
        constexpr const std::function<T(T)>& activation_fn(
            size_t layer_idx
        ) const
        {
            if (layer_idx < (size_t)1 || layer_idx >= n_layers)
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
            if (layer_idx < (size_t)1 || layer_idx >= n_layers)
            {
                throw std::invalid_argument("invalid layer index");
            }
            return _activation_derivs[layer_idx - (size_t)1];
        }

        // node values in a layer
        constexpr std::span<T> values(size_t layer_idx)
        {
            if (layer_idx >= n_layers)
            {
                throw std::invalid_argument("invalid layer index");
            }

            if (layer_idx == 0)
            {
                return std::span<T>(data.data(), input_size());
            }

            size_t data_idx = input_size();
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
            if (layer_idx < (size_t)1 || layer_idx >= n_layers)
            {
                throw std::invalid_argument("invalid layer index");
            }

            size_t data_idx = input_size();
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
            if (layer_idx < (size_t)1 || layer_idx >= n_layers)
            {
                throw std::invalid_argument("invalid layer index");
            }

            if (node_idx >= layer_sizes()[layer_idx])
            {
                throw std::invalid_argument("invalid node index");
            }

            size_t data_idx = input_size();
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

        // randomize weights and biases using a uniform distribution
        template<typename RandomEngine>
        void randomize_uniform(
            RandomEngine engine,
            T min_weight,
            T max_weight,
            T min_bias,
            T max_bias
        )
        {
            std::uniform_real_distribution<T> weight_dist(
                min_weight,
                max_weight
            );

            std::uniform_real_distribution<T> bias_dist(
                min_bias,
                max_bias
            );

            for (size_t i = 1; i < num_layers(); i++)
            {
                for (auto& v : biases(i))
                {
                    v = bias_dist(engine);
                }
                for (size_t j = 0; j < layer_sizes()[i]; j++)
                {
                    for (auto& v : weights(i, j))
                    {
                        v = weight_dist(engine);
                    }
                }
            }
        }

        // randomize weights using Uniform Xavier Initialization and randomize
        // biases using a uniform distribution.
        // https://www.geeksforgeeks.org/xavier-initialization
        template<typename RandomEngine, T(*sqrt_fn)(T) = std::sqrt>
        void randomize_xavier_uniform(
            RandomEngine engine,
            T min_bias,
            T max_bias
        )
        {
            const T weight_range = sqrt_fn(
                (T)6 / (T)(input_size() + output_size())
            );
            std::uniform_real_distribution<T> weight_dist(
                -weight_range,
                weight_range
            );

            std::uniform_real_distribution<T> bias_dist(
                min_bias,
                max_bias
            );

            for (size_t i = 1; i < num_layers(); i++)
            {
                for (auto& v : biases(i))
                {
                    v = bias_dist(engine);
                }
                for (size_t j = 0; j < layer_sizes()[i]; j++)
                {
                    for (auto& v : weights(i, j))
                    {
                        v = weight_dist(engine);
                    }
                }
            }
        }

        // randomize weights using Normal Xavier Initialization and randomize
        // biases using a uniform distribution.
        // https://www.geeksforgeeks.org/xavier-initialization
        template<typename RandomEngine, T(*sqrt_fn)(T) = std::sqrt>
        void randomize_xavier_normal(
            RandomEngine engine,
            T min_bias,
            T max_bias
        )
        {
            T standard_dev = sqrt_fn(
                (T)2 / (T)(input_size() + output_size())
            );
            std::normal_distribution<T> weight_dist((T)0, standard_dev);

            std::uniform_real_distribution<T> bias_dist(
                min_bias,
                max_bias
            );

            for (size_t i = 1; i < num_layers(); i++)
            {
                for (auto& v : biases(i))
                {
                    v = bias_dist(engine);
                }
                for (size_t j = 0; j < layer_sizes()[i]; j++)
                {
                    for (auto& v : weights(i, j))
                    {
                        v = weight_dist(engine);
                    }
                }
            }
        }

        // evaluate the model (forward pass). this function will modify every
        // value in every layer except the input layer.
        void eval()
        {
            for (size_t layer_idx = 1; layer_idx < n_layers; layer_idx++)
            {
                auto prev_layer_values = values(layer_idx - (size_t)1);
                auto this_layer_values = values(layer_idx);
                auto this_layer_biases = biases(layer_idx);
                const auto& activ = activation_fn(layer_idx);

                const size_t n_nodes = layer_sizes()[layer_idx];
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

        // update the input values based on a given argument, evaluate the
        // network, and return the output values.
        std::span<T> eval(std::span<T> input)
        {
            if (input.size() != input_size())
            {
                throw std::invalid_argument("invalid input data size");
            }

            input_values() = input;
            eval();
            return output_values();
        }

        // calculate the cost for a given data point. this will modify every
        // value in every layer.
        T cost(std::span<T> input, std::span<T> expected_output)
        {
            if (input.size() != input_size())
            {
                throw std::invalid_argument(
                    "invalid input data size"
                );
            }

            if (expected_output.size() != output_size())
            {
                throw std::invalid_argument(
                    "invalid expected output data size"
                );
            }

            input_values() = input;
            eval();

            T c = (T)0;
            auto output = output_values();
            for (size_t i = 0; i < output.size(); i++)
            {
                T diff = output[i] - expected_output[i];
                c += (diff * diff);
            }
            return c;
        }

        // calculate the average cost for given data points. this will modify
        // every value in every layer.
        // * data.size() must be a multiple of (input_size() + output_size()).
        // * data must contain chunks of input data and the corresponding
        //   expected output data.
        T average_cost(std::span<T> data_points)
        {
            const size_t data_point_size = input_size() + output_size();
            const size_t n_data_points = data_points.size() / data_point_size;
            if (data_points.size() % data_point_size != 0)
            {
                throw std::invalid_argument("invalid data size");
            }

            T c = (T)0;
            for (size_t i = 0; i < n_data_points; i++)
            {
                c += cost(
                    data_points.subspan(
                        i * data_point_size,
                        input_size()
                    ),
                    data_points.subspan(
                        i * data_point_size + input_size(),
                        output_size()
                    )
                );
            }
            c /= (T)n_data_points;
            return c;
        }

    private:
        std::array<size_t, n_layers> _layer_sizes;
        std::array<std::function<T(T)>, n_layers - (size_t)1> _activation_fns;
        std::array<std::function<T(T)>, n_layers - (size_t)1>
            _activation_derivs;

        std::vector<T> data;

    };

}
