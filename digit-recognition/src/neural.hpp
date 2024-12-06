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

    template<typename T, T fac_for_negatives = (T)0.01>
    T leaky_relu(T v)
    {
        if (v < (T)0)
            return v * fac_for_negatives;
        return v;
    }

    template<typename T, T fac_for_negatives = (T)0.01>
    T leaky_relu_deriv(T v)
    {
        if (v < (T)0)
            return fac_for_negatives;
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
    // 
    // n_layers is the number of layers. it should be at least 2 to represent an
    // input and an output layer.
    // 
    // if store_gradients is false, then the network can only be used for
    // prediction or evaluation, and not training. if store_gradients is true,
    // weight and bias gradients will be stored right next to their
    // corresponding weight or bias. for example, the bias gradient of some node
    // will be stored sizeof(T) bytes after the bias of that node. also, for
    // each layer, an extra array of values will be stored for representing the
    // weighted sum that we got in each node in a forward pass. we'll call these
    // the pre-activation values.
    template<typename T, size_t n_layers, bool store_gradients>
    class Network
    {
    public:
        Network(
            const std::array<size_t, n_layers>& layer_sizes,

            const std::array<std::function<T(T)>, n_layers - 1u>&
            activation_fns,

            const std::array<std::function<T(T)>, n_layers - 1u>&
            activation_derivs
        )
            : _layer_sizes(layer_sizes),
            _activation_fns(activation_fns),
            _activation_derivs(activation_derivs)
        {
            if constexpr (n_layers < 2u)
            {
                throw std::invalid_argument(
                    "n_layers should be at least 2 to represent an input and "
                    "an output layer."
                );
            }

            for (const size_t layer_size : _layer_sizes)
            {
                if (layer_size < 1u)
                {
                    throw std::invalid_argument(
                        "a layer must contain at least 1 node"
                    );
                }
            }

            // if store_gradients is true, weight and bias gradients will be
            // stored right next to their corresponding weight or bias. for
            // example, the bias gradient of some node will be stored sizeof(T)
            // bytes after the bias of that node.

            size_t n_data = _layer_sizes[0];
            if constexpr (store_gradients)
            {
                for (size_t l = 1u; l < n_layers; l++)
                {
                    size_t n_nodes = _layer_sizes[l];
                    n_data += n_nodes // values
                        + n_nodes // pre-activation values
                        + n_nodes * 2u // biases and their gradients
                        // weights and their gradients
                        + n_nodes * _layer_sizes[l - 1u] * 2u;
                }
            }
            else
            {
                for (size_t l = 1u; l < n_layers; l++)
                {
                    size_t n_nodes = _layer_sizes[l];
                    n_data += n_nodes // values
                        + n_nodes // biases
                        + n_nodes * _layer_sizes[l - 1u]; // weights
                }
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
            return layer_sizes()[n_layers - 1u];
        }

        // activation function for a hidden layer or the output layer
        constexpr const std::function<T(T)>& activation_fn(
            size_t layer_idx
        ) const
        {
            if (layer_idx < 1u || layer_idx >= n_layers)
            {
                throw std::invalid_argument("invalid layer index");
            }
            return _activation_fns[layer_idx - 1u];
        }

        // derivative of the activation function for a hidden layer or the
        // output layer.
        constexpr const std::function<T(T)>& activation_deriv(
            size_t layer_idx
        ) const
        {
            if (layer_idx < 1u || layer_idx >= n_layers)
            {
                throw std::invalid_argument("invalid layer index");
            }
            return _activation_derivs[layer_idx - 1u];
        }

        // node values in a layer. values can represent different things in
        // different stages. the values in the first layer always represent the
        // input values. after a forward pass, the values in the hidden layers
        // and the output layer represent the activation values of the nodes.
        // after a backward pass, however, the values in the hidden layers and
        // the output layer represent the gradient of the cost with respect to
        // the activation of the nodes.
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
            if constexpr (store_gradients)
            {
                for (size_t l = 1u; l < layer_idx; l++)
                {
                    size_t n_nodes = layer_sizes()[l];
                    size_t n_prev_nodes = layer_sizes()[l - 1u];
                    data_idx += n_nodes // values
                        + n_nodes // pre-activation values
                        + n_nodes * 2u // biases + gradients
                        + n_nodes * n_prev_nodes * 2u; // weights + gradients
                }
            }
            else
            {
                for (size_t l = 1u; l < layer_idx; l++)
                {
                    size_t n_nodes = layer_sizes()[l];
                    data_idx += n_nodes // values
                        + n_nodes // biases
                        + n_nodes * layer_sizes()[l - 1u]; // weights
                }
            }

            return std::span<T>(
                data.data() + data_idx,
                layer_sizes()[layer_idx]
            );
        }

        // node pre-activation values in a layer. this is just the weighted sum
        // for each node before it went through the activation function. this
        // is only available when store_gradients is true.
        constexpr std::span<T> pre_activ(size_t layer_idx)
        {
            if constexpr (!store_gradients)
            {
                throw std::logic_error(
                    "pre-activation values are only stored when "
                    "store_gradients is true"
                );
            }

            if (layer_idx < 1u || layer_idx >= n_layers)
            {
                throw std::invalid_argument("invalid layer index");
            }

            size_t data_idx = input_size();
            for (size_t l = 1u; l < layer_idx; l++)
            {
                size_t n_nodes = layer_sizes()[l];
                size_t n_prev_nodes = layer_sizes()[l - 1u];
                data_idx += n_nodes // values
                    + n_nodes // pre-activation values
                    + n_nodes * 2u // biases + gradients
                    + n_nodes * n_prev_nodes * 2u; // weights + gradients
            }
            data_idx += layer_sizes()[layer_idx]; // this layer's values

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

        // node biases in a layer. if store_gradients is true, then every bias
        // value will be immediately followed by its gradient.
        constexpr std::span<T> biases(size_t layer_idx)
        {
            if (layer_idx < 1u || layer_idx >= n_layers)
            {
                throw std::invalid_argument("invalid layer index");
            }

            size_t data_idx = input_size();
            if constexpr (store_gradients)
            {
                for (size_t l = 1u; l < layer_idx; l++)
                {
                    size_t n_nodes = layer_sizes()[l];
                    size_t n_prev_nodes = layer_sizes()[l - 1u];
                    data_idx += n_nodes // values
                        + n_nodes // pre-activation values
                        + n_nodes * 2u // biases + gradients
                        + n_nodes * n_prev_nodes * 2u; // weights + gradients
                }

                // this layer's values and pre-activation values
                data_idx += layer_sizes()[layer_idx] * 2u;

                return std::span<T>(
                    data.data() + data_idx,
                    layer_sizes()[layer_idx] * 2u
                );
            }
            else
            {
                for (size_t l = 1u; l < layer_idx; l++)
                {
                    size_t n_nodes = layer_sizes()[l];
                    data_idx += n_nodes // node values
                        + n_nodes // biases
                        + (n_nodes * layer_sizes()[l - 1u]); // weights
                }
                data_idx += layer_sizes()[layer_idx]; // this layer's values

                return std::span<T>(
                    data.data() + data_idx,
                    layer_sizes()[layer_idx]
                );
            }
        }

        // weights for a specific node in a layer. if store_gradients is true,
        // then every weight value will be immediately followed by its gradient.
        constexpr std::span<T> weights(size_t layer_idx, size_t node_idx)
        {
            if (layer_idx < 1u || layer_idx >= n_layers)
            {
                throw std::invalid_argument("invalid layer index");
            }

            if (node_idx >= layer_sizes()[layer_idx])
            {
                throw std::invalid_argument("invalid node index");
            }

            size_t data_idx = input_size();
            if constexpr (store_gradients)
            {
                for (size_t l = 1u; l < layer_idx; l++)
                {
                    size_t n_nodes = layer_sizes()[l];
                    size_t n_prev_nodes = layer_sizes()[l - 1u];
                    data_idx += n_nodes // values
                        + n_nodes // pre-activation values
                        + n_nodes * 2u // biases + gradients
                        + n_nodes * n_prev_nodes * 2u; // weights + gradients
                }

                size_t n_nodes = layer_sizes()[layer_idx];
                size_t n_prev_nodes = layer_sizes()[layer_idx - 1u];

                // this layer's values and pre-activation values
                data_idx += layer_sizes()[layer_idx] * 2u;

                // biases + gradients
                data_idx += n_nodes * 2u;

                // weights + gradients for nodes up to node_idx
                data_idx += node_idx * n_prev_nodes * 2u;

                return std::span<T>(
                    data.data() + data_idx,
                    n_prev_nodes * 2u
                );
            }
            else
            {
                for (size_t l = 1u; l < layer_idx; l++)
                {
                    size_t n_nodes = layer_sizes()[l];
                    data_idx += n_nodes // node values
                        + n_nodes // biases
                        + (n_nodes * layer_sizes()[l - 1u]); // weights
                }

                size_t n_nodes = layer_sizes()[layer_idx];
                size_t n_prev_nodes = layer_sizes()[layer_idx - 1u];

                data_idx += n_nodes * 2u; // this layer's values & biases
                data_idx += node_idx * n_prev_nodes; // weights

                return std::span<T>(
                    data.data() + data_idx,
                    n_prev_nodes
                );
            }
        }

        // randomize weights and biases using custom distributions
        template<
            typename RandomEngine,
            typename WeightDistribution,
            typename BiasDistribution
        >
        void randomize(
            RandomEngine& engine,
            WeightDistribution& weight_dist,
            BiasDistribution& bias_dist
        )
        {
            if constexpr (store_gradients)
            {
                for (size_t l = 1u; l < n_layers; l++)
                {
                    auto b = biases(l);
                    for (size_t i = 0u; i < b.size(); i += 2u)
                    {
                        b[i] = bias_dist(engine);
                    }

                    for (size_t n = 0u; n < layer_sizes()[l]; n++)
                    {
                        auto w = weights(l, n);
                        for (size_t i = 0u; i < w.size(); i += 2u)
                        {
                            w[i] = weight_dist(engine);
                        }
                    }
                }
            }
            else
            {
                for (size_t l = 1u; l < num_layers(); l++)
                {
                    for (auto& v : biases(l))
                    {
                        v = bias_dist(engine);
                    }
                    for (size_t n = 0u; n < layer_sizes()[l]; n++)
                    {
                        for (auto& v : weights(l, n))
                        {
                            v = weight_dist(engine);
                        }
                    }
                }
            }
        }

        // randomize weights and biases using a uniform distribution
        template<typename RandomEngine>
        void randomize_uniform(
            RandomEngine& engine,
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
            randomize(engine, weight_dist, bias_dist);
        }

        // randomize weights using Uniform Xavier Initialization and randomize
        // biases using a uniform distribution.
        // https://www.geeksforgeeks.org/xavier-initialization
        template<typename RandomEngine, T(*sqrt_fn)(T) = std::sqrt>
        void randomize_xavier_uniform(
            RandomEngine& engine,
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

            randomize(engine, weight_dist, bias_dist);
        }

        // randomize weights using Normal Xavier Initialization and randomize
        // biases using a uniform distribution.
        // https://www.geeksforgeeks.org/xavier-initialization
        template<typename RandomEngine, T(*sqrt_fn)(T) = std::sqrt>
        void randomize_xavier_normal(
            RandomEngine& engine,
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

            randomize(engine, weight_dist, bias_dist);
        }

        // zero out weight and bias gradients in a layer
        void zero_gradients(size_t layer_idx)
        {
            if constexpr (!store_gradients)
            {
                throw std::logic_error(
                    "can't zero out gradients when store_gradients is false"
                );
            }

            if (layer_idx < 1u || layer_idx >= n_layers)
            {
                throw std::invalid_argument("invalid layer index");
            }

            auto b = biases(layer_idx);
            for (size_t i = 1u; i < b.size(); i += 2u)
            {
                b[i] = (T)0;
            }

            for (size_t n = 0u; n < layer_sizes()[n]; n++)
            {
                auto w = weights(layer_idx, n);
                for (size_t i = 1u; i < w.size(); i += 2u)
                {
                    w[i] = (T)0;
                }
            }
        }

        // zero out weight and bias gradients in all layers
        void zero_gradients()
        {
            if constexpr (!store_gradients)
            {
                throw std::logic_error(
                    "can't zero out gradients when store_gradients is false"
                );
            }

            for (size_t l = 1u; l < n_layers; l++)
            {
                auto b = biases(l);
                for (size_t i = 1u; i < b.size(); i += 2u)
                {
                    b[i] = (T)0;
                }

                for (size_t n = 0u; n < layer_sizes()[n]; n++)
                {
                    auto w = weights(l, n);
                    for (size_t i = 1u; i < w.size(); i += 2u)
                    {
                        w[i] = (T)0;
                    }
                }
            }
        }

        // evaluate the model. this will modify every value in every layer
        // except the input layer. if store_gradients is true, this will modify
        // all pre-activation values as well.
        void forward_pass()
        {
            for (size_t layer_idx = 1u; layer_idx < n_layers; layer_idx++)
            {
                auto prev_layer_values = values(layer_idx - 1u);
                auto this_layer_values = values(layer_idx);
                auto this_layer_biases = biases(layer_idx);
                const auto& activ = activation_fn(layer_idx);

                std::span<T> this_layer_pre_activ;
                if constexpr (store_gradients)
                {
                    this_layer_pre_activ = pre_activ(layer_idx);
                }

                const size_t n_nodes = layer_sizes()[layer_idx];
                const size_t n_prev_nodes = layer_sizes()[layer_idx - 1];

                if constexpr (store_gradients)
                {
                    for (size_t node_idx = 0u; node_idx < n_nodes; node_idx++)
                    {
                        auto w = weights(layer_idx, node_idx);

                        T weighted_sum = (T)0;
                        for (size_t i = 0u; i < n_prev_nodes; i++)
                        {
                            weighted_sum += w[i * 2u] * prev_layer_values[i];
                        }
                        weighted_sum += this_layer_biases[node_idx * 2u];

                        this_layer_pre_activ[node_idx] = weighted_sum;
                        this_layer_values[node_idx] = activ(weighted_sum);
                    }
                }
                else
                {
                    for (size_t node_idx = 0u; node_idx < n_nodes; node_idx++)
                    {
                        auto w = weights(layer_idx, node_idx);

                        T weighted_sum = (T)0;
                        for (size_t i = 0u; i < n_prev_nodes; i++)
                        {
                            weighted_sum += w[i] * prev_layer_values[i];
                        }
                        weighted_sum += this_layer_biases[node_idx];

                        this_layer_values[node_idx] = activ(weighted_sum);
                    }
                }
            }
        }

        // calculate the gradient of the cost function with respect to every
        // weight and bias using backpropagation.
        // if accumulate_gradients is true, then we'll only add values onto
        // weight and bias gradients instead of replacing them entirely. this is
        // useful for averaging gradients for several training examples, but
        // keep in mind to call zero_gradients() first to reset the gradients,
        // and divide the final gradients by the number of training examples.
        template<bool accumulate_gradients>
        void backward_pass(std::span<T> input, std::span<T> expected_output)
        {
            // Note to others and future self:
            // First of all, I highly suggest checking out the helpful links
            // provided in README.md.
            // I know the maths might be confusing at first. It took me nearly 3
            // days of watching videos, reading articles, and asking fellow NN
            // enthusiasts on the internet for help before I finally had an idea
            // of how to implement backpropagation.
            // The basic idea is that, if you know how the activations in a
            // layer affect the cost function (dcost_dact), then you can
            // calculate all the weight and bias gradients in that layer
            // (gradient of the cost function with respect to those weights or
            // biases).
            // Now, calculating this dcost_dact vector for the output layer is
            // straightforward, because you only need your predicted output
            // values and the expected output values to calculate it (and, of
            // course, the cost function itself). However, for other layers, we
            // don't know how their activations affect the cost function. In
            // other words, we don't know their dcost_dact. So once we calculate
            // the output layer's dcost_dact, we use those and the weights in
            // the output layer to calculate the second-to-last layer's
            // dcost_dact. And then the cycles repeats, because now we know
            // the dcost_dact of the second-to-last layer, so we can calculate
            // its weight and bias gradients just like before, and then we can
            // calculate dcost_dact for the layer before that, and then
            // calculate its gradient, and so on and so forth.
            // Note that we use the chain rule from calculus to compute said
            // gradients. Backpropagation is just a way to avoid duplicate
            // calculations.

            if constexpr (!store_gradients)
            {
                throw std::logic_error(
                    "can't do backward pass when store_gradients is false"
                );
            }

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
            forward_pass();

            // calculate the gradient of the cost function with respect to the
            // node activations in the output layer. before we modify these
            // values, they're just the current output or prediction of the
            // network that we got after the forward pass.
            auto dcost_dout = output_values();
            for (size_t i = 0u; i < output_size(); i++)
            {
                dcost_dout[i] = (T)2 * (dcost_dout[i] - expected_output[i]);
            }

            // start from the last layer (output layer) and go backward
            for (ptrdiff_t l = n_layers - 1u; l >= 1; l--)
            {
                // gradient of the activation function with respect to the
                // current node's pre-activation value.
                const auto& dact_dz = activation_deriv(l);

                // gradient of the cost function with respect to the activations
                // in the current layer.
                auto dcost_dact = values(l);

                auto prev_layer_values = values(l - 1);
                auto this_layer_pre_activ = pre_activ(l);
                auto this_layer_biases = biases(l);

                // calculate the gradient of the cost function with respect to
                // the weights and biases.
                for (size_t n = 0u; n < layer_sizes()[l]; n++)
                {
                    // gradient of the cost function with respect to the current
                    // node's pre-activation value.
                    TODO_cache_dcost_dz_with_double_buffering;
                    T dcost_dz =
                        dcost_dact[n]
                        * dact_dz(this_layer_pre_activ[n]);

                    // bias gradient
                    if constexpr (accumulate_gradients)
                    {
                        this_layer_biases[n * 2u + 1u] += dcost_dz;
                    }
                    else
                    {
                        this_layer_biases[n * 2u + 1u] = dcost_dz;
                    }

                    // weight gradients
                    auto w = weights(l, n);
                    if constexpr (accumulate_gradients)
                    {
                        for (size_t pn = 0u; pn < layer_sizes()[l - 1]; pn++)
                        {
                            w[pn * 2u + 1u] += dcost_dz * prev_layer_values[pn];
                        }
                    }
                    else
                    {
                        for (size_t pn = 0u; pn < layer_sizes()[l - 1]; pn++)
                        {
                            w[pn * 2u + 1u] = dcost_dz * prev_layer_values[pn];
                        }
                    }
                }

                if (l <= 1)
                    continue;

                // calculate the gradient of the cost function with respect to
                // the node activations in the previous layer.
                auto prev_dcost_dact = values(l - 1);
                for (size_t pn = 0u; pn < layer_sizes()[l - 1]; pn++)
                {
                    T dcost_dact_pn = (T)0;
                    for (size_t n = 0u; n < layer_sizes()[l]; n++)
                    {
                        TODO_cache_dcost_dz_with_double_buffering;
                        T dcost_dz =
                            dcost_dact[n]
                            * dact_dz(this_layer_pre_activ[n]);

                        dcost_dact_pn += dcost_dz * weights(l, n)[pn * 2u];
                    }

                    prev_dcost_dact[pn] = dcost_dact_pn;
                }
            }
        }

        // perform averaged backward pass for more than one training example
        // (data point).
        void averaged_backward_pass()
        {
            if constexpr (!store_gradients)
            {
                throw std::logic_error(
                    "can't do averaged backward pass when store_gradients is "
                    "false"
                );
            }

            TODO;
        }

        // calculate the cost for a given data point using squared error loss
        // (SEL). this will modify every value in every layer.
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
            forward_pass();

            T c = (T)0;
            auto output = output_values();
            for (size_t i = 0u; i < output.size(); i++)
            {
                T diff = output[i] - expected_output[i];
                c += (diff * diff);
            }
            return c;
        }

        // calculate the average cost for given data points using squared error
        // loss (SEL). this will modify every value in every layer.
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
            for (size_t i = 0u; i < n_data_points; i++)
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
        std::array<std::function<T(T)>, n_layers - 1u> _activation_fns;
        std::array<std::function<T(T)>, n_layers - 1u>
            _activation_derivs;

        std::vector<T> data;

    };

}
