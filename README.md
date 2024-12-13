# Introduction

This a little program I wrote that lets you train a
[Deep Neural Network (DNN)](https://en.wikipedia.org/wiki/Neural_network_(machine_learning))
to recognize handwritten digits.

This project uses the
[MNIST Handwritten Digits Dataset](https://yann.lecun.com/exdb/mnist/)
for training and testing a neural network. The dataset is not included in the
repository for license issues. The user may create a subdirectory in the
program's binary directory named "MNIST" and copy the 4 IDX files there at their
own risk.

# How It's Made

This project is written in C++ with Visual Studio 2022. The target platform is
Windows. However, considering all the libraries used are platform-independent,
it should be fairly easy to port to other major desktop platforms.

# Libraries Used

| Library | Used for |
|--|--|
| [GLEW](https://glew.sourceforge.net/) | OpenGL extensions |
| [GLFW](https://www.glfw.org/) | Window and OpenGL context creation |
| [Dear ImGui](https://github.com/ocornut/imgui) | Graphical user interface |

The main GUI font is [Outfit](https://fonts.google.com/specimen/Outfit?preview.text=Digit%20Recognition%20Train%20Layer%20Batch%20Seed%20Accuracy).

# Deep Neural Networks

A deep neural network consists of an **input layer**, one or more **hidden layers**,
and an **output layer**. Each layer contains a few **nodes** or **neurons**. A node simply
holds a single value, called its **activation value**. The input layer holds our
input data (like the digit we want to evaluate), and the output layer holds the
network's **prediction**. In this case, it holds 10 values that represent how much
the networks thinks the input matches digits 0 to 9.

Once a network is trained, we can get a prediction from it by doing a
**forward pass**. In a forward pass, we start from the first hidden layer, and for each of
its nodes, we calculate a **weighted sum** of the activations from the previous
layer (the input layer), add a **bias** value to it, and finally, put it through a
custom **activation function** (for example, a sigmoid function). Then, we repeat
this process for the next layer until we reach the output layer.

# How They Learn

If we could somehow adjust the weights and biases properly for every node, we
could get the network to produce more accurate predictions. This is called
**training**.

## Cost Function

We can define a function which tells us how accurate a network is. The smaller
the cost value, the more accurate the network is. There are several methods for
calculating the cost, but a popular one
([Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error))
is to simply add up

```(expected value - network prediction) ^ 2```

for every node in the output layer, and average this cost value for every
training example (digit samples from the training dataset), or at least a subset
of them (called a **batch**) to make things faster.

## Backpropagation

We can then use the [**Chain Rule**](https://en.wikipedia.org/wiki/Chain_rule)
from calculus to calculate how every weight and bias in every layer affects
the final averaged cost value. This is also called the derivative of the cost
function with respect to each weight and bias. In my code, I simply call these
values weight and bias gradients.

We make use of an algorithm named
[**Backpropagation**](https://en.wikipedia.org/wiki/Backpropagation)
to make the process of
calculating weight and bias gradients a whole lot faster. Basically, it starts
from the last layer and moves backwards, and in each step, calculates the
gradient of the cost function with respect to the pre-activation values
(the weighted sums before being put into activation functions) of the nodes in
the current layer, and uses that gradient to calculate how each weight and bias
in the current layer affects the final cost. Again, read about the chain rule,
and check out the links at the end of this document, they're helpful.

## Gradient Descent

Imagine a 1D polynomial function. Let's say we're on a point on this function,
and we want to reach a local minimum. We could use methods from calculus to
find the local minima, but another method is to simply calculate the derivative
of the function's output with respect to its input at the current point, and
simply move slightly in that direction, effectively treating the function as a
line that passes through that point and has the same slope as the original
function at that point. We can then repeat this process many times until we
reach a local minimum, a valley.

The same goes for the cost function. We can treat it as a huge multi-dimensional
function that takes in our weights and biases and outputs a single cost value.
Once we know how every weight and bias affects the averaged cost value for a
given batch (subset of the training dataset), or in other words, the gradient of
the cost with respect to the weights and biases, we can adjust the weights and
biases in the opposite direction of the gradient to make the cost smaller.

```
new_weight = old_weight - (gradient * learning_rate)
```

Before subtracting the gradient value, we scale it by a tiny value like 0.01.
This is called the learning rate. If it's too big, we'll jump around the
multi-dimensional hyperspace of possible weight and bias values and never
converge to a local minimum. If it's too small, training might get too slow,
so we need to pick something in the middle.

# Helpful Resources

It took me a couple days before I understood how to implement
backpropagation, so don't lose hope if the whole thing seems complicated and
huge. That's because it is!

These videos and articles helped me get a much better idea of how neural
networks learn, so I suggest you check them out too.

- [How to Create a Neural Network - Sebastian Lague](https://www.youtube.com/watch?v=hfMk-kjRv4c)

- [Neural Networks (Playlist) - 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

- [The Math Behind Backpropagation - Maximinusjoshus](https://medium.com/featurepreneur/the-mathematics-of-backpropagation-4b114fd64a63)
