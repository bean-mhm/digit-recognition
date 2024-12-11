#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <format>
#include <variant>
#include <optional>
#include <array>
#include <vector>
#include <span>
#include <memory>
#include <functional>
#include <random>
#include <stdexcept>
#include <cmath>
#include <cstdint>

#ifndef GLEW_STATIC
#define GLEW_STATIC
#endif
#include <GL/glew.h>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "GLFW/glfw3.h"

#include "neural.hpp"
#include "endian.hpp"
#include "stream.hpp"
#include "str.hpp"

namespace digit_rec
{

    static constexpr auto WINDOW_TITLE = "Digit Recognition - bean-mhm";

    // width and height of the window
    static constexpr uint32_t WINDOW_WIDTH = 800;
    static constexpr uint32_t WINDOW_HEIGHT = WINDOW_WIDTH * 3u / 4u;

    // the amount of horizontal and vertical padding in the window, proportional
    // to the window width.
    static constexpr float WINDOW_PAD = .04f;

    // spacing between the 2 columns in the settings layout, proportional to the
    // window height.
    static constexpr float COLUMN_SPACING = .015f;

    static constexpr ImVec4 COLOR_BG{ .043f, .098f, .141f, 1.f };

    static constexpr float FONT_SIZE = 21.f;
    static constexpr auto FONT_PATH =
        "./JetBrainsMono/JetBrainsMono-Regular.ttf";

    static constexpr auto TRAIN_IMAGES_PATH =
        "./MNIST/train-images.idx3-ubyte";
    static constexpr auto TRAIN_LABELS_PATH =
        "./MNIST/train-labels.idx1-ubyte";
    static constexpr auto TEST_IMAGES_PATH = "./MNIST/t10k-images.idx3-ubyte";
    static constexpr auto TEST_LABELS_PATH = "./MNIST/t10k-labels.idx1-ubyte";

    static constexpr size_t DIGIT_WIDTH = 28;
    static constexpr size_t DIGIT_HEIGHT = 28;
    static constexpr size_t N_DIGIT_VALUES = DIGIT_WIDTH * DIGIT_HEIGHT;

    struct DigitSample
    {
        // pixel values for a digit stored in a row major format
        std::array<uint8_t, N_DIGIT_VALUES> values;

        // digit label from 0 to 9. I didn't use uint8_t for better alignment.
        uint32_t label;
    };

    enum class UiMode
    {
        Settings,
        Training,
        Drawboard
    };

    enum class ActivationFunc : int
    {
        Relu,
        LeakyRelu,
        Tanh
    };
    static constexpr const char* ActivationFunc_str[] = {
        "ReLU",
        "Leaky ReLU",
        "Tanh"
    };

    class App
    {
    public:
        App() = default;
        void run();

    private:
        void init();
        void loop();
        void cleanup();

    private:
        GLFWwindow* window = nullptr;
        ImGuiIO* io = nullptr;
        ImFont* font = nullptr;

        UiMode ui_mode = UiMode::Settings;

        char val_layer_sizes[64]{};
        float val_learning_rate = .01f;
        ActivationFunc val_hidden_activation = ActivationFunc::LeakyRelu;
        ActivationFunc val_output_activation = ActivationFunc::Tanh;
        uint32_t val_batch_size = 100;
        uint32_t val_seed = 12345678;
        bool val_random_transform = true;

        std::vector<DigitSample> train_samples;
        std::vector<DigitSample> test_samples;
        std::unique_ptr<neural::Network<float, true>> net = nullptr;

        // accuracy of the network over time
        std::vector<float> accuracy_history;

        void init_ui();
        void draw_ui();

        void layout_settings();
        void layout_training();
        void layout_drawboard();

        void load_digit_samples(
            std::string_view images_path,
            std::string_view labels_path,
            std::vector<DigitSample>& out_samples
        );

        // returns std::nullopt on success, and an error message on failure.
        std::optional<std::string> start_training();

        void stop_training();

    };

}