#include "app_digit_rec.hpp"

namespace digit_rec
{

    void App::run()
    {
        init();
        while (!glfwWindowShouldClose(window))
        {
            loop();
        }
        cleanup();
    }

    static void glfw_error_callback(int error, const char* description)
    {
        throw std::runtime_error(std::format(
            "GLFW error {}: {}",
            error,
            description
        ));
    }

    void App::init()
    {
        load_digit_samples(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, train_samples);
        load_digit_samples(TEST_IMAGES_PATH, TEST_LABELS_PATH, test_samples);

        init_ui();
    }

    void App::loop()
    {
        draw_ui();
    }

    void App::cleanup()
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void App::init_ui()
    {
        // initialize GLFW
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit())
            throw std::runtime_error("failed to initialize GLFW");

        // OpenGL 3.2 + GLSL 150
        static constexpr auto glsl_version = "#version 150";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // required on Mac

        // create window
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            WINDOW_TITLE,
            nullptr,
            nullptr
        );
        if (!window)
        {
            glfwTerminate();
            throw std::runtime_error("failed to create window");
        }

        // make the window's context current and enable VSync
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        // initialize GLEW for loading OpenGL extensions
        glewExperimental = GL_TRUE;
        GLenum glew_init_result = glewInit();
        if (glew_init_result != GLEW_OK)
        {
            throw std::runtime_error(std::format(
                "failed to initialize GLEW: {}",
                glew_init_result
            ));
        }

        // setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        io = &ImGui::GetIO();

        // enable keyboard and gamepad
        io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io->ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

        // load style
        ImGui::StyleColorsDark();

        // setup backends
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glsl_version);

        // load fonts
        font = io->Fonts->AddFontFromFileTTF(FONT_PATH, FONT_SIZE);
        if (!font)
        {
            throw std::runtime_error("failed to load fonts");
        }
    }

    static constexpr float scaled(float size)
    {
        return size * (float)WINDOW_WIDTH;
    }

    void App::draw_ui()
    {
        // poll and handle events (inputs, window resize, etc.)
        // you can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to
        // tell if dear imgui wants to use your inputs.
        // - when io.WantCaptureMouse is true, do not dispatch mouse input data
        //   to your main application, or clear/overwrite your copy of the mouse
        //   data.
        // - when io.WantCaptureKeyboard is true, do not dispatch keyboard input
        //   data to your main application, or clear/overwrite your copy of the
        //   keyboard data.
        // generally you may always pass all inputs to dear imgui, and hide them
        // from your application based on those two flags.
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0)
        {
            ImGui_ImplGlfw_Sleep(10);
            return;
        }

        // start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // layout
        {
            ImGui::SetNextWindowPos({
                0.f,
                scaled(.6f * WINDOW_PAD)
                });
            ImGui::SetNextWindowSize({
                (float)WINDOW_WIDTH,
                (float)WINDOW_HEIGHT - 2.f * scaled(.6f * WINDOW_PAD)
                });
            ImGui::Begin(
                "##mainwindow",
                nullptr,
                ImGuiWindowFlags_NoTitleBar
                | ImGuiWindowFlags_NoResize
                | ImGuiWindowFlags_NoMove
                | ImGuiWindowFlags_NoScrollbar
                | ImGuiWindowFlags_NoCollapse
                | ImGuiWindowFlags_NoBackground
                | ImGuiWindowFlags_NoSavedSettings
            );

            switch (ui_mode)
            {
            case digit_rec::UiMode::Settings:
                layout_settings();
                break;
            case digit_rec::UiMode::Training:
                layout_training();
                break;
            case digit_rec::UiMode::Drawboard:
                layout_drawboard();
                break;
            default:
                break;
            }

            ImGui::End();
        }

        // rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(COLOR_BG.x, COLOR_BG.y, COLOR_BG.z, COLOR_BG.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // swap front and back buffers
        glfwSwapBuffers(window);
    }

    void App::layout_settings()
    {
        static constexpr float COLUMN_WIDTH = scaled(
            .5f - COLUMN_SPACING - WINDOW_PAD
        );
        static constexpr float COLUMN_0_START = scaled(WINDOW_PAD);
        static constexpr float COLUMN_1_START = scaled(.5f + COLUMN_SPACING);

        ImGui::SameLine(COLUMN_0_START);
        ImGui::SetNextItemWidth(COLUMN_WIDTH);
        ImGui::Text("Layer Sizes");

        ImGui::SameLine(COLUMN_1_START);
        ImGui::SetNextItemWidth(COLUMN_WIDTH);
        ImGui::Text("Learning Rate");

        ImGui::NewLine();

        static bool init_val_layer_sizes = false;
        if (!init_val_layer_sizes)
        {
            init_val_layer_sizes = true;
            sprintf_s(
                val_layer_sizes,
                sizeof(val_layer_sizes) / sizeof(char),
                "%zu, 16, 16, 10",
                N_DIGIT_VALUES
            );
        }
        ImGui::SameLine(COLUMN_0_START);
        ImGui::SetNextItemWidth(COLUMN_WIDTH);
        ImGui::InputText("##layersizes", val_layer_sizes, 64);

        ImGui::SameLine(COLUMN_1_START);
        ImGui::SetNextItemWidth(COLUMN_WIDTH);
        ImGui::SliderFloat(
            "##learnrate",
            &val_learning_rate,
            0.f,
            1.f,
            "%.4f",
            ImGuiSliderFlags_AlwaysClamp
            | ImGuiSliderFlags_Logarithmic
        );

        ImGui::NewLine();
        ImGui::NewLine();

        //

        ImGui::SameLine(COLUMN_0_START);
        ImGui::SetNextItemWidth(COLUMN_WIDTH);
        ImGui::Text("Hidden Layer Activation");

        ImGui::SameLine(COLUMN_1_START);
        ImGui::SetNextItemWidth(COLUMN_WIDTH);
        ImGui::Text("Output Layer Activation");

        ImGui::NewLine();

        ImGui::SameLine(COLUMN_0_START);
        ImGui::SetNextItemWidth(COLUMN_WIDTH);
        ImGui::Combo(
            "##hiddenact",
            reinterpret_cast<int*>(&val_hidden_activation),
            ActivationFunc_str,
            sizeof(ActivationFunc_str) / sizeof(ActivationFunc_str[0])
        );

        ImGui::SameLine(COLUMN_1_START);
        ImGui::SetNextItemWidth(COLUMN_WIDTH);
        ImGui::Combo(
            "##outputact",
            reinterpret_cast<int*>(&val_output_activation),
            ActivationFunc_str,
            sizeof(ActivationFunc_str) / sizeof(ActivationFunc_str[0])
        );

        ImGui::NewLine();
        ImGui::NewLine();

        //

        ImGui::SameLine(COLUMN_0_START);
        ImGui::SetNextItemWidth(COLUMN_WIDTH);
        ImGui::Text("Batch Size");

        ImGui::SameLine(COLUMN_1_START);
        ImGui::SetNextItemWidth(COLUMN_WIDTH);
        ImGui::Text("Seed");

        ImGui::NewLine();

        const uint32_t min_batch_size = 1u;
        const uint32_t max_batch_size = 2000u;

        ImGui::SameLine(COLUMN_0_START);
        ImGui::SetNextItemWidth(COLUMN_WIDTH);
        ImGui::DragScalar(
            "##batchsize",
            ImGuiDataType_U32,
            &val_batch_size,
            1.f,
            &min_batch_size,
            &max_batch_size
        );

        ImGui::SameLine(COLUMN_1_START);
        ImGui::SetNextItemWidth(COLUMN_WIDTH);
        ImGui::InputScalar("##seed", ImGuiDataType_U32, &val_seed);

        ImGui::NewLine();
        ImGui::NewLine();

        //

        ImGui::SameLine(COLUMN_0_START);
        ImGui::SetNextItemWidth(COLUMN_WIDTH);
        ImGui::Checkbox(
            "Randomly Transform Training Images",
            &val_random_transform
        );

        ImGui::Dummy({ 1.f, scaled(.16f) });
        ImGui::NewLine();

        //

        ImGui::SameLine(COLUMN_0_START);
        ImGui::Button(
            "Train",
            {
                scaled(1.f - 2.f * WINDOW_PAD),
                scaled(.1f)
            }
        );
    }

    void App::layout_training()
    {}

    void App::layout_drawboard()
    {}

    void App::load_digit_samples(
        std::string_view images_path,
        std::string_view labels_path,
        std::vector<DigitSample>& out_samples
    )
    {
        auto stream_images = stream::open_binary_file(images_path);
        auto stream_labels = stream::open_binary_file(labels_path);

        int32_t magic_images = stream::read_bigend<int32_t>(stream_images);
        if (magic_images != 2051u)
        {
            throw std::runtime_error(
                "invalid magic number, make sure your files aren't corrupted"
            );
        }

        int32_t magic_labels = stream::read_bigend<int32_t>(stream_labels);
        if (magic_labels != 2049u)
        {
            throw std::runtime_error(
                "invalid magic number, make sure your files aren't corrupted"
            );
        }

        int32_t n_items = stream::read_bigend<int32_t>(stream_images);
        int32_t n_items_labels = stream::read_bigend<int32_t>(stream_labels);
        if (n_items != n_items_labels)
        {
            throw std::runtime_error(
                "item counts don't match in images and labels"
            );
        }

        int32_t image_width = stream::read_bigend<int32_t>(stream_images);
        int32_t image_height = stream::read_bigend<int32_t>(stream_images);
        if (image_width != DIGIT_WIDTH || image_height != DIGIT_HEIGHT)
        {
            throw std::runtime_error(std::format(
                "invalid image dimensions {}x{}, expected {}x{}",
                image_width, image_height, DIGIT_WIDTH, DIGIT_HEIGHT
            ));
        }

        size_t prev_size = out_samples.size();
        out_samples.resize(prev_size + n_items);
        for (size_t i = prev_size; i < out_samples.size(); i++)
        {
            stream::read<uint8_t>(
                stream_images,
                out_samples[i].values.data(),
                N_DIGIT_VALUES
            );
            out_samples[i].label = stream::read<uint8_t>(stream_labels);
        }
    }

}
