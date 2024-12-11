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
        if (train_samples.size() < 100u || test_samples.size() < 100u)
        {
            throw std::runtime_error(std::format(
                "the number of training or test samples is extremely low "
                "(training samples: {}, test samples: {})",
                train_samples.size(),
                test_samples.size()
            ));
        }

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

    static void setup_imgui_style()
    {
        // Digit Rec style from ImThemes
        ImGuiStyle& style = ImGui::GetStyle();

        style.Alpha = 1.0f;
        style.DisabledAlpha = 1.0f;
        style.WindowPadding = ImVec2(12.0f, 12.0f);
        style.WindowRounding = 4.0f;
        style.WindowBorderSize = 0.0f;
        style.WindowMinSize = ImVec2(20.0f, 20.0f);
        style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
        style.WindowMenuButtonPosition = ImGuiDir_None;
        style.ChildRounding = 4.0f;
        style.ChildBorderSize = 1.0f;
        style.PopupRounding = 4.0f;
        style.PopupBorderSize = 1.0f;
        style.FramePadding = ImVec2(11.0f, 6.0f);
        style.FrameRounding = 4.0f;
        style.FrameBorderSize = 1.0f;
        style.ItemSpacing = ImVec2(12.0f, 6.0f);
        style.ItemInnerSpacing = ImVec2(6.0f, 3.0f);
        style.CellPadding = ImVec2(12.0f, 6.0f);
        style.IndentSpacing = 20.0f;
        style.ColumnsMinSpacing = 6.0f;
        style.ScrollbarSize = 12.0f;
        style.ScrollbarRounding = 20.0f;
        style.GrabMinSize = 28.0f;
        style.GrabRounding = 20.0f;
        style.TabRounding = 4.0f;
        style.TabBorderSize = 1.0f;
        style.TabMinWidthForCloseButton = 0.0f;
        style.ColorButtonPosition = ImGuiDir_Right;
        style.ButtonTextAlign = ImVec2(0.5f, 0.5f);
        style.SelectableTextAlign = ImVec2(0.0f, 0.0f);

        style.Colors[ImGuiCol_Text] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
        style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.2745098173618317f, 0.3176470696926117f, 0.4509803950786591f, 1.0f);
        style.Colors[ImGuiCol_WindowBg] = ImVec4(0.04313725605607033f, 0.09803921729326248f, 0.1411764770746231f, 1.0f);
        style.Colors[ImGuiCol_ChildBg] = ImVec4(0.05387832224369049f, 0.1191623359918594f, 0.1673820018768311f, 1.0f);
        style.Colors[ImGuiCol_PopupBg] = ImVec4(0.03921568766236305f, 0.07112683355808258f, 0.08627451211214066f, 1.0f);
        style.Colors[ImGuiCol_Border] = ImVec4(1.0f, 1.0f, 1.0f, 0.0313725508749485f);
        style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.0784313753247261f, 0.08627451211214066f, 0.1019607856869698f, 0.0f);
        style.Colors[ImGuiCol_FrameBg] = ImVec4(0.1013096645474434f, 0.1450867503881454f, 0.1888412237167358f, 1.0f);
        style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.1139825657010078f, 0.1711567640304565f, 0.2231759428977966f, 1.0f);
        style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.1357733607292175f, 0.3936567902565002f, 0.5021458864212036f, 1.0f);
        style.Colors[ImGuiCol_TitleBg] = ImVec4(0.0363609567284584f, 0.04964735731482506f, 0.06008583307266235f, 1.0f);
        style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.02564055100083351f, 0.06896454095840454f, 0.1030042767524719f, 1.0f);
        style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.0363609567284584f, 0.04964735731482506f, 0.06008583307266235f, 1.0f);
        style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.02564055100083351f, 0.06896454095840454f, 0.1030042767524719f, 1.0f);
        style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(9.999899930335232e-07f, 9.999932899518171e-07f, 9.999999974752427e-07f, 0.1759656667709351f);
        style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.9999899864196777f, 0.9999949932098389f, 1.0f, 0.1072961091995239f);
        style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.9999899864196777f, 0.9999949932098389f, 1.0f, 0.1459227204322815f);
        style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.9999899864196777f, 0.9999949932098389f, 1.0f, 0.2403433322906494f);
        style.Colors[ImGuiCol_CheckMark] = ImVec4(0.133452445268631f, 0.5546011924743652f, 0.6909871101379395f, 1.0f);
        style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.4077253341674805f, 0.7330952882766724f, 1.0f, 0.540772557258606f);
        style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.545064389705658f, 0.865276575088501f, 1.0f, 0.6309012770652771f);
        style.Colors[ImGuiCol_Button] = ImVec4(0.1224741563200951f, 0.1990198194980621f, 0.2618025541305542f, 1.0f);
        style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.140396773815155f, 0.2377501279115677f, 0.3175965547561646f, 1.0f);
        style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.133452445268631f, 0.5546011924743652f, 0.6909871101379395f, 1.0f);
        style.Colors[ImGuiCol_Header] = ImVec4(0.5921568870544434f, 0.8705882430076599f, 1.0f, 0.0470588244497776f);
        style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.5803921818733215f, 0.8666666746139526f, 1.0f, 0.08627451211214066f);
        style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.1357733607292175f, 0.3936567902565002f, 0.5021458864212036f, 1.0f);
        style.Colors[ImGuiCol_Separator] = ImVec4(0.1490196138620377f, 0.1843137294054031f, 0.250980406999588f, 1.0f);
        style.Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.1568627506494522f, 0.1843137294054031f, 0.250980406999588f, 1.0f);
        style.Colors[ImGuiCol_SeparatorActive] = ImVec4(0.1568627506494522f, 0.1843137294054031f, 0.250980406999588f, 1.0f);
        style.Colors[ImGuiCol_ResizeGrip] = ImVec4(0.1224741563200951f, 0.1990198194980621f, 0.2618025541305542f, 1.0f);
        style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.140396773815155f, 0.2377501279115677f, 0.3175965547561646f, 1.0f);
        style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.133452445268631f, 0.5546011924743652f, 0.6909871101379395f, 1.0f);
        style.Colors[ImGuiCol_Tab] = ImVec4(0.08642634004354477f, 0.1531297415494919f, 0.1974248886108398f, 1.0f);
        style.Colors[ImGuiCol_TabHovered] = ImVec4(0.112730011343956f, 0.1997324824333191f, 0.2575107216835022f, 1.0f);
        style.Colors[ImGuiCol_TabActive] = ImVec4(0.09664943069219589f, 0.3192445337772369f, 0.4248927235603333f, 1.0f);
        style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.08642634004354477f, 0.1531297415494919f, 0.1974248886108398f, 1.0f);
        style.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.09664943069219589f, 0.3192445337772369f, 0.4248927235603333f, 1.0f);
        style.Colors[ImGuiCol_PlotLines] = ImVec4(0.3923630714416504f, 0.6473053693771362f, 0.7682403326034546f, 1.0f);
        style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.1845493316650391f, 0.9475069642066956f, 1.0f, 1.0f);
        style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.3043710589408875f, 0.5694959759712219f, 0.6952790021896362f, 1.0f);
        style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.3920683860778809f, 0.833932101726532f, 0.9227467775344849f, 1.0f);
        style.Colors[ImGuiCol_TableHeaderBg] = ImVec4(0.3991416096687317f, 0.837547779083252f, 1.0f, 0.1802574992179871f);
        style.Colors[ImGuiCol_TableBorderStrong] = ImVec4(9.999899930335232e-07f, 9.999934036386549e-07f, 9.999999974752427e-07f, 0.1931330561637878f);
        style.Colors[ImGuiCol_TableBorderLight] = ImVec4(1.0f, 1.0f, 1.0f, 0.05098039284348488f);
        style.Colors[ImGuiCol_TableRowBg] = ImVec4(0.07029048353433609f, 0.1252296715974808f, 0.1545064449310303f, 1.0f);
        style.Colors[ImGuiCol_TableRowBgAlt] = ImVec4(0.1013096645474434f, 0.1584118008613586f, 0.1888412237167358f, 1.0f);
        style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.1024885177612305f, 0.3535920679569244f, 0.459227442741394f, 1.0f);
        style.Colors[ImGuiCol_DragDropTarget] = ImVec4(0.133452445268631f, 0.5546011924743652f, 0.6909871101379395f, 1.0f);
        style.Colors[ImGuiCol_NavHighlight] = ImVec4(0.133452445268631f, 0.5546011924743652f, 0.6909871101379395f, 1.0f);
        style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(0.133452445268631f, 0.5546011924743652f, 0.6909871101379395f, 1.0f);
        style.Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.5254902243614197f, 0.0f, 0.0f, 0.3294117748737335f);
        style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.5098039507865906f);
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
        setup_imgui_style();

        // setup backends
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glsl_version);

        // load fonts
        font = io->Fonts->AddFontFromFileTTF(FONT_PATH, FONT_SIZE);
        font_bold = io->Fonts->AddFontFromFileTTF(FONT_BOLD_PATH, FONT_SIZE);
        if (!font || !font_bold)
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

            ImGui::PushFont(font);

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

            ImGui::PopFont();

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
        size_t n_decimal = 4;
        if (val_learning_rate < .001f) n_decimal = 5;
        if (val_learning_rate < .0001f) n_decimal = 6;
        ImGui::Text(
            std::format("Learning Rate: %.{}f", n_decimal).c_str(),
            val_learning_rate
        );

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
        static float learning_rate_root4 = std::pow(val_learning_rate, .25f);
        ImGui::SliderFloat(
            "##learnrate",
            &learning_rate_root4,
            0.f,
            1.f,
            "##",
            ImGuiSliderFlags_AlwaysClamp
            | ImGuiSliderFlags_NoRoundToFormat
            | ImGuiSliderFlags_NoInput
        );
        val_learning_rate = std::pow(learning_rate_root4, 4.f);

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
            &max_batch_size,
            nullptr,
            ImGuiSliderFlags_AlwaysClamp
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

        ImGui::Dummy({ 1.f, scaled(.15f) });
        ImGui::NewLine();

        //

        static std::string error_text = "";

        ImGui::SameLine(COLUMN_0_START);
        if (ImGui::Button(
            "Train",
            {
                scaled(1.f - 2.f * WINDOW_PAD),
                scaled(.1f)
            }
        ))
        {
            auto result = start_training();
            if (result.has_value())
            {
                error_text = result.value();
                ImGui::OpenPopup("Error");
            }
        }

        //

        ImGui::SetNextWindowSize({ scaled(.7f), 0.f });
        ImGui::SetNextWindowPos(
            { ImGui::GetWindowSize().x * .5f, ImGui::GetWindowSize().y * .5f },
            0,
            { .5f, .5f }
        );
        if (ImGui::BeginPopupModal(
            "Error",
            nullptr,
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove
        ))
        {
            ImGui::SameLine(scaled(DIALOG_PAD));
            ImGui::SetNextItemWidth(
                ImGui::GetWindowSize().x - 2.f * scaled(DIALOG_PAD)
            );
            ImGui::TextWrapped(error_text.c_str());

            ImGui::NewLine();

            ImGui::SameLine(scaled(DIALOG_PAD));
            if (ImGui::Button(
                "Ok",
                {
                    ImGui::GetWindowSize().x - 2.f * scaled(DIALOG_PAD),
                    scaled(.05f)
                }
            ))
            {
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }
    }

    void App::layout_training()
    {
        static constexpr float PADDED_WIDTH = scaled(
            .5f - COLUMN_SPACING - WINDOW_PAD
        );
        static constexpr float CONTENT_START = scaled(WINDOW_PAD);
        static constexpr float CONTENT_WIDTH = scaled(
            1.f - 2.f * WINDOW_PAD
        );

        ImGui::SameLine(CONTENT_START);
        ImGui::SetNextItemWidth(CONTENT_WIDTH);
        if (accuracy_history.empty())
        {
            ImGui::Text("Accuracy");
        }
        else
        {
            ImGui::Text("Accuracy: %.1f%%", accuracy_history.back() * 100.f);
        }
        network_summary_tooltip();

        ImGui::NewLine();

        //

        ImGui::SameLine(CONTENT_START);
        ImGui::SetNextItemWidth(CONTENT_WIDTH);
        ImGui::PlotLines(
            "##accuracyplot",
            accuracy_history.data(),
            (int)accuracy_history.size(),
            0,
            (const char*)0,
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max(),
            ImVec2{ CONTENT_WIDTH, scaled(.48f) }
        );

        ImGui::NewLine();
        ImGui::NewLine();

        //

        ImGui::SameLine(CONTENT_START);
        if (ImGui::Button(
            "Stop",
            {
                CONTENT_WIDTH,
                scaled(.1f)
            }
        ))
        {
            stop_training();
        }
    }

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

    std::optional<std::string> App::start_training()
    {
        // parse and verify layer sizes

        std::vector<int64_t> layer_sizes_i64;
        for (auto& s : str::split(val_layer_sizes, ","))
        {
            str::trim_inplace(s);
            try
            {
                layer_sizes_i64.push_back(std::stoll(s));
            }
            catch (const std::exception&)
            {
                return
                    "Layer sizes must be a list of positive integers "
                    "separated by commas.";
            }
        }

        std::vector<size_t> layer_sizes;
        for (auto layer_size_i64 : layer_sizes_i64)
        {
            if (layer_size_i64 < 0)
            {
                return "Layer sizes can't be negative.";
            }
            else if (layer_size_i64 < 1)
            {
                return "A layer must contain at least 1 node / neuron.";
            }
            layer_sizes.push_back((size_t)layer_size_i64);
        }

        if (layer_sizes.size() < 2u)
        {
            return "There should be at least 2 layers (input and output).";
        }
        if (layer_sizes.size() > 10u)
        {
            return "Too many layers.";
        }
        if (layer_sizes[0] != N_DIGIT_VALUES)
        {
            return std::format(
                "The size of the first layer (input) must always be {}.",
                N_DIGIT_VALUES
            );
        }
        if (layer_sizes.back() != 10)
        {
            return "The size of the last layer (output) must always be 10.";
        }

        // activation functions and their derivatives
        std::vector<std::function<float(float)>> activation_fns;
        std::vector<std::function<float(float)>> activation_derivs;

        // hidden layer activation functions
        if (layer_sizes.size() > 2u)
        {
            std::function<float(float)> func;
            std::function<float(float)> deriv;
            switch (val_hidden_activation)
            {
            case digit_rec::ActivationFunc::Relu:
                func = neural::relu<float>;
                deriv = neural::relu_deriv<float>;
                break;
            case digit_rec::ActivationFunc::LeakyRelu:
                func = neural::leaky_relu<float, .01f>;
                deriv = neural::leaky_relu_deriv<float, .01f>;
                break;
            case digit_rec::ActivationFunc::Tanh:
                func = neural::tanh<float>;
                deriv = neural::tanh_deriv<float>;
                break;
            default:
                break;
            }

            for (size_t i = 0; i < layer_sizes.size() - 2u; i++)
            {
                activation_fns.push_back(func);
                activation_derivs.push_back(deriv);
            }
        }

        // output layer activation function
        switch (val_output_activation)
        {
        case digit_rec::ActivationFunc::Relu:
            activation_fns.push_back(neural::relu<float>);
            activation_derivs.push_back(neural::relu_deriv<float>);
            break;
        case digit_rec::ActivationFunc::LeakyRelu:
            activation_fns.push_back(neural::leaky_relu<float, .01f>);
            activation_derivs.push_back(neural::leaky_relu_deriv<float, .01f>);
            break;
        case digit_rec::ActivationFunc::Tanh:
            activation_fns.push_back(neural::tanh<float>);
            activation_derivs.push_back(neural::tanh_deriv<float>);
            break;
        default:
            break;
        }

        // recreate neural network
        net = std::make_unique<neural::Network<float, true>>(
            layer_sizes,
            activation_fns,
            activation_derivs
        );

        // initialize network with random weights and biases
        std::mt19937 rng_initialization(val_seed);
        net->randomize_xavier_normal(rng_initialization, -.01f, .01f);

        // reset accuracy history and the number of training steps
        accuracy_history.clear();
        n_training_steps = 0;

        // start the training thread
        last_accuracy_calc_time = std::chrono::high_resolution_clock::now();
        training_thread = std::make_unique<std::jthread>(
            [this](std::stop_token stoken)
            {
                recalculate_accuracy_and_add_to_history();

                // number of floats in a single training example which contains
                // input data + expected output data.
                static constexpr size_t TRAINING_DATA_SIZE =
                    N_DIGIT_VALUES + 10u;

                std::vector<float> training_data(
                    (size_t)val_batch_size * TRAINING_DATA_SIZE
                );

                std::vector<std::span<float>> spans(val_batch_size);
                for (size_t i = 0; i < val_batch_size; i++)
                {
                    spans[i] = std::span<float>(
                        training_data.data() + (i * TRAINING_DATA_SIZE),
                        TRAINING_DATA_SIZE
                    );
                }

                std::mt19937 rng(val_seed);
                std::uniform_int_distribution<size_t> sizet_dist(
                    0,
                    train_samples.size() - 1u
                );

                while (!stoken.stop_requested())
                {
                    // training step
                    for (size_t i = 0; i < val_batch_size; i++)
                    {
                        // pointer to input data for this training example
                        float* input_data =
                            training_data.data() + (i * TRAINING_DATA_SIZE);

                        // pointer to expected output data for this example
                        float* output_data =
                            training_data.data()
                            + (i * TRAINING_DATA_SIZE)
                            + N_DIGIT_VALUES;

                        // randomly pick a digit sample from the dataset
                        const auto& samp = train_samples[sizet_dist(rng)];

                        // update input data
                        for (size_t i = 0; i < N_DIGIT_VALUES; i++)
                        {
                            input_data[i] = (float)samp.values[i] / 255.f;
                        }

                        // update expected output data
                        for (uint32_t i = 0; i < 10; i++)
                        {
                            output_data[i] = (i == samp.label) ? 1.f : 0.f;
                        }
                    }
                    net->train(spans, val_learning_rate);
                    n_training_steps++;

                    // recalculate the accuracy if needed
                    auto elapsed_ms =
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now()
                            - last_accuracy_calc_time
                        ).count();
                    if (elapsed_ms > 1000)
                    {
                        recalculate_accuracy_and_add_to_history();
                        last_accuracy_calc_time =
                            std::chrono::high_resolution_clock::now();
                    }
                }
            }
        );

        // switch UI to training mode
        ui_mode = UiMode::Training;
    }

    void App::stop_training()
    {
        if (training_thread)
        {
            training_thread->request_stop();
            training_thread->join();
        }

        // switch UI to drawboard
        ui_mode = UiMode::Drawboard;
    }

    void App::recalculate_accuracy_and_add_to_history()
    {
        auto net_input = net->input_values();
        auto net_output = net->output_values();

        size_t total_tests = test_samples.size();
        size_t n_correct_predict = 0;

        for (const auto& samp : test_samples)
        {
            for (size_t i = 0; i < N_DIGIT_VALUES; i++)
            {
                net_input[i] = (float)samp.values[i] / 255.f;
            }

            net->forward_pass();

            uint32_t predicted_label = 0;
            float max_output = net_output[0];
            for (uint32_t i = 1; i < 10; i++)
            {
                if (net_output[i] > max_output)
                {
                    max_output = net_output[i];
                    predicted_label = i;
                }
            }

            if (predicted_label == samp.label)
            {
                n_correct_predict++;
            }
        }

        accuracy_history.push_back(
            (float)n_correct_predict / (float)total_tests
        );
    }

    void App::bold_text(const char* s, va_list args)
    {
        ImGui::PushFont(font_bold);
        ImGui::Text(s, args);
        ImGui::PopFont();
    }

    void App::network_summary_tooltip()
    {
        if (!net || !ImGui::IsItemHovered())
            return;

        if (!ImGui::BeginTooltip())
            return;

        std::string s_layer_sizes;
        for (size_t i = 0; i < net->layer_sizes().size(); i++)
        {
            if (i != 0)
                s_layer_sizes += ", ";
            s_layer_sizes += std::to_string(net->layer_sizes()[i]);
        }
        bold_text("Layer Sizes:");
        ImGui::SameLine();
        ImGui::Text(s_layer_sizes.c_str());

        bold_text("Learning Rate:");
        ImGui::SameLine();
        ImGui::Text("%.6f", val_learning_rate);

        bold_text("Hidden Layer Activation:");
        ImGui::SameLine();
        ImGui::Text(ActivationFunc_str[(size_t)val_hidden_activation]);

        bold_text("Output Layer Activation:");
        ImGui::SameLine();
        ImGui::Text(ActivationFunc_str[(size_t)val_output_activation]);

        bold_text("Batch Size:");
        ImGui::SameLine();
        ImGui::Text("%u", val_batch_size);

        bold_text("Seed:");
        ImGui::SameLine();
        ImGui::Text("%u", val_seed);

        bold_text("Randomly Transform Images:");
        ImGui::SameLine();
        ImGui::Text("%s", val_random_transform ? "Yes" : "No");

        ImGui::NewLine();
        bold_text("Training Steps:");
        ImGui::SameLine();
        ImGui::Text("%llu", n_training_steps.load());

        ImGui::EndTooltip();

        /*return std::format(
            "Layer Sizes: {}\n"
            "Learning Rate: {}\n"
            "Hidden Layer Activation: {}\n"
            "Output Layer Activation: {}\n"
            "Batch Size: {}\n"
            "Seed: {}\n"
            "Randomly Transform Images: {}\n"
            "Training Steps: {}",

            s_layer_sizes,
            val_learning_rate,
            ActivationFunc_str[(size_t)val_hidden_activation],
            ActivationFunc_str[(size_t)val_output_activation],
            val_batch_size,
            val_seed,
            val_random_transform,
            n_training_steps.load()
        );*/
    }

}
