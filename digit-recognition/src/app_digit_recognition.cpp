#include "app_digit_recognition.hpp"

namespace digitrec
{

    void AppDigitRecognition::run()
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

    void AppDigitRecognition::init()
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
        window = glfwCreateWindow(
            WINDOW_SIZE,
            WINDOW_SIZE,
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
        font = io->Fonts->AddFontFromFileTTF(FONT_PATH, 18.f);
        if (!font)
        {
            throw std::runtime_error("failed to load fonts");
        }

        // load digit dataset
        load_digit_samples(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, train_samples);
        load_digit_samples(TEST_IMAGES_PATH, TEST_LABELS_PATH, test_samples);
    }

    void AppDigitRecognition::loop()
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

        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Hello, world!");

            ImGui::Text("This is some useful text.");

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);

            if (ImGui::Button("Button"))
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text(
                "Application average %.3f ms/frame (%.1f FPS)",
                1000.f / io->Framerate,
                io->Framerate
            );

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

    void AppDigitRecognition::cleanup()
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void AppDigitRecognition::load_digit_samples(
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
