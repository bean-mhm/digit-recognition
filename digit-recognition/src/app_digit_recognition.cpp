#include "app_digit_recognition.hpp"

namespace digitrec
{

    AppDigitRecognition::AppDigitRecognition()
    {}

    void AppDigitRecognition::run()
    {
        load_digit_samples(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, train_samples);
        load_digit_samples(TEST_IMAGES_PATH, TEST_LABELS_PATH, test_samples);
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
