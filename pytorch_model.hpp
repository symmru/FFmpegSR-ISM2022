#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <cstring>
#include <cstdint>

static constexpr unsigned int NUM_CHANNELS = 1; // RGB = 3 channels


class PytorchModel {
private:
    std::string model_path;
    torch::jit::script::Module module;

public:
    PytorchModel(const char* model_path_in):
        model_path(model_path_in)
    {
        // Loads a torchscript model found at the given file path and puts the model on the GPU.
        this->module = torch::jit::load(model_path_in);
    }

    uint8_t* forward(uint8_t* image_data, size_t batch_size, size_t num_channels, unsigned int width, unsigned int height);
};
