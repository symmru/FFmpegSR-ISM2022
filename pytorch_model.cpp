#include "pytorch_model.hpp"

// compute *255 and /255 in cuda

// TODO need to look into their quantize util function to see what it does
/*
 * Makes a forward call to the loaded Torchscript model.
 */
uint8_t* PytorchModel::forward(uint8_t* image_data, size_t batch_size, size_t num_channels, unsigned int width, unsigned int height) {
    // Had to add this because CUDA was running out of memory - https://github.com/pytorch/pytorch/issues/17095
    torch::NoGradGuard no_grad;

    // Read byte data into the an input tensor in the format required by the model.
    auto lr_image_tensor = torch::from_blob(
        image_data,
        {
            static_cast<long>(batch_size),
            static_cast<long>(num_channels),
            height,
            width
        },
        torch::kUInt8
     ).contiguous().to(torch::kFloat32);

    // Send the input tensor to the GPU.
    torch::Tensor lr_image_tensor_cuda = lr_image_tensor.to(torch::kCUDA);


    lr_image_tensor_cuda = lr_image_tensor_cuda/255.0;
    
    
    // Make the forward call on the model.
    std::vector<torch::jit::IValue> inputs = {lr_image_tensor_cuda};

    torch::Tensor hr_image_tensor_cuda = this->module.forward(inputs).toTensor(); // TODO not sure if the toTensor() is needed

    hr_image_tensor_cuda = hr_image_tensor_cuda * 255;
    // Move the resulting output tensor to the CPU and convert it to a byte array.
    auto hr_image_tensor_cpu = hr_image_tensor_cuda.to(torch::kCPU);
    auto hr_image_tensor_cpu_int = hr_image_tensor_cpu.to(torch::kUInt8);

    // Create a byte array that's the size the high resolution/output tensor and copy the contents to a new byte array. 
    uint8_t* result = new uint8_t[hr_image_tensor_cpu_int.numel()];

    std::memcpy(
        result, 
        hr_image_tensor_cpu_int.data_ptr<uint8_t>(), 
        sizeof(uint8_t) * hr_image_tensor_cpu_int.numel()
    );

    return result;
}