#include <iostream>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>

int main(int argc, char *argv[]) {
    
    char *model_path = NULL;  // e.g. mode.so
    model_path = argv[1];

    c10::InferenceMode mode;
    torch::inductor::AOTIModelContainerRunnerCuda *runner;
    runner = new torch::inductor::AOTIModelContainerRunnerCuda(model_path, 1);
    std::vector<torch::Tensor> inputs = {
        torch::randn({5, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)),
        torch::tensor({2, 1, 0, 0, 0, 0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA)),
    };
    std::vector<torch::Tensor> outputs = runner->run(inputs);
    std::cout << "call successfull" << std::endl;
    return 0;
}