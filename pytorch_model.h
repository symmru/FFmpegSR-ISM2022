#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CPytorchModel;
typedef struct CPytorchModel CPytorchModel_t;

CPytorchModel_t* CPytorchModel_init(char* model_path);
void CPytorchModel_del(CPytorchModel_t* self);

uint8_t* CPytorchModel_forward(CPytorchModel_t* self, uint8_t* data_in, size_t batch_size, size_t num_channels, size_t width, size_t height);



#ifdef __cplusplus
}
#endif
