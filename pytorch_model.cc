#include "pytorch_model.h"
#include "pytorch_model.hpp"

#include <stdlib.h>


struct CPytorchModel {
    void* obj;
};


CPytorchModel_t* CPytorchModel_init(char* model_path) {
    CPytorchModel_t* c_obj = (CPytorchModel_t*) malloc(sizeof(CPytorchModel_t));
    PytorchModel* cpp_obj = new PytorchModel(model_path);

    c_obj->obj = cpp_obj;

    return c_obj;
}


void CPytorchModel_del(CPytorchModel_t* self) {
    if (self == NULL) { return; }

    free((PytorchModel*) (self->obj));
    free(self);
}


uint8_t* CPytorchModel_forward(CPytorchModel_t* self, uint8_t* data_in, size_t batch_size, size_t num_channels, size_t width, size_t height) {
    if (self == NULL) { return NULL; }

    PytorchModel* cpp_obj = (PytorchModel*) (self->obj);
    return cpp_obj->forward(data_in, batch_size, num_channels, width, height);
}
