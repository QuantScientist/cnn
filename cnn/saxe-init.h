#ifndef CNN_SAXE_INIT_H_
#define CNN_SAXE_INIT_H_

#include <cnn/tensor.h>

namespace cnn {

struct Tensor;

void OrthonormalRandom(unsigned dim, float g, Tensor& x);

}

#endif
