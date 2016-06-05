#ifndef CNN_EIGEN_INIT_H
#define CNN_EIGEN_INIT_H

namespace cnn {

    void Initialize(int& argc, char**& argv, 
        int init_device_id = -1, /// this can be set to a non-negative number so that a particular GPU can be selected
        unsigned random_seed = 0, bool demo = false);

    void Free();
} // namespace cnn

#endif
