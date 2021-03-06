#pragma once

namespace cnn {

//#define USE_DOUBLE
#ifdef USE_DOUBLE
typedef double real;
#else
typedef float real;
#endif

/// for memory alignment
#define ALIGN 6
#define CNN_ALIGN 64
// 2^ALIGN = 2^6 = 64

#define ENCODER_LAYER 0
#define INTENTION_LAYER 1
#define DECODER_LAYER 2  
#define ALIGN_LAYER 3
#define EMBEDDING_LAYER 4

#define INPUT_LAYER ENCODER_LAYER
#define HIDDEN_LAYER INTENTION_LAYER
#define OUTPUT_LAYER DECODER_LAYER


/// this is for ngram models
#define MIN_OCC_COUNT 20

/// assume column major
#define IDX2C(i, j, ld) (((j) * (ld)) + (i)) // 0 based indexing

/// for GPU
#define MAX_THREADS_PER_BLOCK 512

///for gradient checking
#define GRADIENT_CHECK_DIGIT_SIGNIFICANT_LEVEL 5
#define GRADIENT_CHECK_PARAM_DELTA 1e-5

/// for math
#define LZERO -57.00

/// for gpu allocated memory on host, i.e., pinned memory
#define GPU_ALLOC_HOST_MEM_SIZE 12768

/// preallocate a GPU memory of consts 1/k
/// the following is the maximum numbers [1/2,1/3,...1/(MEM_PRE_ALLOCATED_CONSTS_NUMBERS+1)]
#define MEM_PRE_ALLOCATED_CONSTS_NUMBERS 100


/// for device id
#define CPUDEVICE -1

/// this is defined if having lookup table parameters stored in CPU, so that large model can still be used
/// other parameters will be stored on CPU or GPU depending on the build
//#define USE_CPU_FOR_LOOKUP_PARAM

/// this is defined if using IDF value as part of cost to rank candidates
//#define RANKING_COMBINE_IDF

/// if GPU memory is small, function and gradient cannot have large allocated space
/// the solution is use both USE_CPU_FOR_LOOKUP_PARAM and also turn on the following SMALL_GPU
/// #define SMALL_GPU

/// whether to do binary serialization 
#define BINARY_BOOST

/// for beam search decoder to control the numaximum number of hypothesis
/// for speed-up
#define MAX_NUMBER_OF_HYPOTHESIS 200

/// for ranker 
#define MAX_NUMBER_OF_CANDIDATES 9

/// for random number generation
/// use curand to generate random numbers
/// #ifdef USE_CURAND

/// for decoding, the number of sentences to be evaluated
#define NBR_DEV_PARALLEL_UTTS 5

/// for ranking candidate responses, whether combine tfidf.
#define RANKING_COMBINE_TFIDF

};
