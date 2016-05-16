#include "cnn/init.h"
#include "cnn/aligned-mem-pool.h"
#include "cnn/cnn.h"
#include "cnn/model.h"
#include <iostream>
#include <random>
#include <cmath>

#if HAVE_CUDA
#include "cnn/cuda.h"
#include <device_launch_parameters.h>
#endif

using namespace std;

namespace cnn {
    
    AlignedMemoryPool<ALIGN>* fxs = nullptr;
    AlignedMemoryPool<ALIGN>* dEdfs = nullptr;
    AlignedMemoryPool<ALIGN>* mem_nodes= nullptr;   /// for nodes allocation/delocation. operation of new/delete of each node has been overwritten to use this memory pool for speed-up
    AlignedMemoryPool<ALIGN>* glb_temp_working_mem = nullptr;
    AlignedMemoryPool<ALIGN>* glb_temp_lookup_gradient_value_mem = nullptr; /// this saves gradient on those sparse lookup table parameters that have non-zero gradiens. these values and gradients are temporary
    mt19937* rndeng = nullptr;

    char* getCmdOption(char ** begin, char ** end, const std::string & option)
    {
        char ** itr = std::find(begin, end, option);
        if (itr != end && ++itr != end)
        {
            return *itr;
        }
        return 0;
    }

	static void RemoveArgs(int& argc, char**& argv, int& argi, int n) {
	  for (int i = argi + n; i < argc; ++i)
	    argv[i - n] = argv[i];
	  argc -= n;
	  assert(argc >= 0);
	}
	
    bool cmdOptionExists(char** begin, char** end, const std::string& option)
    {
        return std::find(begin, end, option) != end;
    }

    void Initialize(int& argc, char**& argv, int init_device_id, unsigned random_seed, bool demo) 
    {

        cerr << "Initializing...\n";

        if (random_seed == 0)
        {
            if (cmdOptionExists(argv, argv + argc, "--seed"))
            {
                string seed = getCmdOption(argv, argv + argc, "--seed");
                stringstream(seed) >> random_seed;
            }
            else
            {
                random_device rd;
                random_seed = rd();
            }
        }

#if HAVE_CUDA
        Initialize_GPU(argc, argv, random_seed, init_device_id);
#else
        kSCALAR_MINUSONE = (cnn::real*)cnn_mm_malloc(sizeof(cnn::real), CNN_ALIGN);
        *kSCALAR_MINUSONE = -1;
        kSCALAR_ONE = (cnn::real*)cnn_mm_malloc(sizeof(cnn::real), CNN_ALIGN);
        *kSCALAR_ONE = 1;
        kSCALAR_ZERO = (cnn::real*)cnn_mm_malloc(sizeof(cnn::real), CNN_ALIGN);
        *kSCALAR_ZERO = 0;
#endif

        rndeng = new mt19937(random_seed);

        cerr << "Allocating memory...\n";
		unsigned long num_mb = 512UL;
        mem_nodes = new AlignedMemoryPool<ALIGN>(512UL * (1UL << 20), true);
        glb_temp_working_mem = new AlignedMemoryPool<ALIGN>(1UL << 16); /// save gradient norms
        glb_temp_lookup_gradient_value_mem = new AlignedMemoryPool<ALIGN>(1UL << 25);

        if (demo)
        {
            fxs = new AlignedMemoryPool<ALIGN>(512UL * (1UL << 20));
            dEdfs = new AlignedMemoryPool<ALIGN>(512UL * (1UL << 20));
        }
        else
        {
#ifdef SMALL_GPU
            fxs = new AlignedMemoryPool<ALIGN>(512UL * (1UL << 20));
            dEdfs = new AlignedMemoryPool<ALIGN>(512UL * (1UL << 20));
#else
            fxs = new AlignedMemoryPool<ALIGN>(768UL * (1UL << 22)); /// 3G memory
            dEdfs = new AlignedMemoryPool<ALIGN>(768UL * (1UL << 22)); /// 3G memory
#endif
        }
        cerr << "Done.\n";
    }

  void Free() 
  {
        cerr << "Freeing memory ...\n";
        cnn_mm_free(kSCALAR_MINUSONE);
        cnn_mm_free(kSCALAR_ONE);
        cnn_mm_free(kSCALAR_ZERO);

        delete (rndeng); 
        delete (fxs);
        delete (dEdfs);
        delete (mem_nodes);
        delete (glb_temp_working_mem);

        for (auto p : kSCALAR_ONE_OVER_INT)
            cnn_mm_free(p);

#ifdef HAVE_CUDA
        Free_GPU();
#endif
        cerr << "Done.\n";
  }

} // namespace cnn
