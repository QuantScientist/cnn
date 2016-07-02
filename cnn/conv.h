#ifndef CNN_CONV_H_
#define CNN_CONV_H_

#include "cnn/cnn.h"
#include "cnn/gpu-ops.h"

namespace cnn {

struct AddVectorToAllColumns : public Node {
  explicit AddVectorToAllColumns(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
};

struct KMaxPooling : public Node {
  explicit KMaxPooling(const std::initializer_list<VariableIndex>& a, unsigned k = 1) : Node(a), k(k) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  size_t aux_storage_size() const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
  unsigned k;
};

struct FoldRows : public Node {
  explicit FoldRows(const std::initializer_list<VariableIndex>& a, unsigned nrows) : Node(a), nrows(nrows) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
  unsigned nrows;
};

// y = x_1 *conv x_2
// x_1 \in R^{d x s} (input)
// x_2 \in R^{d x m} (filter)
struct Conv1DNarrow : public Node {
  explicit Conv1DNarrow(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
};

// y = x_1 *conv x_2
// x_1 \in R^{d x s} (input)
// x_2 \in R^{d x m} (filter)
struct Conv1DWide : public Node {
  explicit Conv1DWide(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
};

struct Conv2D: public Node {
    cudnnTensorDescriptor_t srcTensorDesc;
    cudnnTensorDescriptor_t dstTensorDesc;
    cudnnTensorFormat_t tensorFormat;
    cudnnDataType_t dataType;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnTensorDescriptor_t biasTensorDesc;
    int convAlgorithm;
    int *n, *c, *h, *w;
    unsigned stride_x, stride_y;

    void createHandles()
    {
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&filterDesc));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    }
    void destroyHandles()
    {
        CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
        CHECK_CUDNN(cudnnDestroyFilterDescriptor(filterDesc));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
    }

    /// conv2d(obs, filter, bias)
    explicit Conv2D(const std::initializer_list<VariableIndex>& a, unsigned stride_x, unsigned stride_y) : 
        Node(a), stride_x(stride_x), stride_y(stride_y)
    {
        createHandles();
        n = new int[1]; c = new int[1]; h = new int[1]; w = new int[1];
        *n = 0;
        *c = 0; 
        *h = 0; 
        *w = 0;
        convAlgorithm = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
        switch (sizeof(cnn::real))
        {
            case 2: dataType = CUDNN_DATA_HALF; break;
            case 4: dataType = CUDNN_DATA_FLOAT; break;
            case 8: dataType = CUDNN_DATA_DOUBLE; break;
            default: throw("Unsupported data type");
        }
        tensorFormat = CUDNN_TENSOR_NCHW;
    }
    std::string as_string(const std::vector<std::string>& arg_names) const override;
    Dim dim_forward(const std::vector<Dim>& xs) const override;
    void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
    void backward_impl(const std::vector<const Tensor*>& xs,
        const Tensor& fx,
        const Tensor& dEdf,
        unsigned i,
        Tensor& dEdxi) const override;
    ~Conv2D() {
        destroyHandles();
        delete n; delete c; delete h; delete w; 
    }
};

struct Pooling : public Node {
    cudnnTensorDescriptor_t srcTensorDesc;
    cudnnTensorDescriptor_t dstTensorDesc;
    cudnnPoolingDescriptor_t poolingDesc;
    cudnnTensorFormat_t tensorFormat;
    cudnnDataType_t dataType;
    int *n, *c, *h, *w;
    int window_x, window_y, stride_x, stride_y;

    void createHandles()
    {
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
        CHECK_CUDNN(cudnnCreatePoolingDescriptor(&poolingDesc));
    }
    void destroyHandles()
    {
        CHECK_CUDNN(cudnnDestroyPoolingDescriptor(poolingDesc));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
    }

    /// Pooling(obs)
    explicit Pooling(const std::initializer_list<VariableIndex>& a, int window_x, int window_y, int stride_x, int stride_y) : Node(a), window_x(window_x), window_y(window_y), stride_x(stride_x), stride_y(stride_y){
        createHandles();
        n = new int[1]; c = new int[1]; h = new int[1]; w = new int[1];
        *n = 0;
        *c = 0;
        *h = 0;
        *w = 0;
        switch (sizeof(cnn::real))
        {
        case 2: dataType = CUDNN_DATA_HALF; break;
        case 4: dataType = CUDNN_DATA_FLOAT; break;
        case 8: dataType = CUDNN_DATA_DOUBLE; break;
        default: throw("Unsupported data type");
        }
        tensorFormat = CUDNN_TENSOR_NCHW;
    }
    std::string as_string(const std::vector<std::string>& arg_names) const override;
    Dim dim_forward(const std::vector<Dim>& xs) const override;
    void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
    void backward_impl(const std::vector<const Tensor*>& xs,
        const Tensor& fx,
        const Tensor& dEdf,
        unsigned i,
        Tensor& dEdxi) const override;
    ~Pooling() {
        destroyHandles();
        delete n; delete c; delete h; delete w;
    }
};

} // namespace cnn

#endif
