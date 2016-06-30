#ifndef CNN_GPU_OPS_H
#define CNN_GPU_OPS_H

#include <cnn/macros.h>

namespace cnn {
namespace gpu {

    void vpairwise_rank_loss(int n, cnn::real margin, const cnn::real* xgood, const cnn::real* xbad, cnn::real* y);

    void set_to_value_of(int n, cnn::real* x0, cnn::real val);
    void set_to_value_of(int n, cnn::real* x0, cnn::real *val);

/// for convlution networks
    void conv1dwide(const int n, const int m, const cnn::real* xs, const int k, const cnn::real *fx, cnn::real *fy);
    void conv1dwide_backward(const int i, const int n, const int m, const cnn::real* xs, const int k, const cnn::real *fx, const cnn::real* dEdf, cnn::real *dEdx);
    void conv2dnarrow(const cnn::real* kscalar_one, const cnn::real* kscalar_zero,
        const int xrow, const int xcol, const cnn::real* xs,
        const int i_wkspace_sz, cnn::real* wkspace,
        const int frow, const int fcol, const cnn::real *fx,
        const int yrow, const int ycol, cnn::real *fy);

    /// add bias
    void addVectorToAllColumns(const int n, const cnn::real * xs, const int m, const cnn::real* fx, cnn::real *fy);
    void addVectorToAllColumns_backward(const int i, const int r, const int c, const cnn::real* dEdf, cnn::real *dEdxi);

    void foldRows(const int n, const int m, const cnn::real *xs, const int stride, const int orows, cnn::real *fy);
    void foldRows_backward(const int orows, const cnn::real* dEdf, const int n, const int m, cnn::real *fy);

    void kMaxPooling(const int n, const int m, const cnn::real *xs, const int k, cnn::real *fy, int* aux_mem);
    void kMaxPooling_backward(const int n, const int m, const cnn::real *xs, const int k, const cnn::real * dEdf, cnn::real *dEdxi, const int* aux_mem);

    void vpairwise_rank_loss(int n, cnn::real margin, const cnn::real* xgood, const cnn::real* xbad, cnn::real* y);
    void vpairwise_rank_loss_backward(int n, bool d_wrt_correct, const cnn::real* fx, const cnn::real* dEdf, cnn::real* dEdx);
    void vcwise_product(int n, const cnn::real* x0, const cnn::real* x1, cnn::real* y);
    void vcwise_product_backward(int n, const cnn::real* dEdy, const cnn::real* x_other, cnn::real* dEdx);
    void vcwise_quotient(int n, const cnn::real* x0, const cnn::real* x1, cnn::real* y);
    void vcwise_quotient_backward(int n, const cnn::real* dEdy, const cnn::real* x_other, cnn::real* dEdx);
    void vconstant_minusx(int n, cnn::real c, const cnn::real* x, cnn::real* y);
    /// c should be zero if used as back-propagation of y = x - c, since dx += dy should be the gradient to x
    void vconstant_minusx_backward(int n, cnn::real c, const cnn::real* x, cnn::real* y);
    void vconstant_multiplyx(int n, cnn::real c, const cnn::real* x, cnn::real* y);
    void vconstant_multiplyx_backward(int n, cnn::real c, const cnn::real* x, cnn::real* y);
    void vnegate(int n, const cnn::real* x, cnn::real* y);
    void vnegate_backward(int n, const cnn::real* dEdf, cnn::real* dEdx);
    void vrelu(int n, const cnn::real* x, cnn::real* y);
    void vrelu_backward(int n, const cnn::real* fx, const cnn::real* dEdf, cnn::real* dEdx);
    void vexponential_linear_units(int n, const cnn::real* x, const cnn::real scale, cnn::real* y);
    void vexponential_linear_units_backward(int n, const cnn::real* fx, const cnn::real* dEdf, const cnn::real scale, cnn::real* dEdx);
    void vexp(int n, const cnn::real* x, cnn::real* y);
    void vtanh(int n, const cnn::real* x, cnn::real* y);
    void vtanh_backward(int n, const cnn::real* fx, const cnn::real* dEdf, cnn::real* dEdx);
    void vlog(int n, const cnn::real* x, cnn::real* y);
    void vlog_backward(int n, const cnn::real* fx, const cnn::real* dEdf, cnn::real* dEdx);
    void vlogistic(int n, const cnn::real* x, cnn::real* y);
    void vlogistic_backward(int n, const cnn::real* fx, const cnn::real* dEdf, cnn::real* dEdx);
    void l2_norm_reducer(int n, const cnn::real* x0, cnn::real* y, bool square, bool accumulate);
    void sqrt_of_l2_norm_reducer(int n, cnn::real* x0, cnn::real& res);
    void sqeucdist(int n, const cnn::real* x0, const cnn::real *x1, cnn::real* y);
    void sqeucdist_backward(int n, const cnn::real* dEdy, const cnn::real* x0, const cnn::real* x1, cnn::real* dEdx, int i);
    void pnlsoftmax(int n, int elem_idx, const cnn::real* x0, cnn::real* y, cnn::real* logz);
    void pnlsoftmax_backward(int n, int elem_idx, const cnn::real* x0, const cnn::real* dEdf, const cnn::real* logz, cnn::real* dEdx);
    void logsoftmax(int row, int col, const cnn::real* x0, cnn::real* y);
    void logsoftmax_backward(int row, int col, const cnn::real *fx, const cnn::real *dEdf, cnn::real *dEdx);
    void softmax(int row, int col, const cnn::real* x0, cnn::real* y);
    void softmax_backward(int row, int col, const cnn::real *fx, const cnn::real *dEdf, cnn::real *dEdx);
    void sgd_update(int n, const cnn::real* g, cnn::real* x, cnn::real scale, cnn::real lambda);
    void sgd_update(int n, const cnn::real* g, cnn::real* x, cnn::real* scale, cnn::real* lambda);
    void sgd_momentum_update(int n, const cnn::real* g, cnn::real* x, cnn::real * v, cnn::real scale, cnn::real lambda, cnn::real momentum);
    void rmsprop_update(int n, const cnn::real* g, cnn::real* x, cnn::real *r, cnn::real scale, cnn::real lambda, cnn::real rho, cnn::real epsilon, cnn::real grd_squared_norm);
    void rmsprop_momentum_update(int n, const cnn::real* g, cnn::real* x, cnn::real* v, cnn::real *r, cnn::real scale, cnn::real lambda, cnn::real momentum, cnn::real rho, cnn::real epsilon, cnn::real grd_squared_norm);

    void rmsprop_smoothing_den(int n, cnn::real rho, const cnn::real *grd_squared_norm, cnn::real *r);
    void clip_gradients(int n, const cnn::real *dense_param_grad_norm,
        int m, const cnn::real *sparse_param_grad_norm,
        cnn::real clip_threshold, int samples,
        cnn::real* gscale);
    void rmsprop_momentum_update(int n, const cnn::real* r, cnn::real* x, const cnn::real* g, cnn::real* v, cnn::real* gscale, cnn::real lambda, cnn::real scale, cnn::real momentum, cnn::real epsilon);

    void vector_sum(int rows, int cols, const cnn::real * a, cnn::real* c, const bool isColWise);
    void vector_add_const(int rows, int cols, const cnn::real * a, int brow, int bcol, const cnn::real* b, cnn::real * c, bool isColWise);

    /// Y = a X + b
    /// a anb b are scalar
    void vsax_plus_sb(int n, cnn::real a, cnn::real b, cnn::real* x, cnn::real* y);

    /// clip each element of x0 if its absolute value is larger than the threshold
    void simple_clipping(int n, const cnn::real* x0, cnn::real* y, cnn::real threshold);

    void addBias(const cudnnTensorDescriptor_t dstTensorDesc,
        const cnn::real* bias_d,
        int c, cnn::real *data,
        cudnnTensorDescriptor_t biasTensorDesc,
        cudnnTensorFormat_t tensorFormat,
        cudnnDataType_t dataType);
    void convBackwardBias(const cudnnTensorDescriptor_t dstTensorDesc,
        const cnn::real* d_dst,
        cudnnTensorDescriptor_t biasTensorDesc,
        cnn::real *d_bias);

    void convoluteForwardOutputSize(const int conv_inputs,
        const int conv_outputs, const int conv_kernel_dim_x,
        const int conv_kernel_dim_y,
        int* n, int* c, int* h, int* w,
        cudnnTensorDescriptor_t srcTensorDesc,
        cudnnTensorDescriptor_t dstTensorDesc,
        cudnnTensorFormat_t tensorFormat,
        cudnnDataType_t dataType,
        cudnnFilterDescriptor_t filterDesc,
        cudnnConvolutionDescriptor_t convDesc);

    void convoluteForward(
        cnn::real *cnn_filter_data_d,
        cnn::real *cnn_bias_d,
        int n, int c, int h, int w,
        cnn::real* srcData, cnn::real** dstData,
        cudnnTensorDescriptor_t srcTensorDesc,
        cudnnTensorDescriptor_t dstTensorDesc,
        cudnnTensorFormat_t tensorFormat,
        cudnnDataType_t dataType,
        cudnnFilterDescriptor_t filterDesc,
        cudnnConvolutionDescriptor_t convDesc,
        cudnnTensorDescriptor_t biasTensorDesc,
        int convAlgorithm);

    void convoluteBackwardToFilter(
        cnn::real* srcData, /// observation
        cnn::real* dyDst,   /// gradient to be backpropagated
        cnn::real * dFilter,  /// gradient to be propagated to
        cudnnTensorDescriptor_t srcTensorDesc,
        cudnnTensorDescriptor_t dstTensorDesc,
        cudnnTensorFormat_t tensorFormat,
        cudnnDataType_t dataType,
        cudnnConvolutionDescriptor_t convDesc,
        cudnnFilterDescriptor_t filterDesc,
        int convAlgorithm);

    void convoluteBackwardToData(
        cnn::real * filterData, /// filter 
        cnn::real* dyDst,   /// gradient to be backpropagated
        cnn::real * dxData,  /// gradient to be propagated to
        cudnnTensorDescriptor_t srcTensorDesc,
        cudnnTensorDescriptor_t dstTensorDesc,
        cudnnTensorFormat_t tensorFormat,
        cudnnDataType_t dataType,
        cudnnConvolutionDescriptor_t convDesc,
        cudnnFilterDescriptor_t filterDesc,
        int convAlgorithm);

    void poolingForwardOutputSize(cudnnPoolingDescriptor_t poolingDesc,
        cudnnTensorDescriptor_t srcTensorDesc,
        cudnnTensorDescriptor_t dstTensorDesc,
        cudnnTensorFormat_t tensorFormat,
        cudnnDataType_t dataType,
        int *n, int *c, int *h, int *w,
        int window_x, int window_y,
        int stride_x, int stride_y);

    void poolForward(
        cnn::real* srcData, cnn::real* dstData,
        int* n, int* c, int* h, int* w,
        cudnnTensorDescriptor_t srcTensorDesc,
        cudnnTensorDescriptor_t dstTensorDesc,
        cudnnPoolingDescriptor_t     poolingDesc,
        cudnnTensorFormat_t tensorFormat,
        cudnnHandle_t cudnnHandle,
        cudnnDataType_t dataType);

    void poolBackward(cnn::real* xObs, /// input to the pooling
        cnn::real * yDst, /// response from the pooling
        cnn::real* dyDst, /// gradients to be propagated from
        cnn::real* dxSrc, /// gradients to be propagated to
        cudnnTensorDescriptor_t srcTensorDesc,
        cudnnTensorDescriptor_t dstTensorDesc,
        cudnnPoolingDescriptor_t     poolingDesc,
        cudnnHandle_t cudnnHandle);

} // namespace gpu
} // namespace cnn

#endif
