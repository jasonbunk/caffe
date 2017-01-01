#ifndef CAFFE_BILATERAL_FILTER_LAYER_HPP_
#define CAFFE_BILATERAL_FILTER_LAYER_HPP_

#include <vector>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/bilateral_filter/permutohedral_ops.h"

namespace caffe {

template <typename Dtype>
class BilateralFilterLayer : public Layer<Dtype> {
 public:
  explicit BilateralFilterLayer(const LayerParameter& param)
#ifndef CPU_ONLY
      : Layer<Dtype>(param), stdv_space_(-1.0f), stdv_color_(-1.0f), create_spatial_dimension_features_(true), stream_(NULL) {}
#else
      : Layer<Dtype>(param), stdv_space_(-1.0f), stdv_color_(-1.0f), create_spatial_dimension_features_(true) {}
#endif

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual inline const char* type() const { return "BilateralFilter"; }

  ~BilateralFilterLayer() {cudastream_free();}
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void cudastream_free();
  void cudastream_init();

  float stdv_space_;
  float stdv_color_;
  bool create_spatial_dimension_features_;
  vector<float> stdv_widths_host_;
  shared_ptr< PermutohedralOp_CPU<Dtype> > bilateral_interface_cpu_;
#ifndef CPU_ONLY
  cudaStream_t* stream_;
  shared_ptr< PermutohedralOp_GPU<Dtype> > bilateral_interface_gpu_;
#endif
};

}  // namespace caffe

#endif
