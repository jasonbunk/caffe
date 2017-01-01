#include <vector>

#include "caffe/layers/bilateral_filter_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BilateralFilterLayer<Dtype>::cudastream_init() {
#ifndef CPU_ONLY
  if(stream_ == NULL) {
    stream_ = new cudaStream_t;
    CUDA_CHECK(cudaStreamCreate(stream_));
  }
#endif
}
template <typename Dtype>
void BilateralFilterLayer<Dtype>::cudastream_free() {
#ifndef CPU_ONLY
  if(stream_ != NULL) {
    cudaStreamDestroy(*stream_);
    delete [] stream_;
    stream_ = NULL;
  }
#endif
}

template <typename Dtype>
void BilateralFilterLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  cudastream_init();
  BilateralFilterParameter param = this->layer_param().bilateral_filter_param();
  if(param.has_stdv_space()) {
    stdv_space_ = param.stdv_space();
  }
  if(param.has_stdv_space()) {
    stdv_color_ = param.stdv_color();
  }
  if(param.has_create_spatial_dimension_features()) {
    create_spatial_dimension_features_ = param.create_spatial_dimension_features();
  }
}

template <typename Dtype>
void BilateralFilterLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[0]->num_axes(), bottom[1]->num_axes());
  CHECK_GE(bottom[0]->num_axes(), 3);

  LayerSetUp(bottom, top);
  top[0]->ReshapeLike(*bottom[0]);

  const int nspatialch_wrt = create_spatial_dimension_features_ ? (bottom[1]->num_axes() - 2) : 0;
  const int nchannels_wrt = bottom[1]->shape(1) + nspatialch_wrt;

  if (stdv_color_ > 0.0f || stdv_space_ > 0.0f) {
    CHECK(stdv_color_ > 0.0f && stdv_space_ > 0.0f);
    CHECK(nchannels_wrt > nspatialch_wrt);
    stdv_widths_host_.resize(nchannels_wrt);
    for(int ii=0; ii<nspatialch_wrt; ++ii)
      stdv_widths_host_[ii] = stdv_space_;
    for(int ii=nspatialch_wrt; ii<nchannels_wrt; ++ii)
      stdv_widths_host_[ii] = stdv_color_;
  }

  switch (Caffe::mode()) {
    case Caffe::CPU:
    bilateral_interface_cpu_.reset(new PermutohedralOp_CPU<Dtype>(stdv_widths_host_));
    break;
  #ifndef CPU_ONLY
    case Caffe::GPU:
    bilateral_interface_gpu_.reset(new_permutohedral_gpu_op<Dtype>(nchannels_wrt,
                                                            stdv_widths_host_,
                                                            create_spatial_dimension_features_));
    break;
  #endif
    default: LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void BilateralFilterLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  bilateral_interface_cpu_->Forward(bottom[0], bottom[1], top[0]);
}

template <typename Dtype>
void BilateralFilterLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  bilateral_interface_cpu_->Backward(
                                propagate_down[0], propagate_down[1],
                                bottom[0], bottom[1], top[0]);
  // Scale gradient
  const Dtype loss_weight = top[0]->cpu_diff()[0];
  if(propagate_down[0]) {
    caffe_scal(bottom[0]->count(), loss_weight, bottom[0]->mutable_cpu_diff());
  }
  if(propagate_down[1]) {
    caffe_scal(bottom[1]->count(), loss_weight, bottom[1]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(BilateralFilterLayer);
#endif

INSTANTIATE_CLASS_f(BilateralFilterLayer);
REGISTER_LAYER_CLASS_f(BilateralFilter);

}  // namespace caffe
