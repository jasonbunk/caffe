#include <vector>

#include "caffe/layers/bilateral_filter_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BilateralFilterLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  bilateral_interface_gpu_->Forward(stream_,
                                    -1,
                                    bottom[0], bottom[1], top[0]);
}

template <typename Dtype>
void BilateralFilterLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  bilateral_interface_gpu_->Backward(stream_,
                                      -1,
                                      propagate_down[0], propagate_down[1],
                                      bottom[0], bottom[1], top[0]);
  // Scale gradient
  const Dtype loss_weight = top[0]->cpu_diff()[0];
  if(propagate_down[0]) {
    caffe_gpu_scal(bottom[0]->count(), loss_weight, bottom[0]->mutable_gpu_diff());
  }
  if(propagate_down[1]) {
    caffe_gpu_scal(bottom[1]->count(), loss_weight, bottom[1]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS_f(BilateralFilterLayer);

}  // namespace caffe
