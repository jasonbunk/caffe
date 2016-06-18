#include <vector>

#include "caffe/layers/one_versus_all_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void OneVersusAllLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void OneVersusAllLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "ONE_VERSUS_ALL_LOSS layer inputs must have the same count.";
      
  int numinbatch = bottom[0]->shape(0);
  int batchdim   = bottom[0]->count() / numinbatch;
  int spatialdim = bottom[0]->count() / (numinbatch * bottom[0]->shape(1));
  int numchannel = batchdim / spatialdim;
  CHECK_EQ(numchannel, 1) <<
      "Must be binary classification with 1 output channel (per position)..."
      << " todo: multilabel";
  
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void OneVersusAllLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
  Dtype loss = 0;
  tsigsum_mul = 0;
  tsigsum_add = 0;
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();


#if 0
  // Compute the loss (negative log likelihood)
  // Stable version of loss computation from input data
  for (int i = 0; i < count; ++i) {
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
  }
#else
  for (int i = 0; i < count; ++i) {
    tsigsum_mul += target[i] * sigmoid_output_data[i];
    tsigsum_add += target[i] + sigmoid_output_data[i];
  }
  if(tsigsum_mul > kLOG_THRESHOLD && tsigsum_add > kLOG_THRESHOLD) {
    loss = log(tsigsum_add) - 0.69314718055994530942 - log(tsigsum_mul);
  }
#endif


  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void OneVersusAllLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype dsig = 0;
    
#if 0
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
#else
  if(tsigsum_mul > kLOG_THRESHOLD && tsigsum_add > kLOG_THRESHOLD) {
    for (int i = 0; i < count; ++i) {
      dsig = sigmoid_output_data[i] * (1 - sigmoid_output_data[i]);
      bottom_diff[i] = -target[i] * dsig / tsigsum_mul + dsig / tsigsum_add;
    }
  } else {
    for (int i = 0; i < count; ++i)
      bottom_diff[i] = 0.0;
  }
#endif


    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(OneVersusAllLossLayer, Backward);
#endif

INSTANTIATE_CLASS(OneVersusAllLossLayer);
REGISTER_LAYER_CLASS(OneVersusAllLoss);

}  // namespace caffe
