#include <vector>

#include "caffe/layers/one_versus_all_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void OneVersusAllLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  do_sigmoid_ = this->layer_param_.one_versus_all_param().do_sigmoid();

  if(do_sigmoid_) {
    sigmoid_bottom_vec_.clear();
    sigmoid_bottom_vec_.push_back(bottom[0]);
    sigmoid_top_vec_.clear();
    sigmoid_top_vec_.push_back(sigmoid_output_.get());
    sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  }
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

  if(do_sigmoid_) {
    sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  }
}

template <typename Dtype>
void OneVersusAllLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* preds = NULL;
  if(do_sigmoid_) {
    sigmoid_bottom_vec_[0] = bottom[0];
    sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
    preds = sigmoid_output_->cpu_data();
  } else {
    preds = bottom[0]->cpu_data();
  }
  Dtype loss = 0;
  tsigsum_mul = 0;
  tsigsum_add = 0;
  const Dtype* target = bottom[1]->cpu_data();
  const int count = bottom[0]->count();
  Dtype pred;

#if 0
  const Dtype* input_data = bottom[0]->cpu_data();
  // Compute the loss (negative log likelihood)
  // Stable version of loss computation from input data
  for (int i = 0; i < count; ++i) {
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
  }
#else
  for (int i = 0; i < count; ++i) {
    if (has_ignore_label_ && target[i] == ignore_label_) {
      continue;
    }
    CHECK(preds[i] > static_cast<Dtype>(-0.000001) && preds[i] < static_cast<Dtype>(1.000001))
          <<" pred["<<i<<"] == "<<preds[i]<<" which is < 0 or > 1 !!  note: preds[i]-1.0 = "<<(preds[i]-1.0);
    CHECK(target[i] >= 0.0 && target[i] <= 1.0)
          <<" target was "<<target[i]<<", ignore_label_: "
          <<(has_ignore_label_ ? ignore_label_ : -999999);
    pred = std::min(static_cast<Dtype>(1.0), std::max(static_cast<Dtype>(0.0), preds[i]));
    tsigsum_mul += target[i] * pred;
    tsigsum_add += target[i] + pred;
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
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

#if 0
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
#else
    caffe_set(count, static_cast<Dtype>(0), bottom_diff);

    if(tsigsum_mul > kLOG_THRESHOLD && tsigsum_add > kLOG_THRESHOLD) {
      const Dtype one_over_tsigsum_add = static_cast<Dtype>(1.0) / tsigsum_add;
      const Dtype one_over_tsigsum_mul = static_cast<Dtype>(1.0) / tsigsum_mul;
      if(do_sigmoid_) {
        const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
        for (int i = 0; i < count; ++i) {
          if (has_ignore_label_ == false || target[i] != ignore_label_) {
            bottom_diff[i] = (one_over_tsigsum_add - target[i] * one_over_tsigsum_mul)
                * sigmoid_output_data[i] * (static_cast<Dtype>(1.0) - sigmoid_output_data[i]);
          }
        }
      } else {
        for (int i = 0; i < count; ++i) {
          if (has_ignore_label_ == false || target[i] != ignore_label_) {
            bottom_diff[i] = one_over_tsigsum_add - target[i] * one_over_tsigsum_mul;
          }
        }
      }
    }
#endif

    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

//#ifdef CPU_ONLY
//STUB_GPU_BACKWARD(OneVersusAllLossLayer, Backward);
//#endif

INSTANTIATE_CLASS(OneVersusAllLossLayer);
REGISTER_LAYER_CLASS(OneVersusAllLoss);

}  // namespace caffe
