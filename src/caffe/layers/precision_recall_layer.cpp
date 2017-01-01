#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/precision_recall_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PrecisionRecallLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
  pred_thresh_ = this->layer_param_.precision_recall_param().pred_threshold();
}

template <typename Dtype>
void PrecisionRecallLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  label_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(1,4); //3 outputs: precision, recall, harmonic mean of precision & recall
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void PrecisionRecallLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  CHECK_LE(num_labels, 2) << "Precision and recall is only for binary classification";

  caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
  Dtype count0 = ((Dtype)0);
  Dtype count1 = ((Dtype)0);
  Dtype correct0 = ((Dtype)0);
  Dtype correct1 = ((Dtype)0);
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value = static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      if(label_value == 0) {count0 += 1.0;} else {count1 += 1.0;}
      CHECK_GE(label_value, 0);
      CHECK_LE(label_value,num_labels) << "(has_ignore_label_ = "
                  <<has_ignore_label_<<", ignore_label_ = "<<ignore_label_<<")";
      if(num_labels == 1) {
        // check if true label is top prediction
        if (bottom_data[i * dim + j] < pred_thresh_ && label_value == 0) {
          correct0++;
        } else if (bottom_data[i * dim + j] >= pred_thresh_ && label_value == 1) {
          correct1++;
        }
      } else {
        // Top-1 accuracy
        std::vector<std::pair<Dtype, int> > bottom_data_vector;
        for (int k = 0; k < num_labels; ++k) {
          bottom_data_vector.push_back(std::make_pair(
              bottom_data[i * dim + k * inner_num_ + j], k));
        }
        std::partial_sort(
            bottom_data_vector.begin(), bottom_data_vector.begin() + 1,
            bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
        // check if true label is top prediction
        if (bottom_data_vector[0].second == label_value) {
          if(label_value == 0) {correct0++;} else {correct1++;}
        }
      }
    }
  }

  Dtype falsepositives = count0 - correct0;

  Dtype precis = (correct1 + falsepositives) <= 0.0 ? 0.0 : (correct1 / (correct1 + falsepositives));
  Dtype recall = count1 <= 0.0 ? 0.0 : correct1 / count1;

  //Dtype specifcity = (falsepositives + correct0) <= 0.0 ? 0.0 : correct0 / (falsepositives + correct0);

  Dtype hmdenom = precis + recall;
  //Dtype hmdenom = specifcity + recall;

  top[0]->mutable_cpu_data()[0] = precis;
  top[0]->mutable_cpu_data()[1] = recall;
  //top[0]->mutable_cpu_data()[2] = specifcity;
  top[0]->mutable_cpu_data()[2] = hmdenom <= kLOG_THRESHOLD ? 0.0 : (2.0 * precis * recall / hmdenom);
  //top[0]->mutable_cpu_data()[3] = hmdenom <= 0.0 ? 0.0 : (2.0 * specifcity * recall / hmdenom);

  //output #3 is accuracy
  top[0]->mutable_cpu_data()[3] = (correct0 + correct1) / (count0 + count1);
}

INSTANTIATE_CLASS(PrecisionRecallLayer);
REGISTER_LAYER_CLASS(PrecisionRecall);

}  // namespace caffe
