#include <vector>

#include "caffe/layers/testopencvpreview_layer.hpp"
#include "caffe/util/math_functions.hpp"
#if USE_OPENCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#define MINOF2(x,y) ((x)<(y)?(x):(y))
#define MAXOF2(x,y) ((x)>(y)?(x):(y))

namespace caffe {

template <typename Dtype>
void TestOpenCVPreviewLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not allow in-place computation.";
}

template <typename Dtype>
void TestOpenCVPreviewLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  caffe_copy(bottom[0]->count(), bottom_data, top[0]->mutable_cpu_data());
  visualize_buf(bottom[0]);
}

template <typename Dtype>
void TestOpenCVPreviewLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->cpu_diff();
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(count, top_diff, bottom_diff);
  }
}

template <typename Dtype>
void visualize_buf(Blob<Dtype> const* buf) {
#if USE_OPENCV
    const Dtype* buf_data = buf->cpu_data();
    CHECK_EQ(buf->num_axes(), 4);
    const int nbatch = buf->shape(0);
    const int nchans = MINOF2(buf->shape(1),3);
    const int nrows = buf->shape(2);
    const int ncols = buf->shape(3);
    const int bytesperchan = ncols*nrows*sizeof(Dtype);
    std::vector< cv::Mat_<Dtype> > testimgchans;
    for(int cc=0; cc<nchans; ++cc) {
      testimgchans.push_back(cv::Mat_<Dtype>(nrows, ncols));
    }
    cv::Mat mergedmat;
    double minval,maxval,globalmin,globalmax;
    globalmin = 1e20; globalmax = -1e20;

    for(int mm=0; mm<nbatch; ++mm) {
      for(int cc=0; cc<nchans; ++cc) {
        memcpy(testimgchans[cc].data,
               buf_data + buf->offset(mm, cc, 0, 0),
               bytesperchan);
        cv::minMaxIdx(testimgchans[cc], &minval, &maxval);
        globalmin = MINOF2(globalmin, minval);
        globalmax = MAXOF2(globalmax, maxval);
      }
      std::cout<<"caffe-img (displayed): (min,max) = ("<<globalmin<<", "<<globalmax<<")"<<std::endl;
      cv::merge(testimgchans, mergedmat);
      cv::imshow("caffe-img", (mergedmat-globalmin)/(globalmax-globalmin));
      cv::waitKey(0);
    }
#else
    std::cout<<"visualize_buf(): can't visualize image, please recompile with USE_OPENCV == 1"<<std::endl;
#endif
}

// instantiate
template void visualize_buf(Blob<float> const* buf);
template void visualize_buf(Blob<double> const* buf);

INSTANTIATE_CLASS(TestOpenCVPreviewLayer);
REGISTER_LAYER_CLASS(TestOpenCVPreview);

}  // namespace caffe
