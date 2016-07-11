#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/rbf_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include <iostream>
using std::cout; using std::endl;

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

#define NMINIBATCH 3
#define NWIDTH 4
#define NHIGHT 5

template <typename TypeParam>
class RBFLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  RBFLayerTest()
      : blob_bottom_(new Blob<Dtype>(NMINIBATCH, 2, NWIDTH, NHIGHT)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    ConstantFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~RBFLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};


TYPED_TEST_CASE(RBFLayerTest, TestDtypesAndDevices);


TYPED_TEST(RBFLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  RBFParameter* rbf_param = layer_param.mutable_rbf_param();
  rbf_param->set_num_output(1);
  rbf_param->set_centroids_per_output(3);
  shared_ptr<RBFLayer<Dtype> > layer(new RBFLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), NMINIBATCH);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), NWIDTH);
  EXPECT_EQ(this->blob_top_->width(), NHIGHT);
}


TYPED_TEST(RBFLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  RBFParameter* rbf_param = layer_param.mutable_rbf_param();
  rbf_param->set_num_output(1);
  rbf_param->set_centroids_per_output(3);
  rbf_param->mutable_scalar_filler()->set_type("constant");
  rbf_param->mutable_width_filler()->set_type("constant");
  rbf_param->mutable_centroid_filler()->set_type("constant");
  shared_ptr<RBFLayer<Dtype> > layer(new RBFLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  
  Dtype* databottom = this->blob_bottom_->mutable_cpu_data();
  const int minibatchdim = 2*NWIDTH*NHIGHT;
  const int spatialdim = NWIDTH*NHIGHT;
  
  for(int mm=0; mm<NMINIBATCH; ++mm) {
    for(int ss=0; ss<(NWIDTH*NHIGHT); ++ss) {
      if(ss < 3) {
        databottom[mm*minibatchdim + 0*spatialdim + ss] = 2.0;
        databottom[mm*minibatchdim + 1*spatialdim + ss] = 1.0;
      } else {
        databottom[mm*minibatchdim + 0*spatialdim + ss] = -1.0;
        databottom[mm*minibatchdim + 1*spatialdim + ss] = -3.0;
      }
    }
  }
  Dtype* datascalar = layer->blobs()[0].get()->mutable_cpu_data();
  Dtype* datawidths = layer->blobs()[1].get()->mutable_cpu_data();
  Dtype* datacenter = layer->blobs()[2].get()->mutable_cpu_data();
  
  datascalar[0] = 7.0;
  datascalar[1] = -7.0;
  datascalar[2] = 4.7707331819676028;
  
  datawidths[0] = -sqrt(2.0);
  datawidths[1] = -sqrt(25.0);
  datawidths[2] = -sqrt(0.25);
  
  datacenter[0*2+0] = -1.5;
  datacenter[0*2+1] = -2.5;
  
  datacenter[1*2+0] = 2.2;
  datacenter[1*2+1] = 1.0;
  
  datacenter[2*2+0] = 0.5;
  datacenter[2*2+1] = -1.0;
  
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
  const Dtype* data = this->blob_top_->cpu_data();
  
  for(int mm=0; mm<NMINIBATCH; ++mm) {
    for(int ss=0; ss<(NWIDTH*NHIGHT); ++ss) {
      Dtype expval0 = 1.0 - 7.0*exp(-1.0);
      Dtype expval1 = 1.0 + 7.0*exp(-1.0);
      if(ss < 3) {
        CHECK_LT(fabs(data[mm*spatialdim + ss] - expval0), 1e-5)
                  << "failed forward pass: data[mm*spatialdim + ss]: "<<data[mm*minibatchdim + ss]<<", expval0: "<<expval0;
      } else {
        CHECK_LT(fabs(data[mm*spatialdim + ss] - expval1), 1e-5)
                  << "failed forward pass: data[mm*spatialdim + ss]: "<<data[mm*minibatchdim + ss]<<", expval1: "<<expval1;
      }
    }
  }
  
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&(*layer), this->blob_bottom_vec_,
      this->blob_top_vec_);
}


TYPED_TEST(RBFLayerTest, TestGradientSHARED) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  RBFParameter* rbf_param = layer_param.mutable_rbf_param();
  rbf_param->set_num_output(6);
  rbf_param->set_centroids_per_output(7);
  rbf_param->set_share_centroids(true);
  rbf_param->mutable_scalar_filler()->set_type("gaussian");
  rbf_param->mutable_width_filler()->set_type("gaussian");
  rbf_param->mutable_centroid_filler()->set_type("gaussian");
  RBFLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(RBFLayerTest, TestGradientUNSHARED) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  RBFParameter* rbf_param = layer_param.mutable_rbf_param();
  rbf_param->set_num_output(6);
  rbf_param->set_centroids_per_output(7);
  rbf_param->set_share_centroids(false);
  rbf_param->mutable_scalar_filler()->set_type("gaussian");
  rbf_param->mutable_width_filler()->set_type("gaussian");
  rbf_param->mutable_centroid_filler()->set_type("gaussian");
  RBFLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


}  // namespace caffe
