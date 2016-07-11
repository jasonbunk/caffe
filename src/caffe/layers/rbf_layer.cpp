#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/rbf_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RBFLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  CHECK_EQ(bottom[0]->num_axes(), 4) << "todo: generalized dimensions";
  
  channelsin_ = bottom[0]->shape(1);
  numout_ = this->layer_param_.rbf_param().num_output();
  centroidsperout_ = this->layer_param_.rbf_param().centroids_per_output();
  sharecenters_ = this->layer_param_.rbf_param().share_centroids();
  
  CHECK_GT(channelsin_, 0);
  CHECK_GT(numout_, 0);
  CHECK_GT(centroidsperout_, 0);
  
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(3);
    // shape is same for alphas and betas
    vector<int> weight_shape(2);
    weight_shape[0] = numout_;
    weight_shape[1] = centroidsperout_;
    // Initialize and fill scalars for each basis centroid
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > scalar_filler(GetFiller<Dtype>(
                this->layer_param_.rbf_param().scalar_filler()));
    scalar_filler->Fill(this->blobs_[0].get());
    // Initialize and fill widths of each basis centroid
    this->blobs_[1].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > width_filler(GetFiller<Dtype>(
                this->layer_param_.rbf_param().width_filler()));
    width_filler->Fill(this->blobs_[1].get());
    // Initialize and fill centroids
    if(sharecenters_) {
      weight_shape.resize(2);
      weight_shape[0] = centroidsperout_;
      weight_shape[1] = channelsin_;
    } else {
      weight_shape.resize(3);
      weight_shape[0] = numout_;
      weight_shape[1] = centroidsperout_;
      weight_shape[2] = channelsin_;
    }
    this->blobs_[2].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > centroid_filler(GetFiller<Dtype>(
                this->layer_param_.rbf_param().centroid_filler()));
    centroid_filler->Fill(this->blobs_[2].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void RBFLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  CHECK_EQ(bottom[0]->num_axes(), 4) << "todo: generalized dimensions";
  
  CHECK_EQ(bottom[0]->shape(1), channelsin_)
             << "Number of input channels must be unchanged.";
  
  // The top shape will be the bottom shape, except the channel dimension will
  // have a different number of outputs.
  vector<int> top_shape = bottom[0]->shape();
  top_shape[1] = numout_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void RBFLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int spatialdim = bottom[0]->shape(2) * bottom[0]->shape(3);
  const int batchdim = bottom[0]->shape(0);
  const int minperdatum = bottom[0]->count() / batchdim;
  const int moutprdatum = top[0]->count() / batchdim;
  
  const Dtype* scalars = this->blobs_[0]->cpu_data();
  const Dtype* widths  = this->blobs_[1]->cpu_data();
  const Dtype* centers = this->blobs_[2]->cpu_data();
  
  const int cenoutdim = centroidsperout_ * channelsin_;
  Dtype temp, beta, sumdif, sumtot;
  
  for(int mm=0; mm<batchdim; ++mm) {
    for(int ss=0; ss<spatialdim; ++ss) {
      for(int ii=0; ii<numout_; ++ii) {
        sumtot = ((Dtype)0);
        for(int jj=0; jj<centroidsperout_; ++jj) {
          beta = widths[ii*centroidsperout_ + jj]; // shorter notation
          sumdif = ((Dtype)0);
          for(int kk=0; kk<channelsin_; ++kk) {
            if(sharecenters_) {
              temp = bottom_data[mm*minperdatum + kk*spatialdim + ss]
                         - centers[jj*channelsin_ + kk];
            } else {
              temp = bottom_data[mm*minperdatum + kk*spatialdim + ss]
                         - centers[ii*cenoutdim + jj*channelsin_ + kk];
            }
            sumdif += fabs(temp*temp);
          }
          sumdif = exp(-sumdif*beta*beta);
          
          sumtot += sumdif * scalars[ii*centroidsperout_ + jj];
        }
        top_data[mm*moutprdatum + ii*spatialdim + ss] = sumtot;
      }
    }
  }
}

template <typename Dtype>
void RBFLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
  const Dtype* bottom_data = bottom[0]->cpu_data();
  
  Dtype* difscalar = this->blobs_[0]->mutable_cpu_diff();
  Dtype* difwidths = this->blobs_[1]->mutable_cpu_diff();
  Dtype* difcenter = this->blobs_[2]->mutable_cpu_diff();
  Dtype* bottomdiff = bottom[0]->mutable_cpu_diff();
  
  caffe_memset(this->blobs_[0]->count()*sizeof(Dtype), 0, difscalar);
  caffe_memset(this->blobs_[1]->count()*sizeof(Dtype), 0, difwidths);
  caffe_memset(this->blobs_[2]->count()*sizeof(Dtype), 0, difcenter);
  caffe_memset(bottom[0]->count()*sizeof(Dtype), 0, bottomdiff);
  
  const int spatialdim = bottom[0]->shape(2) * bottom[0]->shape(3);
  const int batchdim = bottom[0]->shape(0);
  const int minperdatum = bottom[0]->count() / batchdim;
  
  const Dtype* scalars = this->blobs_[0]->cpu_data();
  const Dtype* widths  = this->blobs_[1]->cpu_data();
  const Dtype* centers = this->blobs_[2]->cpu_data();
  
  const int moutprdatum = top[0]->count() / batchdim;
  const Dtype* top_diff = top[0]->cpu_diff();
  
  const int cenoutdim = centroidsperout_ * channelsin_;
  Dtype temp, beta, sumdif, expterm;
  
  if (this->param_propagate_down_[0]
   || this->param_propagate_down_[1]
   || this->param_propagate_down_[2]
   || propagate_down[0]) {
    
    for(int ii=0; ii<numout_; ++ii) {
      for(int jj=0; jj<centroidsperout_; ++jj) {
        beta = widths[ii*centroidsperout_ + jj]; // shorter notation
        for(int mm=0; mm<batchdim; ++mm) {
          for(int ss=0; ss<spatialdim; ++ss) {
            sumdif = ((Dtype)0);
            for(int kk=0; kk<channelsin_; ++kk) {
              if(sharecenters_) {
                temp = bottom_data[mm*minperdatum + kk*spatialdim + ss]
                           - centers[jj*channelsin_ + kk];
              } else {
                temp = bottom_data[mm*minperdatum + kk*spatialdim + ss]
                           - centers[ii*cenoutdim + jj*channelsin_ + kk];
              }
              sumdif += fabs(temp*temp);
            }
            expterm = exp(-sumdif*beta*beta)
                        * top_diff[mm*moutprdatum + ii*spatialdim + ss];
            // gradient w.r.t. scalar
            if(this->param_propagate_down_[0]) {
              difscalar[ii*centroidsperout_ + jj] += expterm;
            }
            
            expterm *= scalars[ii*centroidsperout_ + jj];
            
            // gradient w.r.t. width
            if(this->param_propagate_down_[1]) {
              difwidths[ii*centroidsperout_ + jj] -=
               sumdif * expterm * 2.0 * beta; //(beta >= 0.0 ? 1.0 : -1.0);
            }
            
            for(int kk=0; kk<channelsin_; ++kk) {
              sumdif = 2.0 * expterm * beta*beta;
              if(sharecenters_) {
                sumdif *= (bottom_data[mm*minperdatum + kk*spatialdim + ss]
                         - centers[jj*channelsin_ + kk]);
              } else {
                sumdif *= (bottom_data[mm*minperdatum + kk*spatialdim + ss]
                         - centers[ii*cenoutdim + jj*channelsin_ + kk]);
                
              }
              // gradient w.r.t. centroid
              if(this->param_propagate_down_[2]) {
                if(sharecenters_) {
                  difcenter[               jj*channelsin_ + kk] += sumdif;
                } else {
                  difcenter[ii*cenoutdim + jj*channelsin_ + kk] += sumdif;
                }
              }
              // gradient w.r.t. input
              if(propagate_down[0]) {
                bottomdiff[mm*minperdatum + kk*spatialdim + ss] -= sumdif;
              }
            }
          }
        }
      }
    }
  }
}

//#ifdef CPU_ONLY
//STUB_GPU(RBFLayer);
//#endif

INSTANTIATE_CLASS(RBFLayer);
REGISTER_LAYER_CLASS(RBF);

}  // namespace caffe
