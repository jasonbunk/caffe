#include <vector>

#include <boost/thread.hpp>

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
void rbf_msi_compute_forward(std::vector<int>* argshack, const int threadidx,
           const bool sharecenters_,
           const Dtype* scalars, const Dtype* widths, const Dtype* centers,
           const Dtype* bottom_data, Dtype* top_data) {
  
  const int numthreads       = (*argshack)[0];
  const int centroidsperout_ = (*argshack)[1];
  const int channelsin_      = (*argshack)[2];
  const int numout_          = (*argshack)[3];
  
  const int batchdim         = (*argshack)[4]; // bottom[0]->shape(0);
  const int spatialdim       = (*argshack)[5]; // bottom[0]->shape(2) * bottom[0]->shape(3);
  const int minperdatum      = (*argshack)[6]; // bottom[0]->count() / batchdim;
  const int moutprdatum      = (*argshack)[7]; // top[0]->count() / batchdim;

  const int cenoutdim = centroidsperout_ * channelsin_;
  Dtype temp, beta, sumdif, sumtot;

  const int msiinterval = (batchdim*spatialdim*numout_) / numthreads;
  const int msimin = msiinterval*threadidx;
  const int msimax = (threadidx == (numthreads-1)) ?
                                        (batchdim*spatialdim*numout_) :
                                        msiinterval*(threadidx+1);
  int mm,ss,ii,remaind;
  const int denom_si = spatialdim*numout_;
  
  //for(int mm=0; mm<batchdim; ++mm) {
  //  for(int ss=0; ss<spatialdim; ++ss) {
  //    for(int ii=0; ii<numout_; ++ii) {
  
  for(int msi = msimin; msi < msimax; ++msi) {
    mm = msi / denom_si;
    remaind = msi - mm*denom_si;
    ss = remaind / numout_;
    ii = remaind - ss*numout_;
    
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

template <typename Dtype>
void RBFLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  const Dtype* scalars = this->blobs_[0]->cpu_data();
  const Dtype* widths  = this->blobs_[1]->cpu_data();
  const Dtype* centers = this->blobs_[2]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  std::vector<boost::thread*> allthreads;
  const int numthreads = boost::thread::hardware_concurrency();
  
  // this ugly argument passing is because
  // boost::thread only allows up to 9 arguments to be passed to a function
  std::vector<int> argshack;
  argshack.push_back(numthreads);
  argshack.push_back(centroidsperout_);
  argshack.push_back(channelsin_);
  argshack.push_back(numout_);
  
  argshack.push_back(bottom[0]->shape(0));
  argshack.push_back(bottom[0]->shape(2) * bottom[0]->shape(3));
  argshack.push_back(bottom[0]->count() / bottom[0]->shape(0));
  argshack.push_back(top[0]->count() / bottom[0]->shape(0));
  
  for(int ii=0; ii<numthreads; ++ii) {
    allthreads.push_back(new boost::thread(&rbf_msi_compute_forward<Dtype>,
                  &argshack, ii,
                  sharecenters_,
                  scalars, widths, centers,
                  bottom_data, top_data));
  }
  
  for(int ii=0; ii<numthreads; ++ii) {
    allthreads[ii]->join();
    delete allthreads[ii];
  }
}

template <typename Dtype>
void rbf_msi_compute_backward(std::vector<int>* argshack, const int threadidx,
           const bool sharecenters_,
           std::vector<const Dtype*>* dtypeargshack,
           std::vector<bool>* boolpropdowns,
           Dtype* difscalar, Dtype* difwidths, Dtype* difcenter,
           Dtype* bottomdiff) {
             
  const int numthreads       = (*argshack)[0];
  const int centroidsperout_ = (*argshack)[1];
  const int channelsin_      = (*argshack)[2];
  const int numout_          = (*argshack)[3];
  
  const int spatialdim  = (*argshack)[4]; // bottom[0]->shape(2) * bottom[0]->shape(3)
  const int batchdim    = (*argshack)[5]; // bottom[0]->shape(0)
  const int minperdatum = (*argshack)[6]; // bottom[0]->count() / batchdim
  const int moutprdatum = (*argshack)[7]; // top[0]->count() / batchdim
  const int cenoutdim   = (*argshack)[8]; // centroidsperout_ * channelsin_
  
  const Dtype* bottom_data = (*dtypeargshack)[0]; // bottom[0]->cpu_data()
  const Dtype* scalars     = (*dtypeargshack)[1]; // this->blobs_[0]->cpu_data()
  const Dtype* widths      = (*dtypeargshack)[2]; // this->blobs_[1]->cpu_data()
  const Dtype* centers     = (*dtypeargshack)[3]; // this->blobs_[2]->cpu_data()
  const Dtype* top_diff    = (*dtypeargshack)[4]; // top[0]->cpu_diff()
  
  Dtype temp, beta, sumdif, expterm;
  
  const int ijmsinterval = (numout_*centroidsperout_*batchdim*spatialdim) / numthreads;
  const int ijmsmin = ijmsinterval*threadidx;
  const int ijmsmax = (threadidx == (numthreads-1)) ?
                                (numout_*centroidsperout_*batchdim*spatialdim) :
                                ijmsinterval*(threadidx+1);
  int ii,jj,mm,ss,remaind;
  const int denom_jms = centroidsperout_*batchdim*spatialdim;
  const int denom_ms  = batchdim*spatialdim;
  
  //for(int ii=0; ii<numout_; ++ii) {
  //  for(int jj=0; jj<centroidsperout_; ++jj) {
  //    for(int mm=0; mm<batchdim; ++mm) {
  //      for(int ss=0; ss<spatialdim; ++ss) {
  for(int ijms = ijmsmin; ijms < ijmsmax; ++ijms) {
    ii = ijms / denom_jms;
    remaind = ijms - ii*denom_jms;
    jj = remaind / denom_ms;
    remaind = remaind - jj*denom_ms;
    mm = remaind / spatialdim;
    ss = remaind - mm*spatialdim;
    
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
    expterm = exp(-sumdif*beta*beta)
                * top_diff[mm*moutprdatum + ii*spatialdim + ss];
    // gradient w.r.t. scalar
    if((*boolpropdowns)[0]) {
      difscalar[ii*centroidsperout_ + jj] += expterm;
    }
    
    expterm *= scalars[ii*centroidsperout_ + jj];
    
    // gradient w.r.t. width
    if((*boolpropdowns)[1]) {
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
      if((*boolpropdowns)[2]) {
        if(sharecenters_) {
          difcenter[               jj*channelsin_ + kk] += sumdif;
        } else {
          difcenter[ii*cenoutdim + jj*channelsin_ + kk] += sumdif;
        }
      }
      // gradient w.r.t. input
      if((*boolpropdowns)[3]) {
        bottomdiff[mm*minperdatum + kk*spatialdim + ss] -= sumdif;
      }
    }
  }
}

template <typename Dtype>
void RBFLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  std::vector<Dtype*> diffs;
  diffs.push_back(this->blobs_[0]->mutable_cpu_diff());
  diffs.push_back(this->blobs_[1]->mutable_cpu_diff());
  diffs.push_back(this->blobs_[2]->mutable_cpu_diff());
  diffs.push_back(bottom[0]->mutable_cpu_diff());
  std::vector<int> difsizes;
  difsizes.push_back(this->blobs_[0]->count());
  difsizes.push_back(this->blobs_[1]->count());
  difsizes.push_back(this->blobs_[2]->count());
  difsizes.push_back(bottom[0]->count());

  std::vector<boost::thread*> allthreads;
  const int numthreads = boost::thread::hardware_concurrency();

  std::vector<int> argshack;
  std::vector<const Dtype*> dtypeargshack;
  std::vector<bool> boolpropdowns;

  argshack.push_back(numthreads);
  argshack.push_back(centroidsperout_);
  argshack.push_back(channelsin_);
  argshack.push_back(numout_);

  argshack.push_back(bottom[0]->shape(2) * bottom[0]->shape(3));
  argshack.push_back(bottom[0]->shape(0));
  argshack.push_back(bottom[0]->count() / bottom[0]->shape(0));
  argshack.push_back(top[0]->count() / bottom[0]->shape(0));
  argshack.push_back(centroidsperout_ * channelsin_);

  dtypeargshack.push_back(bottom[0]->cpu_data());
  dtypeargshack.push_back(this->blobs_[0]->cpu_data());
  dtypeargshack.push_back(this->blobs_[1]->cpu_data());
  dtypeargshack.push_back(this->blobs_[2]->cpu_data());
  dtypeargshack.push_back(top[0]->cpu_diff());

  boolpropdowns.push_back(this->param_propagate_down_[0]);
  boolpropdowns.push_back(this->param_propagate_down_[1]);
  boolpropdowns.push_back(this->param_propagate_down_[2]);
  boolpropdowns.push_back(propagate_down[0]);

  if (boolpropdowns[0]
   || boolpropdowns[1]
   || boolpropdowns[2]
   || boolpropdowns[3]) {

    std::vector< std::vector<Dtype*> > diffbuffers(numthreads, std::vector<Dtype*>());

    for(int ii=0; ii<numthreads; ++ii) {
      for(int jj=0; jj<4; ++jj) {
        diffbuffers[ii].push_back((Dtype*)calloc(difsizes[jj],sizeof(Dtype)));
      }
      allthreads.push_back(new boost::thread(&rbf_msi_compute_backward<Dtype>,
                    &argshack, ii,
                    sharecenters_,
                    &dtypeargshack,
                    &boolpropdowns,
                    diffbuffers[ii][0], diffbuffers[ii][1],
                    diffbuffers[ii][2], diffbuffers[ii][3]));
    }

    for(int jj=0; jj<4; ++jj) {
      caffe_memset(difsizes[jj]*sizeof(Dtype), 0, diffs[jj]);
    }

    for(int ii=0; ii<numthreads; ++ii) {
      allthreads[ii]->join();

      for(int jj=0; jj<4; ++jj) {
        for(int kk=0; kk<difsizes[jj]; ++kk) {
          diffs[jj][kk] += diffbuffers[ii][jj][kk];
        }
        free(diffbuffers[ii][jj]);
      }
      delete allthreads[ii];
    }
  }
}

//#ifdef CPU_ONLY
//STUB_GPU(RBFLayer);
//#endif

INSTANTIATE_CLASS(RBFLayer);
REGISTER_LAYER_CLASS(RBF);

}  // namespace caffe
