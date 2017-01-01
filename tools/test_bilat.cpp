#include <iostream>
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include <boost/thread.hpp>
using std::cout; using std::endl;


int main(int argc, char** argv) {

#ifndef CPU_ONLY
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif

  const std::string file_prototxt("examples/BILATERAL_TEST/Test.prototxt");
  //const std::string file_prototxt("examples/BILATERAL_TEST/Test_meanfield.prototxt");
  caffe::Net<float> mynet(file_prototxt, caffe::TEST);
  while(1) {
    std::cout<<"mynet.Forward();"<<std::endl;
    const caffe::vector<caffe::Blob<float>*> ret = mynet.Forward();
    boost::this_thread::sleep(boost::posix_time::milliseconds(1));
  }
  return 0;
}
