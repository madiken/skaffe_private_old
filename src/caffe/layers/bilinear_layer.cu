#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <math.h> 
namespace caffe {

template <typename Dtype>
void BilinearLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  for (int i = 0; i < bottom[0]->num(); i++){
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, bottom[0]->channels(), bottom[1]->channels(), bottom[0]->width() * bottom[0]->height(),
        (Dtype)1., bottom[0]->gpu_data() + i * bottom[0]->channels() * bottom[0]->width() * bottom[0]->height(), bottom[1]->gpu_data() + i * bottom[1]->channels() * bottom[1]->width() * bottom[1]->height(), (Dtype)0., top[0]->mutable_gpu_data() + i * top[0]->channels() * top[0]->width() * top[0]->height());
  }
}

template <typename Dtype>
void BilinearLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
  for (int i = 0; i < bottom[0]->num(); i++){

	  if (propagate_down[0]) {
	    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, bottom[0]->channels(), bottom[0]->width() * bottom[0]->height(), bottom[1]->channels(), (Dtype)1., top[0]->gpu_diff() + i * top[0]->channels() * top[0]->width() * top[0]->height(), bottom[1]->gpu_data() + i * bottom[1]->channels() * bottom[1]->width() * bottom[1]->height(), (Dtype)0., bottom[0]->mutable_gpu_diff() + i * bottom[0]->channels() * bottom[0]->width() * bottom[0]->height());
	  }
	  if (propagate_down[1]) {
	    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, bottom[1]->channels(), bottom[0]->width() * bottom[0]->height(), bottom[0]->channels(), (Dtype)1., top[0]->gpu_diff() + i * top[0]->channels() * top[0]->width() * top[0]->height(), bottom[0]->gpu_data() + i * bottom[0]->channels() * bottom[0]->width() * bottom[0]->height(), (Dtype)0., bottom[1]->mutable_gpu_diff() + i * bottom[1]->channels() * bottom[1]->width() * bottom[1]->height());
	  }
  }
}
INSTANTIATE_LAYER_GPU_FUNCS(BilinearLayer);
}  // namespace caffe
