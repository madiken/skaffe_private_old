#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
void EltwiseProdGroupLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int data_offset = 0;
  int mask_offset = 0;
  for(int i = 0; i < bottom[0]->num(); i++){
    for(int j = 0; j < bottom[0]->channels(); j++){
      data_offset = i * bottom[0]->height() * bottom[0]->width() * bottom[0]->channels() + j * bottom[0]->height() * bottom[0]->width();
      mask_offset = i * bottom[1]->height() * bottom[1]->width();
      caffe_gpu_mul(bottom[0]->height() * bottom[0]->width(), bottom[0]->gpu_data() + data_offset, bottom[1]->gpu_data() + mask_offset, top[0]->mutable_gpu_data() + data_offset);
    }
  }
}

template <typename Dtype>
void EltwiseProdGroupLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
 
  int data_offset = 0;
  int mask_offset = 0;
  for(int i = 0; i < bottom[0]->num(); i++){
    for(int j = 0; j < bottom[0]->channels(); j++){
      data_offset = i * bottom[0]->height() * bottom[0]->width() * bottom[0]->channels() + j * bottom[0]->height() * bottom[0]->width();
      mask_offset = i * bottom[1]->height() * bottom[1]->width() ;
      //caffe_copy(bottom[0]->height() * bottom[0]->width(), bottom[1]->gpu_data() + mask_offset, bottom[0]->mutable_gpu_diff() + data_offset);
      caffe_gpu_mul(bottom[0]->height() * bottom[0]->width(), top[0]->gpu_diff() + data_offset, bottom[1]->gpu_data() + mask_offset, bottom[0]->mutable_gpu_diff() + data_offset);
    }
  }
  
 // caffe_gpu_mul(bottom[0]->count(), bottom[0]->gpu_diff(), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff()); 

}

INSTANTIATE_LAYER_GPU_FUNCS(EltwiseProdGroupLayer);
}  // namespace caffe
