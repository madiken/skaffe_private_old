#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EltwiseProdGroupLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  //shape must be the same except for channels num
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()); 
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[1]->channels(), 1); 
}

template <typename Dtype>
void EltwiseProdGroupLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EltwiseProdGroupLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int data_offset = 0;
  int mask_offset = 0;
  for(int i = 0; i < bottom[0]->num(); i++){
    for(int j = 0; j < bottom[0]->channels(); j++){
      data_offset = i * bottom[0]->height() * bottom[0]->width() * bottom[0]->channels() + j * bottom[0]->height() * bottom[0]->width();
      mask_offset = i * bottom[1]->height() * bottom[1]->width();
      caffe_mul(bottom[0]->height() * bottom[0]->width(), bottom[0]->cpu_data() + data_offset, bottom[1]->cpu_data() + mask_offset, top[0]->mutable_cpu_data() + data_offset);
    }
  }
}

template <typename Dtype>
void EltwiseProdGroupLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
 
  int data_offset = 0;
  int mask_offset = 0;
  for(int i = 0; i < bottom[0]->num(); i++){
    for(int j = 0; j < bottom[0]->channels(); j++){
      data_offset = i * bottom[0]->height() * bottom[0]->width() * bottom[0]->channels() + j * bottom[0]->height() * bottom[0]->width();
      mask_offset = i * bottom[1]->height() * bottom[1]->width() ;
      //caffe_copy(bottom[0]->height() * bottom[0]->width(), bottom[1]->cpu_data() + mask_offset, bottom[0]->mutable_cpu_diff() + data_offset);
      caffe_mul(bottom[0]->height() * bottom[0]->width(), top[0]->cpu_diff() + data_offset, bottom[1]->cpu_data() + mask_offset, bottom[0]->mutable_cpu_diff() + data_offset);
    }
  }
  
  //caffe_mul(bottom[0]->count(), bottom[0]->cpu_diff(), top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff()); 

}

#ifdef CPU_ONLY
STUB_GPU(EltwiseProdGroupLayer);
#endif

INSTANTIATE_CLASS(EltwiseProdGroupLayer);
REGISTER_LAYER_CLASS(EltwiseProdGroup);

}  // namespace caffe
