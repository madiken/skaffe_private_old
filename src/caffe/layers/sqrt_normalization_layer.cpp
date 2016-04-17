#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <math.h> 
namespace caffe {

template <typename Dtype>
void SqrtNormalizationLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
  buffer_.Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width()); 
}


template <typename Dtype>
void SqrtNormalizationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width()); 
}

template <typename Dtype>
void SqrtNormalizationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  
  
  caffe_cpu_sign(bottom[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data());

  //caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(),  buffer_.mutable_cpu_data());
 // caffe_add_scalar(bottom[0]->count(), Dtype(0.1), buffer_.mutable_cpu_data());

  caffe_abs(bottom[0]->count(), bottom[0]->cpu_data(), buffer_.mutable_cpu_data());
  //caffe_abs(bottom[0]->count(),buffer_.cpu_data(), buffer_.mutable_cpu_data());

  caffe_powx(bottom[0]->count(), buffer_.cpu_data(), Dtype(0.5), buffer_.mutable_cpu_data());
  caffe_mul(bottom[0]->count(), buffer_.cpu_data(), top[0]->cpu_data(), top[0]->mutable_cpu_data());
}

template <typename Dtype>
void SqrtNormalizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
  caffe_set(bottom[0]->count(), Dtype(0.5), bottom[0]->mutable_cpu_diff());
  caffe_mul(bottom[0]->count(), top[0]->cpu_diff(), bottom[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
  caffe_div(bottom[0]->count(), bottom[0]->cpu_diff(), buffer_.cpu_data(), bottom[0]->mutable_cpu_diff());
  for (int i = 0; i < bottom[0]->count(); i++){
        if (isnan(bottom[0]->cpu_diff()[i])) {
          LOG(INFO) << bottom[0]->cpu_diff()[i] << " buffer: " << buffer_.cpu_diff()[i]  << " top: " << top[0]->cpu_diff()[i];
        }
  }

}

#ifdef CPU_ONLY
STUB_GPU(SqrtNormalizationLayer);
#endif

INSTANTIATE_CLASS(SqrtNormalizationLayer);
REGISTER_LAYER_CLASS(SqrtNormalization);
}  // namespace caffe
