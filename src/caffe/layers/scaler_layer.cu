#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ScalerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  caffe_copy(count, bottom_data, top_data);
  if (coeff_ != Dtype(1)) {
    caffe_gpu_scal(count, coeff_, top_data);
  }
}

template <typename Dtype>
void ScalerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();

    caffe_copy(count, top_diff, bottom_diff);
    if (coeff_ != Dtype(1)) {
      caffe_gpu_scal(count, coeff_, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ScalerLayer);


}  // namespace caffe
