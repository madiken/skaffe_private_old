#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void MultidataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  for (int top_index = 0; top_index < top.size(); ++top_index) {
    // Reshape to loaded data.
    top[top_index]->Reshape(this->prefetch_data_[top_index]->num(), 
        this->prefetch_data_[top_index]->channels(), 
        this->prefetch_data_[top_index]->height(), 
        this->prefetch_data_[top_index]->width());
    // Copy the data
    caffe_copy(this->prefetch_data_[top_index]->count(), 
        this->prefetch_data_[top_index]->cpu_data(),
        top[top_index]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(MultidataLayer);

}  // namespace caffe
