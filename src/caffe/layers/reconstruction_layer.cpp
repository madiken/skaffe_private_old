#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ReconstructionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  allow_refill_ = true;
  DummyDataLayer<Dtype>::LayerSetUp(bottom, top);
  Forward_cpu(bottom, top);
  allow_refill_ = false;

  this->blobs_.resize(top.size());
  for (int i = 0; i < top.size(); ++i) {
    this->blobs_[i].reset(top[i]);
  }
}

template <typename Dtype>
void ReconstructionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (allow_refill_) {
    DummyDataLayer<Dtype>::Forward_cpu(bottom, top);
  }
}

INSTANTIATE_CLASS(ReconstructionLayer);
REGISTER_LAYER_CLASS(Reconstruction);

}  // namespace caffe
