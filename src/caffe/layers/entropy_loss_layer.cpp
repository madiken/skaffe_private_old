#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }
  
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);

  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);

  vector<int> entropy_shape = bottom[0]->shape();
  entropy_shape[softmax_axis_] = 1;
  entropy_.Reshape(entropy_shape);

  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  Dtype* entropy_data = entropy_.mutable_cpu_data();
  int dim = prob_.count() / outer_num_;
  int channels = bottom[0]->shape(softmax_axis_);

  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      entropy_data[i * inner_num_ + j] = 0;
      for (int k = 0; k < channels; ++k) {
        int index = i * dim + k * inner_num_ + j;
        entropy_data[i * inner_num_ + j] -= prob_data[index] * 
            log(std::max(prob_data[index], Dtype(FLT_MIN)));
      }
      loss += entropy_data[i * inner_num_ + j];
    }
  }

  top[0]->mutable_cpu_data()[0] = loss / outer_num_;

  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* entropy_data = entropy_.cpu_data();
    int dim = prob_.count() / outer_num_;
    int channels = bottom[0]->shape(softmax_axis_);
    
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        Dtype entropy_value = entropy_data[i * inner_num_ + j];
        for (int k = 0; k < channels; ++k) {
          int index = i * dim + k * inner_num_ + j;
          bottom_diff[index] = -prob_data[index] * (entropy_value + 
              log(std::max(prob_data[index], Dtype(FLT_MIN))));
        }
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(EntropyLossLayer);
#endif

INSTANTIATE_CLASS(EntropyLossLayer);
REGISTER_LAYER_CLASS(EntropyLoss);

}  // namespace caffe
