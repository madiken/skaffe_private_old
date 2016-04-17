#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/messenger.hpp"

namespace caffe {

template <typename Dtype>
class ScalerCoefficientHandler: public Listener {
 public:
  ScalerCoefficientHandler(Dtype lower_bound, Dtype upper_bound, 
                           Dtype alpha, Dtype max_iter, Dtype* coeff)
      : lower_bound_(lower_bound), upper_bound_(upper_bound), alpha_(alpha),
        max_iter_(max_iter), coeff_(*coeff) {
    height_ = upper_bound_ - lower_bound_;
  }

  void handle(void* message) {
    int iter = *(static_cast<int*>(message));
    Dtype progress = std::min(Dtype(1), static_cast<Dtype>(iter) / max_iter_);

    // coeff_ = 2.f * height_ / (1.f + exp(-alpha_ * progress)) - 
    //          height_ + lower_bound_;
    coeff_ = lower_bound_ + progress * height_;

    // LOG(INFO) << "iter = " << iter << " progress = " << progress << " coeff = " << coeff_;
  }

 private:
  Dtype lower_bound_, upper_bound_, alpha_, max_iter_, height_;
  Dtype& coeff_;
};

template <typename Dtype>
void ScalerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  
  lower_bound_ = this->layer_param_.gradient_scaler_param().lower_bound();
  upper_bound_ = this->layer_param_.gradient_scaler_param().upper_bound();
  alpha_ = this->layer_param_.gradient_scaler_param().alpha();
  max_iter_ = this->layer_param_.gradient_scaler_param().max_iter();
  coeff_ = 1;

  DCHECK(lower_bound_ <= upper_bound_);
  DCHECK(alpha_ >= 0);
  DCHECK(max_iter_ >= 1);
  
  Messenger::AddListener("SOLVER_ITER_CHANGED", 
      new ScalerCoefficientHandler<Dtype>(lower_bound_, upper_bound_, 
                                          alpha_, max_iter_, &coeff_));
}

// Compute y = (shift + scale * x)^power
template <typename Dtype>
void ScalerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  caffe_copy(count, bottom_data, top_data);
  if (coeff_ != Dtype(1)) {
    caffe_scal(count, coeff_, top_data);
  }
}

template <typename Dtype>
void ScalerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->cpu_diff();

    caffe_copy(count, top_diff, bottom_diff);
    if (coeff_ != Dtype(1)) {
      caffe_scal(count, coeff_, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ScalerLayer);
#endif

INSTANTIATE_CLASS(ScalerLayer);
REGISTER_LAYER_CLASS(Scaler);

}  // namespace caffe
