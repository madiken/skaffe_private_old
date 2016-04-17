#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class BinomialDevianceLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BinomialDevianceLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(128, 1, 1, 1)),
        blob_bottom_y_(new Blob<Dtype>(128, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()),
        n1(0), n2(0) {
    
    FillerParameter filler_param;
    filler_param.set_mean(0.0);
    filler_param.set_std(0.001); 
    GaussianFiller<Dtype> filler(filler_param); 
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_y_->count(); ++i) {
      int r = caffe_rng_rand() % 2;
      if (r == 0)
           r = -1;

      blob_bottom_y_->mutable_cpu_data()[i] = r;  // -1 or 1
      if (r == 1)
           this->n1++;
      else if (r == -1)
           this->n2++;
    }
    blob_bottom_vec_.push_back(blob_bottom_y_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~BinomialDevianceLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_y_;
    delete blob_top_loss_;
  }
 
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_y_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int  n1;
  int  n2;
};

TYPED_TEST_CASE(BinomialDevianceLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(BinomialDevianceLossLayerTest, TestForward) {

  typedef typename TypeParam::Dtype Dtype; 
  LayerParameter layer_param;
  BinomialDevianceLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype alpha = layer_param.binomial_deviance_loss_param().alpha();
  const Dtype beta = layer_param.binomial_deviance_loss_param().beta();
  const int num = this->blob_bottom_data_->num();
  Dtype loss(0);    
      
  Dtype c = layer_param.binomial_deviance_loss_param().c(); 
  double M = 0; 
  double W = 0; 
  
  for (int i = 0; i < num; ++i) {
    M = static_cast<int>(this->blob_bottom_y_->cpu_data()[i]);
    if (static_cast<int>(this->blob_bottom_y_->cpu_data()[i]) == 1)
      W = 1.0/this->n1;
    else if (static_cast<int>(this->blob_bottom_y_->cpu_data()[i]) == -1){
      W = 1.0/this->n2;
      M = -1 * c;
    }
    loss += W * log(exp(-alpha*(this->blob_bottom_data_->cpu_data()[i]-beta)*M) + 1);
  }
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-6);
}


TYPED_TEST(BinomialDevianceLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BinomialDevianceLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);

  checker.CheckGradientExhaustive(&layer, (this->blob_bottom_vec_),
      (this->blob_top_vec_), 0);
}

}  // namespace caffe
