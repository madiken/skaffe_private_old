#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/common_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SqrtNormalizationLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SqrtNormalizationLayerTest()
      : blob_bottom_data0(new Blob<Dtype>(3, 3, 3, 3)),
        blob_top(new Blob<Dtype>(3, 3, 3, 3)) {
    
    FillerParameter filler_param;
    filler_param.set_mean(0);
    filler_param.set_std(0.0001); 
    GaussianFiller<Dtype> filler(filler_param); 
    filler.Fill(this->blob_bottom_data0);

    blob_bottom_vec_.push_back(blob_bottom_data0);
    blob_top_vec_.push_back(blob_top);
  }
  virtual ~SqrtNormalizationLayerTest() {
    delete blob_bottom_data0;
    delete blob_top;
  }
 
  Blob<Dtype>* const blob_bottom_data0;
  Blob<Dtype>* const blob_top;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SqrtNormalizationLayerTest, TestDtypesAndDevices);

TYPED_TEST(SqrtNormalizationLayerTest, TestForward) {

  typedef typename TypeParam::Dtype Dtype; 
  LayerParameter layer_param;
  SqrtNormalizationLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
//  for (int i = 0; i < this->blob_bottom_vec_[0]->count(); ++i) {
//    EXPECT_NEAR((this->blob_bottom_vec_[0]->cpu_data()[i] > 0? 1: -1)* sqrt(fabs(this->blob_bottom_vec_[0]->cpu_data()[i])) , this->blob_top_vec_//[0]->cpu_data()[i] , 1e-6);
    
//  }
}


TYPED_TEST(SqrtNormalizationLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SqrtNormalizationLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-6, 1e-2, 1701);
  
  checker.CheckGradientExhaustive(&layer, (this->blob_bottom_vec_),
      (this->blob_top_vec_), 0);
  vector<bool> propagate_down;
  propagate_down.push_back(true);

}

}  // namespace caffe
