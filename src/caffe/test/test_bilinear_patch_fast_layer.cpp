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
class BilinearPatchFastLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BilinearPatchFastLayerTest()
      : blob_bottom_data0(new Blob<Dtype>(3, 3, 15, 15)),
        blob_bottom_data1(new Blob<Dtype>(3, 2, 15, 15)),
        blob_bottom_data2(new Blob<Dtype>(3, 6, 15, 15)),
        blob_top(new Blob<Dtype>(3, 6*6, 1 , 1)) {
    
    FillerParameter filler_param;
    filler_param.set_value(1.0);
    filler_param.set_std(1.0); 
    GaussianFiller<Dtype> filler(filler_param); 
    filler.Fill(this->blob_bottom_data0);
    filler.Fill(this->blob_bottom_data1);
    filler.Fill(this->blob_bottom_data2);

    blob_bottom_vec_.push_back(blob_bottom_data0);
    blob_bottom_vec_.push_back(blob_bottom_data1);
    blob_bottom_vec_.push_back(blob_bottom_data2);
    blob_top_vec_.push_back(blob_top);
  }
  virtual ~BilinearPatchFastLayerTest() {
    delete blob_bottom_data0;
    delete blob_bottom_data1;
    delete blob_bottom_data2;
    delete blob_top;
  }
 
  Blob<Dtype>* const blob_bottom_data0;
  Blob<Dtype>* const blob_bottom_data1;
  Blob<Dtype>* const blob_bottom_data2;

  Blob<Dtype>* const blob_top;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BilinearPatchFastLayerTest, TestDtypesAndDevices);

TYPED_TEST(BilinearPatchFastLayerTest, TestForward) {

  typedef typename TypeParam::Dtype Dtype; 
  LayerParameter layer_param;
  BilinearPatchFastLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
}


TYPED_TEST(BilinearPatchFastLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BilinearPatchFastLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);

  checker.CheckGradientExhaustive(&layer, (this->blob_bottom_vec_),
      (this->blob_top_vec_), 0);
  checker.CheckGradientExhaustive(&layer, (this->blob_bottom_vec_),
      (this->blob_top_vec_), 1);
//checker.CheckGradientExhaustive(&layer, (this->blob_bottom_vec_),
  //    (this->blob_top_vec_));
}

}  // namespace caffe
