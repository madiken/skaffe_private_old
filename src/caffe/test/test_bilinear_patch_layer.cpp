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
class BilinearPatchLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BilinearPatchLayerTest()
      : blob_bottom_data0(new Blob<Dtype>(3, 3, 9, 9)),
        blob_bottom_data1(new Blob<Dtype>(3, 2, 9, 9)),
     
        blob_top(new Blob<Dtype>(3, 6*9, 3, 3)) {
    
    FillerParameter filler_param;
    filler_param.set_mean(1.0);
    filler_param.set_std(0.1); 
    GaussianFiller<Dtype> filler(filler_param); 
    filler.Fill(this->blob_bottom_data0);
    filler.Fill(this->blob_bottom_data1);
    

    blob_bottom_vec_.push_back(blob_bottom_data0);
    blob_bottom_vec_.push_back(blob_bottom_data1);
   
    blob_top_vec_.push_back(blob_top);
  }
  virtual ~BilinearPatchLayerTest() {
    delete blob_bottom_data0;
    delete blob_bottom_data1;

    delete blob_top;
  }
 
  Blob<Dtype>* const blob_bottom_data0;
  Blob<Dtype>* const blob_bottom_data1;


  Blob<Dtype>* const blob_top;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BilinearPatchLayerTest, TestDtypesAndDevices);
/*
TYPED_TEST(BilinearPatchLayerTest, TestForward) {

  typedef typename TypeParam::Dtype Dtype; 
  LayerParameter layer_param;
  BilinearPatchLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
}
*/

TYPED_TEST(BilinearPatchLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BilinearPatchParameter* bilinear_patch_param =
      layer_param.mutable_bilinear_patch_param();
  bilinear_patch_param->set_patch_h(3);
  bilinear_patch_param->set_patch_w(3);
  BilinearPatchLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);

  checker.CheckGradientExhaustive(&layer, (this->blob_bottom_vec_),
      (this->blob_top_vec_), 0);
  checker.CheckGradientExhaustive(&layer, (this->blob_bottom_vec_),
      (this->blob_top_vec_), 1);

}

}  // namespace caffe
