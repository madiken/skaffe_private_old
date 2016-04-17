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
class BilinearLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BilinearLayerTest()
      : blob_bottom_data0(new Blob<Dtype>(3, 5, 9, 9)),
        blob_bottom_data1(new Blob<Dtype>(3, 3, 9, 9)),
        blob_top(new Blob<Dtype>(3, 15, 3, 3)),

        blob_bottom_data0_t(new Blob<Dtype>(3, 5, 9, 9)),
        blob_bottom_data1_t(new Blob<Dtype>(3, 3, 9, 9)),
        blob_top_t(new Blob<Dtype>(3, 15, 3, 3))
 {
    
    FillerParameter filler_param;
    filler_param.set_mean(0.0);
    filler_param.set_std(0.1); 
    GaussianFiller<Dtype> filler(filler_param); 
    filler.Fill(this->blob_bottom_data0);
    filler.Fill(this->blob_bottom_data1);

    blob_bottom_vec_.push_back(blob_bottom_data0);
    blob_bottom_vec_.push_back(blob_bottom_data1);
    blob_top_vec_.push_back(blob_top);


    for (int i = 0; i < this->blob_bottom_data0_t->count(); i++){
      this->blob_bottom_data0_t->mutable_cpu_data()[i] =  this->blob_bottom_data0->cpu_data()[i];
    }  
    for (int i = 0; i < this->blob_bottom_data1_t->count(); i++){
      this->blob_bottom_data1_t->mutable_cpu_data()[i] =  this->blob_bottom_data1->cpu_data()[i];
    }  
    blob_bottom_vec_t.push_back(blob_bottom_data0_t);
    blob_bottom_vec_t.push_back(blob_bottom_data1_t);
    blob_top_vec_t.push_back(blob_top_t);
  }
  virtual ~BilinearLayerTest() {
    delete blob_bottom_data0;
    delete blob_bottom_data1;
    delete blob_top;

    delete blob_bottom_data0_t;
    delete blob_bottom_data1_t;
    delete blob_top_t;
  }
 
  Blob<Dtype>* const blob_bottom_data0;
  Blob<Dtype>* const blob_bottom_data1;
  Blob<Dtype>* const blob_top;


  Blob<Dtype>* const blob_bottom_data0_t;
  Blob<Dtype>* const blob_bottom_data1_t;
  Blob<Dtype>* const blob_top_t;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  vector<Blob<Dtype>*> blob_bottom_vec_t;
  vector<Blob<Dtype>*> blob_top_vec_t;

};

TYPED_TEST_CASE(BilinearLayerTest, TestDtypesAndDevices);

TYPED_TEST(BilinearLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype; 
   LayerParameter layer_param;
 
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();

  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(3);

  BilinearLayer<Dtype> layer(layer_param); 
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);


  LayerParameter layer_param_t;
  BilinearPatchParameter* bilinear_patch_param =
      layer_param_t.mutable_bilinear_patch_param();
  bilinear_patch_param->set_patch_h(3);
  bilinear_patch_param->set_patch_w(3);
  
  BilinearPatchLayer<Dtype> layer_t(layer_param_t);
  layer_t.SetUp(this->blob_bottom_vec_t, this->blob_top_vec_t);
  layer_t.Forward(this->blob_bottom_vec_t, this->blob_top_vec_t);


  for (int i = 0; i < this->blob_top_vec_[0]->count(); i++){
      EXPECT_NEAR(this->blob_top_vec_[0]->cpu_data()[i], this->blob_top_vec_t[0]->cpu_data()[i], 1e-6);
  }  


}

/*
TYPED_TEST(BilinearLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
 
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();

  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(3);
  BilinearLayer<Dtype> layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);

  checker.CheckGradientExhaustive(&layer, (this->blob_bottom_vec_),
      (this->blob_top_vec_));
}
*/
}  // namespace caffe
