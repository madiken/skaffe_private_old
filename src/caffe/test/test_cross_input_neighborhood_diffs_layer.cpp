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
class CrossInputNeighborhoodDiffsLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CrossInputNeighborhoodDiffsLayerTest()
      : im_blob0(new Blob<Dtype>(1, 10, 10, 1)),       
        im_blob1(new Blob<Dtype>(1, 10, 10, 1)),       
        im_unfolded_blob0(new Blob<Dtype>(1, 10*5, 10*5, 1)),
        im_unfolded_blob1(new Blob<Dtype>(1, 10*5, 10*5, 1)){
    
    FillerParameter filler_param;
    filler_param.set_mean(0.0);
    filler_param.set_std(0.1); 
    GaussianFiller<Dtype> filler(filler_param); 

    filler.Fill(this->im_blob0);
    filler.Fill(this->im_blob1);
    blob_bottom_vec_.push_back(im_blob0);
    blob_bottom_vec_.push_back(im_blob1);
    blob_top_vec_.push_back(im_unfolded_blob0);
    blob_top_vec_.push_back(im_unfolded_blob1);
  }

  virtual ~CrossInputNeighborhoodDiffsLayerTest() {
    delete im_blob0;
    delete im_blob1;
    delete im_unfolded_blob0;
    delete im_unfolded_blob1;
  }
 
  Blob<Dtype>* const im_blob0;
  Blob<Dtype>* const im_unfolded_blob0;
  Blob<Dtype>* const im_blob1;
  Blob<Dtype>* const im_unfolded_blob1;  
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

};

TYPED_TEST_CASE(CrossInputNeighborhoodDiffsLayerTest, TestDtypesAndDevices);

TYPED_TEST(CrossInputNeighborhoodDiffsLayerTest, TestForward) {
  std::cout << "TestForward!!!!" << std::endl;
  typedef typename TypeParam::Dtype Dtype; 
  LayerParameter layer_param;
  CrossInputNeighborhoodDiffsLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  std::cout << "TestForward!!!! finished" << std::endl;

}

/*
TYPED_TEST(CrossInputNeighborhoodDiffsLayerTest, TestGradient1) {
  std::cout << "TestGradient!!!!" << std::endl;
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  CrossInputNeighborhoodDiffsParameter* cross_input_neighborhood_diffs_param =
      layer_param.mutable_cross_input_neighborhood_diffs_param();
   
  cross_input_neighborhood_diffs_param->set_copy_padding(false);
 


  CrossInputNeighborhoodDiffsLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);

  checker.CheckGradientExhaustive(&layer, (this->blob_bottom_vec_),
      (this->blob_top_vec_));

  std::cout << "TestGradient!!!! finished" << std::endl;
}*/

TYPED_TEST(CrossInputNeighborhoodDiffsLayerTest, TestGradient2) {
  std::cout << "TestGradient!!!!" << std::endl;
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;


  CrossInputNeighborhoodDiffsParameter* cross_input_neighborhood_diffs_param =
      layer_param.mutable_cross_input_neighborhood_diffs_param();
  cross_input_neighborhood_diffs_param->set_copy_padding(true);
 


  CrossInputNeighborhoodDiffsLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);

  checker.CheckGradientExhaustive(&layer, (this->blob_bottom_vec_),
      (this->blob_top_vec_));

  std::cout << "TestGradient!!!! finished" << std::endl;
}


}  // namespace caffe
