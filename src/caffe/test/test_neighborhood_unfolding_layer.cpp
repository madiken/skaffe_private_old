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
class NeighborhoodUnfoldingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  NeighborhoodUnfoldingLayerTest()
      : im_blob(new Blob<Dtype>(2, 10, 10, 1)),       
        im_unfolded_blob(new Blob<Dtype>(2, 10*5, 10*5, 1)){
    
    FillerParameter filler_param;
    filler_param.set_mean(0.0);
    filler_param.set_std(0.1); 
    GaussianFiller<Dtype> filler(filler_param); 

    filler.Fill(this->im_blob);
    blob_bottom_vec_.push_back(im_blob);
    blob_top_vec_.push_back(im_unfolded_blob);
  }

  virtual ~NeighborhoodUnfoldingLayerTest() {
    delete im_blob;
    delete im_unfolded_blob;
  }
 
  Blob<Dtype>* const im_blob;
  Blob<Dtype>* const im_unfolded_blob;
  
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

};

TYPED_TEST_CASE(NeighborhoodUnfoldingLayerTest, TestDtypesAndDevices);

TYPED_TEST(NeighborhoodUnfoldingLayerTest, TestForward) {
  std::cout << "TestForward!!!!" << std::endl;
  typedef typename TypeParam::Dtype Dtype; 
  LayerParameter layer_param;
  NeighborhoodUnfoldingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  std::cout << "TestForward!!!! finished" << std::endl;

}


TYPED_TEST(NeighborhoodUnfoldingLayerTest, TestGradient1) {
  std::cout << "TestGradient!!!!" << std::endl;
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;

  NeighborhoodUnfoldingParameter* neighborhood_unfolding_param =
      layer_param.mutable_neighborhood_unfolding_param();
   
  neighborhood_unfolding_param->set_copy_padding(false);
  neighborhood_unfolding_param->set_self_copying(false);


  NeighborhoodUnfoldingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);

  checker.CheckGradientExhaustive(&layer, (this->blob_bottom_vec_),
      (this->blob_top_vec_), 0);

  std::cout << "TestGradient!!!! finished" << std::endl;
}

TYPED_TEST(NeighborhoodUnfoldingLayerTest, TestGradient2) {
  std::cout << "TestGradient!!!!" << std::endl;
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;


  NeighborhoodUnfoldingParameter* neighborhood_unfolding_param =
      layer_param.mutable_neighborhood_unfolding_param();
  neighborhood_unfolding_param->set_copy_padding(true);
  neighborhood_unfolding_param->set_self_copying(false);


  NeighborhoodUnfoldingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);

  checker.CheckGradientExhaustive(&layer, (this->blob_bottom_vec_),
      (this->blob_top_vec_), 0);

  std::cout << "TestGradient!!!! finished" << std::endl;
}

TYPED_TEST(NeighborhoodUnfoldingLayerTest, TestGradient3) {
  std::cout << "TestGradient!!!!" << std::endl;
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;


  NeighborhoodUnfoldingParameter* neighborhood_unfolding_param =
      layer_param.mutable_neighborhood_unfolding_param();
  neighborhood_unfolding_param->set_copy_padding(false);
  neighborhood_unfolding_param->set_self_copying(true);


  NeighborhoodUnfoldingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);

  checker.CheckGradientExhaustive(&layer, (this->blob_bottom_vec_),
      (this->blob_top_vec_), 0);

  std::cout << "TestGradient!!!! finished" << std::endl;
}


TYPED_TEST(NeighborhoodUnfoldingLayerTest, TestGradient4) {
  std::cout << "TestGradient!!!!" << std::endl;
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;


  NeighborhoodUnfoldingParameter* neighborhood_unfolding_param =
      layer_param.mutable_neighborhood_unfolding_param();
  neighborhood_unfolding_param->set_copy_padding(true);
  neighborhood_unfolding_param->set_self_copying(true);


  NeighborhoodUnfoldingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);

  checker.CheckGradientExhaustive(&layer, (this->blob_bottom_vec_),
      (this->blob_top_vec_), 0);

  std::cout << "TestGradient!!!! finished" << std::endl;
}
}  // namespace caffe
