#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>


#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/facial_point_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

using namespace std;
using namespace cv;
#define __DEBUG
namespace caffe {

template <typename Dtype>
FacialPointDataLayer<Dtype>::~FacialPointDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void FacialPointDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.facial_point_data_param().new_height();
  const int new_width  = this->layer_param_.facial_point_data_param().new_width();
  const bool is_color  = this->layer_param_.facial_point_data_param().is_color();
  string root_folder = this->layer_param_.facial_point_data_param().root_folder();
  const int point_num  = this->layer_param_.facial_point_data_param().point_num();

  CHECK((new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";


  const string& source = this->layer_param_.facial_point_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label = 0;;
  while (infile >> filename ) {
    lines_.push_back(std::make_pair(filename, label));
  }


  // randomly shuffle data
  if (this->layer_param_.facial_point_data_param().shuffle()) {
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }


  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;


// Read an image, and use it to initialize the top blob.
  string  image_path = root_folder + lines_[lines_id_].first + string(".jpg");
  cv::Mat temp_image = imread(image_path.c_str());
  if(temp_image.empty())
  {
      image_path = root_folder + lines_[lines_id_].first + string(".png");
      temp_image = imread(image_path.c_str());
  }
  CHECK(!temp_image.empty()) << "Could not load " << image_path;

  /* load facial points */
  string label_path = root_folder + lines_[lines_id_].first + string(".rct");
  ifstream fin(label_path.c_str());
  CHECK(fin.is_open());
  float x1,x2,y1,y2;
  Rect face_roi;
  fin>>x1>>y1>>x2>>y2;
  fin.close();

  face_roi.x = x1;
  face_roi.y = y1;
  face_roi.width  = x2;
  face_roi.height = y2;

  face_roi.x = face_roi.x>0 ? face_roi.x : 0;
  face_roi.y = face_roi.y>0 ? face_roi.y : 0;
  if(temp_image.cols<(face_roi.x+face_roi.width))
      face_roi.width = temp_image.cols - face_roi.x;

  if(temp_image.rows<(face_roi.y+face_roi.height))
    face_roi.height = temp_image.rows-face_roi.y;


  //cout<<face_roi<<endl;
  //cout<<temp_image.size()<<endl;

  Mat image = temp_image(face_roi);
  cv::Mat cv_img;
  cv::resize(image,temp_image,Size(new_width,new_height));
  if (is_color)
  {
    cv_img = temp_image;
  }else{
    cvtColor(temp_image,cv_img,CV_BGR2GRAY);
  }


  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.facial_point_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;


  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // label
  vector<int> label_shape;
  label_shape.push_back(batch_size);
  label_shape.push_back(point_num*2);

  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void FacialPointDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void FacialPointDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  FacialPointDataParameter facial_point_data_param = this->layer_param_.facial_point_data_param();
  const int batch_size = facial_point_data_param.batch_size();
  const int new_height = facial_point_data_param.new_height();
  const int new_width = facial_point_data_param.new_width();
  const bool is_color = facial_point_data_param.is_color();
  string root_folder = facial_point_data_param.root_folder();
  const int point_num  = facial_point_data_param.point_num();
  //const bool is_flip = facial_point_data_param.is_flip();
 
  

   // Read an image, and use it to initialize the top blob.
  string  image_path = root_folder + lines_[lines_id_].first + string(".jpg");
  cv::Mat temp_image = imread(image_path.c_str());
  if(temp_image.empty())
  {
      image_path = root_folder + lines_[lines_id_].first + string(".png");
      temp_image = imread(image_path.c_str());
  }
  CHECK(!temp_image.empty()) << "Could not load " << image_path;

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  string label_path = root_folder + lines_[lines_id_].first + string(".rct");
  //LOG(INFO)<<label_path;
  ifstream fin(label_path.c_str());
  float x1,x2,y1,y2;
  Rect face_roi;
  fin>>x1>>y1>>x2>>y2;
  fin.close();

  face_roi.x = x1;
  face_roi.y = y1;
  face_roi.width  = x2;
  face_roi.height = y2;

  face_roi.x = face_roi.x>0 ? face_roi.x : 0;
  face_roi.y = face_roi.y>0 ? face_roi.y : 0;
  if(temp_image.cols<(face_roi.x+face_roi.width))
      face_roi.width = temp_image.cols - face_roi.x;

  if(temp_image.rows<(face_roi.y+face_roi.height))
    face_roi.height = temp_image.rows-face_roi.y;

  Mat image = temp_image(face_roi);
  cv::Mat cv_img;
  cv::resize(image,temp_image,Size(new_width,new_height));
  if (is_color)
  {
    cv_img = temp_image;
  }else{
    cvtColor(temp_image,cv_img,CV_BGR2GRAY);
  }

  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  vector<int> label_shape;
  label_shape.push_back(batch_size);
  label_shape.push_back(point_num*2);
  batch->label_.Reshape(label_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);


    // Read an image, and use it to initialize the top blob.
    string  image_path = root_folder + lines_[lines_id_].first + string(".jpg");
    //LOG(INFO)<<image_path;
    cv::Mat temp_image = imread(image_path.c_str());
    if(temp_image.empty())
    {
        image_path = root_folder + lines_[lines_id_].first + string(".png");
        temp_image = imread(image_path.c_str());
    }
    CHECK(!temp_image.empty()) << "Could not load " << image_path;
    //cout<<image_path<<endl;

    // read face rect and landmark infor
    string label_path = root_folder + lines_[lines_id_].first + string(".pts");

    ifstream fin(label_path.c_str());
    CHECK(fin.is_open());
    float x1,y1,c_x,c_y,c_wh,theta, u_rnd, g_rnd;
    cv::Rect est_face_roi, face_roi;
    /*
    cv::Rect init_face_roi;
    fin>>x1>>y1>>x2>>y2;
    init_face_roi.x = x1>0 ? (int)x1 : 0;
    init_face_roi.y = y1>0 ? (int)y1 : 0;
    init_face_roi.width  = temp_image.cols > x2 ? (int)(x2-init_face_roi.x) : (int)(temp_image.cols-init_face_roi.x);
    init_face_roi.height = temp_image.rows > y2 ? (int)(y2-init_face_roi.y) : (int)(temp_image.rows-init_face_roi.y);*/


    float LT_x=FLT_MAX, LT_y=FLT_MAX, RB_x=0, RB_y=0;

    std::vector<cv::Point_<float> > vPoint;

    for (int i = 0; i < point_num; ++i)
   {
      fin>>x1>>y1;
      vPoint.push_back(cv::Point_<float>(x1,y1));
      if(LT_x>x1)
        LT_x = x1;
      if(LT_y>y1)
        LT_y = y1;
      if(RB_x<x1)
        RB_x = x1;
      if(RB_y<y1)
        RB_y = y1;
      
    }
    fin.close();
    //cout<< LT_x <<' ' <<LT_y <<' '<<RB_x<<' '<<RB_y<<endl;
    est_face_roi.x = LT_x;
    est_face_roi.y = LT_y;
    est_face_roi.width = RB_x -LT_x+1;
    est_face_roi.height = RB_y - LT_y+1;
    est_face_roi.width = max(est_face_roi.width,est_face_roi.height);
    est_face_roi.height = est_face_roi.width;


    //******************** begin - perturb face roi *********************************
    caffe_rng_uniform<float>(1, 0, 1, &u_rnd);
    theta = u_rnd * 2 * M_PI;
    c_x = 0.5f*(LT_x + RB_x);
    c_y = 0.5f*(LT_y + RB_y);
    //cout<<"center: "<<c_x<<' '<<c_y<<endl;
  
    caffe_rng_gaussian<float>(1, 0.0f, 1.0f, &g_rnd);
    g_rnd = g_rnd<-2 ? -2:g_rnd;
    g_rnd = g_rnd>2 ? 2:g_rnd;
    //c_x = c_x + cos(theta)*(g_rnd * 0.05f + 0.15f)*est_face_roi.width;
    //c_y = c_y + sin(theta)*(g_rnd * 0.05f + 0.15f)*est_face_roi.width;

    // modified by Xiaohu Shao, 20160724
    c_x = c_x + cos(theta)*(g_rnd * 0.05f + 0.00f)*est_face_roi.width;
    c_y = c_y + sin(theta)*(g_rnd * 0.05f + 0.00f)*est_face_roi.width;

    caffe_rng_gaussian<float>(1, 0.0f, 1.0f, &g_rnd);
    g_rnd = g_rnd<-2 ? -2:g_rnd;
    g_rnd = g_rnd>2 ? 2:g_rnd;
    //cout<<"g_rnd :"<< g_rnd<<endl;

    // modified by Xiaohu Shao, 20160724
    //c_wh = (g_rnd * 0.07f +1.2f) * est_face_roi.width;
    c_wh = (g_rnd * 0.07f +1.3f) * est_face_roi.width;

    face_roi.x = c_x - 0.5f * c_wh;
    face_roi.y = c_y - 0.5f * c_wh;
    face_roi.width =  c_wh;
    face_roi.height = c_wh;
    //face_roi.height = init_face_roi.height + rand() % ((int)(est_face_roi.height*0.5f)) -  est_face_roi.height*0.25f; //rand()//(double)RAND_MAX
    //cout<<c_x<<' '<<c_y<<' '<<face_roi.x<<' '<<face_roi.y<<' '<<face_roi.width<<' '<<face_roi.height<<endl;


    face_roi.x = face_roi.x<LT_x ? face_roi.x : LT_x;
    face_roi.y = face_roi.y<LT_y ? face_roi.y : LT_y;
    face_roi.x = face_roi.x>0 ? face_roi.x : 0;
    face_roi.y = face_roi.y>0 ? face_roi.y : 0;
    

    if(RB_x>(face_roi.x+face_roi.width))
      face_roi.width = RB_x - face_roi.x;
    if(RB_y>(face_roi.y+face_roi.height))
      face_roi.height = RB_y-face_roi.y;

    face_roi.width = max(face_roi.width,face_roi.height);
    face_roi.height = face_roi.width;

    if(temp_image.cols<(face_roi.x+face_roi.width))
      face_roi.width = temp_image.cols - face_roi.x;

    if(temp_image.rows<(face_roi.y+face_roi.height))
      face_roi.height = temp_image.rows-face_roi.y;

    //***************************** end - perturb face roi **********************************
    

    // rotate 
    //cout<<image_path<<endl;
    //cout<<face_roi.x<<' '<<face_roi.y<<' '<<face_roi.width<<' '<<face_roi.height<<endl;
    //cout<<temp_image.cols<<' '<<temp_image.rows<<endl;

  /* if(1)
    {
      //cv::flip(temp_image, temp_image, 1);
      for (int k = 0; k < point_num; ++k)
       {
        
        cv::circle(temp_image,vPoint[k],2,Scalar(255,0,0),2);
       }
      cv::rectangle(temp_image,face_roi,Scalar(0,255,0));
      imshow("src_image",temp_image);
      waitKey();
    }*/


    //****************************************************

    
    //cout<<face_roi<<endl;
    //cout<<temp_image.size()<<endl;
    //cout<< RB_x <<' '<<RB_y<<endl;

    //face_roi = init_face_roi;

    Mat image = temp_image(face_roi);

    cv::Mat cv_img;
    cv::resize(image,temp_image,Size(new_width,new_height));
    if (is_color)
    {
      cv_img = temp_image;
    }else{
      cvtColor(temp_image,cv_img,CV_BGR2GRAY);
    }

    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();


    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();
    //LOG(INFO)<<"***********************************";
    for (int i = 0; i < point_num; ++i){

      prefetch_label[item_id*point_num*2 + i] = 2 * (vPoint[i].x - face_roi.x-face_roi.width/2)/face_roi.width;
      prefetch_label[item_id*point_num*2 + i + point_num] = 2 * (vPoint[i].y - face_roi.y-face_roi.height/2)/face_roi.height;

      //cout<<prefetch_label[item_id*point_num*2 + 2*i]<<' '<<prefetch_label[item_id*point_num*2 + 2*i+1]<<endl;
    //  LOG(INFO)<<prefetch_label[item_id*point_num*2 + 2*i]<<" "<<prefetch_label[item_id*point_num*2 + 2*i+1];
    }


    
    

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.facial_point_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(FacialPointDataLayer);
REGISTER_LAYER_CLASS(FacialPointData);

}  // namespace caffe
#endif  // USE_OPENCV
