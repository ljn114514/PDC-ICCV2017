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
#include "caffe/layers/aflw_facial_point_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

//#define DEBUGINFO 

using namespace std;
using namespace cv;
#define __DEBUG
namespace caffe {

template <typename Dtype> 
static void DeleteVector(vector<Dtype> &v)
{
    vector<Dtype> temp;
    v.swap(temp);
}

/* Load facial points and estimate and perturb face rectangle*/
static void GetRectandPts(string fileName, const int imgWidth, const int imgHeight, const int ptsNum,
                          cv::Rect &faceRect, vector<cv::Point_<float> > &pts)
{
  ifstream fin(fileName.c_str());
  CHECK(fin.is_open());
  float x1, y1, v1;
  float LT_x=FLT_MAX, LT_y=FLT_MAX, RB_x=0, RB_y=0;
  cv::Point_<float> center;
  pts.clear();
  for (int i = 0; i < ptsNum; ++i)
  {
    fin>>x1>>y1>>v1;
    pts.push_back(cv::Point_<float>(x1,y1));
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

  faceRect.x = LT_x;
  faceRect.y = LT_y;
  faceRect.width  = RB_x - LT_x + 1;
  faceRect.height = RB_y - LT_y + 1;

  faceRect.x = faceRect.x>0 ? faceRect.x : 0;
  faceRect.y = faceRect.y>0 ? faceRect.y : 0;
  if(imgWidth<(faceRect.x + faceRect.width))
      faceRect.width = imgWidth - faceRect.x;
  if(imgHeight<(faceRect.y + faceRect.height))
    faceRect.height = imgHeight - faceRect.y;
}

// perturb face roi 
static void PerturbFaceRect(const int imgWidth, const int imgHeight, 
                            float xyMean, float xyStd, float whMean, float whStd, 
                            vector<cv::Point_<float> > &pts, cv::Rect &faceRect)
{
  cv::Point_<float> center;
  center.x = faceRect.x + faceRect.width/2;
  center.y = faceRect.y + faceRect.height/2;

  float c_x, c_y, theta, u_rnd, g_rnd;
  caffe_rng_uniform<float>(1, 0, 1, &u_rnd);
  theta = u_rnd * 2 * M_PI;
  c_x = center.x;
  c_y = center.y;

  caffe_rng_gaussian<float>(1, 0.0f, 1.0f, &g_rnd);
  g_rnd = g_rnd<-2 ? -2:g_rnd;
  g_rnd = g_rnd>2 ? 2:g_rnd;
  c_x = c_x + cos(theta)*(g_rnd * xyStd + xyMean)*faceRect.width;
  c_y = c_y + sin(theta)*(g_rnd * xyStd + xyMean)*faceRect.height;

  caffe_rng_gaussian<float>(1, 0.0f, 1.0f, &g_rnd);
  g_rnd = g_rnd<-2 ? -2:g_rnd;
  g_rnd = g_rnd>2 ? 2:g_rnd;

  center.x = c_x;
  center.y = c_y;
  faceRect.width = (g_rnd * whStd + whMean) * faceRect.width;
  faceRect.height = (g_rnd * whStd + whMean) * faceRect.height;
  faceRect.x = center.x - faceRect.width/2;
  faceRect.y = center.y - faceRect.height/2;

  // make sure face rectangle is not out of the whole image
  faceRect.x = faceRect.x>0 ? faceRect.x : 0;
  faceRect.y = faceRect.y>0 ? faceRect.y : 0;
  if(imgWidth < (faceRect.x + faceRect.width))
      faceRect.width = imgWidth - faceRect.x;
  if(imgHeight < (faceRect.y + faceRect.height))
    faceRect.height = imgHeight - faceRect.y;
  
}

static void LoadFaceRect(string fileName, 
                         int imgWidth, 
                         int imgHeight,
                         cv::Rect &faceRect)
{
    ifstream fin(fileName.c_str());
    CHECK(fin.is_open());

    float x1, y1, x2, y2;

    fin >> x1 >> y1 >> x2 >> y2;
    fin.close();

    faceRect.x =  (int) (x1 >= 0 ? x1:0);
    faceRect.y =  (int) (y1 >= 0 ? y1:0);
    faceRect.width = (int) min(x2 - (float)faceRect.x, (float)(imgWidth - faceRect.x));
    faceRect.height = (int) min(y2 - (float)faceRect.y, (float)(imgHeight - faceRect.y));
}

static void ExtendFaceRect(const int imgWidth, const int imgHeight, float ext_scale, cv::Rect &faceRect)
{
    cv::Rect estFaceRect;
    float c_x, c_y, c_w, c_h;
    estFaceRect.x = faceRect.x;
    estFaceRect.y = faceRect.y;
    estFaceRect.width = faceRect.width;
    estFaceRect.height = faceRect.height;

    c_x = estFaceRect.x + estFaceRect.width/2;
    c_y = estFaceRect.y + estFaceRect.height/2;

    c_w = (int)(ext_scale * estFaceRect.width);
	  c_h = (int)(ext_scale * estFaceRect.height);

    faceRect.x = c_x - c_w / 2;
    faceRect.y = c_y - c_h / 2;
    faceRect.width =  c_w;
    faceRect.height = c_h;

    faceRect.x = faceRect.x>0 ? faceRect.x : 0;
    faceRect.y = faceRect.y>0 ? faceRect.y : 0;

    if(imgWidth < (faceRect.x + faceRect.width))
      faceRect.width = imgWidth - 1 - faceRect.x;

    if(imgHeight<(faceRect.y+faceRect.height))
      faceRect.height = imgHeight - 1 - faceRect.y;
}

template <typename Dtype>
AFLWFacialPointDataLayer<Dtype>::~AFLWFacialPointDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void AFLWFacialPointDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.aflw_facial_point_data_param().new_height();
  const int new_width  = this->layer_param_.aflw_facial_point_data_param().new_width();
  const bool is_color  = this->layer_param_.aflw_facial_point_data_param().is_color();
  string root_folder = this->layer_param_.aflw_facial_point_data_param().root_folder();
  const int point_num  = this->layer_param_.aflw_facial_point_data_param().point_num();
  const bool use_face_rect = this->layer_param_.aflw_facial_point_data_param().use_face_rect();
  const float xy_mean = this->layer_param_.aflw_facial_point_data_param().xy_mean();
  const float xy_std = this->layer_param_.aflw_facial_point_data_param().xy_std();
  const float wh_mean = this->layer_param_.aflw_facial_point_data_param().wh_mean();
  const float wh_std = this->layer_param_.aflw_facial_point_data_param().wh_std();

  CHECK((new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  const string& source = this->layer_param_.aflw_facial_point_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label = 0;;
  while (infile >> filename ) {
    lines_.push_back(std::make_pair(filename, label));
  }

  // randomly shuffle data
  if (this->layer_param_.aflw_facial_point_data_param().shuffle()) {
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
  string label_path = root_folder + lines_[lines_id_].first + string(".pts");
  Rect face_roi;
  std::vector<cv::Point_<float> > pts;

  GetRectandPts(label_path, temp_image.cols, temp_image.rows, point_num, 
                face_roi, pts);
  
  // load face rectangle instead of estimating one.
  if (use_face_rect)
  {
      string rect_path = root_folder + lines_[lines_id_].first + string(".rct");
      LoadFaceRect(rect_path, temp_image.cols, temp_image.rows, face_roi);
  }

  // perturb face retangles
  PerturbFaceRect(temp_image.cols, temp_image.rows, 
                  xy_mean, xy_std, wh_mean, wh_std, 
                  pts, face_roi);
  DeleteVector(pts);

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
  const int batch_size = this->layer_param_.aflw_facial_point_data_param().batch_size();
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

  DeleteVector(label_shape);
}

template <typename Dtype>
void AFLWFacialPointDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void AFLWFacialPointDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  AFLWFacialPointDataParameter facial_point_data_param = this->layer_param_.aflw_facial_point_data_param();
  const int batch_size = facial_point_data_param.batch_size();
  const int new_height = facial_point_data_param.new_height();
  const int new_width = facial_point_data_param.new_width();
  const bool is_color = facial_point_data_param.is_color();
  string root_folder = facial_point_data_param.root_folder();
  const int point_num  = facial_point_data_param.point_num();
  //const bool is_flip = facial_point_data_param.is_flip();
  const bool use_face_rect = facial_point_data_param.use_face_rect();
  const float ext_scale = facial_point_data_param.ext_scale();
  const float xy_mean = facial_point_data_param.xy_mean();
  const float xy_std = facial_point_data_param.xy_std();
  const float wh_mean = facial_point_data_param.wh_mean();
  const float wh_std = facial_point_data_param.wh_std();

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

    // read landmark information
    string label_path = root_folder + lines_[lines_id_].first + string(".pts");
    Rect face_roi;
    std::vector<cv::Point_<float> > vPoint;

    GetRectandPts(label_path, temp_image.cols, temp_image.rows, point_num, 
                  face_roi, vPoint);

    // load face rectangle instead of estimating one.
    if (use_face_rect)
    {
        string rect_path = root_folder + lines_[lines_id_].first + string(".rct");
        LoadFaceRect(rect_path, temp_image.cols, temp_image.rows, face_roi);
    }

    // perturb face retangles
    PerturbFaceRect(temp_image.cols, temp_image.rows, 
                    xy_mean, xy_std, wh_mean, wh_std, 
                    vPoint, face_roi);

    // extend face rectangle to make sure that all landmarks is in face roi
    ExtendFaceRect(temp_image.cols, temp_image.rows, ext_scale, face_roi);

    Mat image = temp_image(face_roi).clone();

    #ifdef DEBUGINFO
      cout<<image_path<<endl;
      cout<<face_roi.x<<' '<<face_roi.y<<' '<<face_roi.width<<' '<<face_roi.height<<endl;
      cout<<temp_image.cols<<' '<<temp_image.rows<<endl;

      if(1)
      {
        for (int k = 0; k < point_num; ++k)
         {
            char tmp[255];
            snprintf(tmp, sizeof(tmp), "%d", k+1);
            string str(tmp);
            cv::circle(temp_image,cvPoint((int)vPoint[k].x, (int)vPoint[k].y),2,Scalar(255,0,0),2);
            cv::putText(temp_image, str, 
                          cvPoint((int)vPoint[k].x, (int)vPoint[k].y),
                          cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 
                          cvScalar(0, 0, 255, 0));
         }
        cv::rectangle(temp_image,face_roi,Scalar(0,255,0));
        imshow("src_image",temp_image);
        if (waitKey() == 27)
        {
          break;
        }
      }
    #endif

    cv::Mat cv_img, tmp_img;
    cv::resize(image,tmp_img,Size(new_width,new_height));
    if (is_color)
    {
      cv_img = tmp_img;
    }else{
      cvtColor(tmp_img,cv_img,CV_BGR2GRAY);
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
    }

    DeleteVector(vPoint);

    #ifdef DEBUGINFO
    if(1)
    {
      for (int k = 0; k < point_num; ++k)
       {
         int idx; 
         cv::Point pt;
         idx = item_id*2*point_num + k;
         pt.x = (int)((prefetch_label[idx] + 1) * new_width / 2);
         idx = item_id*2*point_num + point_num + k;
         pt.y = (int)((prefetch_label[idx] + 1) * new_height / 2);

         cout << " pt: (" << pt.x << ", " << pt.y << ")" << endl; 
         char tmp[255];
         snprintf(tmp, sizeof(tmp), "%d", k+1);
         string str(tmp);
         cv::circle(tmp_img,pt,2,Scalar(255,0,0),2);
         cv::putText(tmp_img, str, 
                     cvPoint(pt.x - 2, pt.y - 2),
                     cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 
                     cvScalar(0, 0, 255, 0));
       }
      imshow("dst_image",tmp_img);
      if (waitKey() == 27)
      {
        break;
      }
    }
    #endif

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.aflw_facial_point_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(AFLWFacialPointDataLayer);
REGISTER_LAYER_CLASS(AFLWFacialPointData);

}  // namespace caffe
#endif  // USE_OPENCV
