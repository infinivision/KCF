/*----------------------------------------------
 * Usage:
 * example_tracking_kcf <video_name>
 *
 * example:
 * example_tracking_kcf Bolt/img/%04.jpg
 * example_tracking_kcf faceocc2.webm
 *--------------------------------------------------*/

#include <opencv2/core/utility.hpp>
#include "tracker.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>

using namespace std;
using namespace cv;

int main( int argc, char** argv ){
  // show help
  if(argc<4){
    cout<<
      " Usage:  argv[1] resetTerm argv[2] resetCount  argv[3] video file\n"
      << endl;
    return 0;
  }

  int resetTerm = atoi(argv[1]);
  int resetCount = atoi(argv[2]);
  TrackerKCF::Params param;
  FileStorage fs;
  fs.open("kcfBench.yaml", FileStorage::READ);
  param.read(fs.root());
  cout << "param.detect_thresh:       " << param.detect_thresh << endl;
  cout << "param.lambda:              " << param.lambda << endl;
  cout << "param.interp_factor:       " << param.interp_factor << endl;
  cout << "param.output_sigma_factor: " << param.output_sigma_factor << endl;
  cout << "param.compressed_size:     " << param.compressed_size << endl;
  cout << "param.pca_learning_rate:   " << param.pca_learning_rate << endl;
  cout << "param.template_len:        " << param.template_len << endl;
  cout << "param.pad_scale:           " << param.pad_scale << endl;
  // create the tracker
  Ptr<Tracker> tracker = TrackerKCF::create(param);
  Mat frame;
  std::string video = argv[3];
  VideoCapture cap(video);
  cap >> frame;
  Rect2d roi = Rect2d(100,100,200,200);
  //quit if ROI was not selected
  if(roi.width==0 || roi.height==0)
    return 0;
  int64 t1,t2,tick_counter=0;
  int frame_idx = 0;
  for(int i=0;i<resetCount;i++){
    // initialize the tracker
    t1 = cv::getTickCount();
    if(!tracker->reset(frame, roi)){
      cout<<"trakcer reset failed";
      exit(1);
    }
    t2 = cv::getTickCount();
    tick_counter += t2 - t1;
    frame_idx++;
    
    for ( ;; ){
      // update the tracking result
      cap >> frame;
      t1 = cv::getTickCount();
      bool isfound = tracker->update(frame, roi);
      t2 = cv::getTickCount();
      tick_counter += t2 - t1;
      frame_idx++;
      if(!isfound){
          cout << "The target has been lost...\n";
          return 0;
      }
      if (frame_idx % resetTerm == 0){
        break;
      }
    }
  }

  cout << "Elapsed sec: " << static_cast<double>(tick_counter) / cv::getTickFrequency() << endl;
  cout << "FPS: " << ((double)(frame_idx)) / (static_cast<double>(tick_counter) / cv::getTickFrequency()) << endl;
}
