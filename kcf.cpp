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
#include <ctime>
#include <unistd.h>

using namespace std;
using namespace cv;

int main( int argc, char** argv ){
  // show help
  if(argc!=3){
    cout<<
//      " Usage:  argv[1] camara index \n"
      " Usage:  argv[1] video file argv[2] start frame index\n"
      << endl;
    return 0;
  }

  TrackerKCF::Params param;
  FileStorage fs;
  fs.open("kcf.yaml", FileStorage::READ);
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
  // set input video
  //int idx = atoi(argv[1]);
  //VideoCapture cap(idx);
  std::string video = argv[1];
  VideoCapture cap(video);

  if (!cap.isOpened()) {
      cout<<"can not open the camera"<<endl;
      cin.get();
      exit(1);
  }
  if(!cap.set(CAP_PROP_POS_FRAMES,int(atoi(argv[2])))){
    cout<< "fail to set video start frame index!" << endl;
    exit(1);
  }

  Mat frame;
  Mat show_frame;
  // get bounding box
  cap >> frame;
  cv::resize(frame,show_frame,cv::Size(1024,768));
  Rect2d roi= selectROI("tracker", show_frame, true, false);

  roi.x = roi.x / 1024.0 * frame.cols;
  roi.y = roi.y / 768.0 * frame.rows;
  roi.width   = roi.width   / 1024.0 * frame.cols;
  roi.height  = roi.height / 768.0 * frame.rows;

  //quit if ROI was not selected
  if(roi.width==0 || roi.height==0)
    return 0;
  // initialize the tracker
  int64 t1 = cv::getTickCount();
  tracker->init(frame, roi);
  int64 t2 = cv::getTickCount();
  int64 tick_counter = t2 - t1;
  int frame_idx = 1;
  // do the tracking
  printf("Start the tracking process, press ESC to quit.\n");

  clock_t timer;
  double  fps;
  String text;
  char buffer [128];
  bool pause = false;
  for ( ;; ){
    int res = waitKey(1);
    //quit on ESC button
    if(res==27)break;    
    if(pause==true){
      if(res==32){
        pause = false;
      } else {
        continue;
      }
    } else {
      if(res==32){
        pause = true;
        continue;
      }
    }

    // get frame from the video
    cap >> frame;
    double frameNO = cap.get(CAP_PROP_POS_FRAMES);
    // stop the program if no more images
    if(frame.rows==0 || frame.cols==0)
      break;

    // update the tracking result
    t1 = cv::getTickCount();
    timer=clock();
    bool isfound = tracker->update(frame, roi);
    t2 = cv::getTickCount();
    timer=clock()-timer;
    tick_counter += t2 - t1;
    fps=(double)CLOCKS_PER_SEC/(double)timer;
    frame_idx++;

    if(!isfound)
    {
        cout << "The target has been lost...\n";
        waitKey(0);
        return 0;
    }
    // draw the tracked object
    rectangle( frame, roi, Scalar( 255, 0, 0 ), 2, 1 );

    sprintf (buffer, "speed: %.0f fps frame index:%d", fps,int(frameNO));
    text = buffer;
    putText(frame, text, Point(20,20), FONT_HERSHEY_PLAIN, 2, Scalar(0,0,0),2);
    sprintf (buffer, "roi length: %d ", int(roi.width));
    text = buffer;
    putText(frame, text, Point(20,35), FONT_HERSHEY_PLAIN, 2, Scalar(0,0,0),2);    
    // show image with the tracked object
    
    cv::resize(frame,show_frame,cv::Size(1024,768));
    imshow("tracker",show_frame);

  }
    cout << "Elapsed sec: " << static_cast<double>(tick_counter) / cv::getTickFrequency() << endl;
    cout << "FPS: " << ((double)(frame_idx)) / (static_cast<double>(tick_counter) / cv::getTickFrequency()) << endl;
}
