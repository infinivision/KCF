#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cstring>
#include <unistd.h>

using namespace std;
using namespace cv;
Mat frame,frame2;
int scale = 1;

void onMouse(int event, int x, int y, int flags, void* param)
{
    char text[100];
    Mat frame3;
    frame3 = frame2.clone();
    sprintf(text, "x=%d, y=%d", x*scale, y*scale);
    drawMarker(frame3, cv::Point(x, y),  cv::Scalar(0, 0, 255), MARKER_CROSS, 10, 1);
    putText(frame3, text, Point(20,35), FONT_HERSHEY_PLAIN, 1.25, Scalar(0,0,0),2);
    imshow("image",frame3);
}

int main( int argc, char** argv ){
    if(argc<2){
        cout << "please inpute image file path" << endl;
        exit(0);
    }
    VideoCapture cap(argv[1]);
    
    if(!cap.isOpened()) {
        cout << "camera open failed!" << endl;
    }
    
    cap >> frame;
    if(frame.cols > 0){
        if(frame.cols>1280) {
            scale = 2;
            resize(frame,frame2,frame.size()/scale);
        } else {
            frame2 = frame.clone();
        }
        
        namedWindow("image",cv::WINDOW_FULLSCREEN);
        setMouseCallback("image", onMouse, 0);
        imshow("image",frame2);
    }

    waitKey(0);
}
