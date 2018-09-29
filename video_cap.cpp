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

void onMouse(int event, int x, int y, int flags, void* param)
{
    char text[100];
    Mat frame3;
    frame3 = frame2.clone();
    sprintf(text, "x=%d, y=%d", x, y);
    drawMarker(frame3, cv::Point(x, y),  cv::Scalar(0, 0, 255), MARKER_CROSS, 10, 1);
    putText(frame3, text, Point(20,35), FONT_HERSHEY_PLAIN, 1.25, Scalar(0,0,0),2);
    imshow("image",frame3);
}

int main( int argc, char** argv ){
    VideoCapture cap("rtsp://admin:mcd12345678@192.168.150.242:554//Streaming/Channels/1");
    //VideoCapture cap("rtsp://admin:mcd12345678@192.168.150.244:554//Streaming/Channels/1");
    
    if(!cap.isOpened()) {
        cout << "camera open failed!" << endl;
    }
    
    string file;
    if(argc==2){
        file = string(argv[1]) + string(".jpg");
    }

    cap >> frame;
    if(frame.cols > 0){
        if(frame.cols>1280) {
            resize(frame,frame2,frame.size()/2);
        } else {
            frame2 = frame.clone();
        }
        
        namedWindow("image");
        setMouseCallback("image", onMouse, 0);
        imshow("image",frame2);
    }

    int res = waitKey(0);
    if (res == 32){
        if(file.length()==0)
            file = "sample.jpg";
        imwrite(file,frame);
    }
    
}
