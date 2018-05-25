#include <opencv2/core/utility.hpp>
#include "tracker.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <Eigen/Dense>
//#include "featureColorName.hpp"

#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;
using namespace Eigen;

typedef Matrix<float,Dynamic,Dynamic,RowMajor> MatrixF;


int main(int argc,char* argv[]){

    float mDataA[] = {1,   2,   3, 4,   5, 6,   4, 0};
    float mDataB[] = {0.1, 0, 0.1, 1, 0.1, 0, 0.1, 0};
    Mat D(3,2,CV_32FC1,mDataA);
    Mat d = D(Rect(0,0,1,2));
    cout << "d: " << d << endl;

    Mat A(2,2,CV_32FC2,mDataA);
    Mat B(2,2,CV_32FC2,mDataB);
    Mat C;
    mulSpectrums(A,B,C,0,true);
    std::cout<< "result C: " << C << std::endl;
    std::cout<< " norm(A): " << norm(A) << std::endl;

    Map< Matrix<float,2,2> > eA(mDataA,2,2);
    Map< Matrix<float,2,2> > eB(mDataB,2,2);
    eA.array() = eA.array() * eB.array();
    cout << "eA.data: " << (long int)eA.data() << endl;
    eA = eB;
    cout << "eA.data: " << (long int)eA.data() << endl;
    cout << "coefficent-wise multiply: " << eA << endl;


    B = Mat(2,3,CV_32FC1,mDataB);
    std::cout<< " B.row(0): " << B.row(0) << std::endl;
    std::cout<< " B.row(1): " << B.row(1) << std::endl;
    float cA[3*2] = {1,2,3,4,5,6};
    float cB[2*2] = {1,1,0,0};
    float cC[3*2] = {0,0,0,0,0,0};

    std::complex<float> * p = (std::complex<float> *) cA;

    cout <<"p0"<< p[0]
         <<"p1"<< p[1]
         <<"p2"<< p[2]
         << endl;
    cout << Map< Matrix<float,3,2> >(cA) << endl;
    cout << Map< Matrix<float,2,2> >(cB) << endl;

    Map< Matrix<float,3,2> > L(cA);
    Map< Matrix<float,2,2> > R(cB);

    cout<< L(2,1) << endl;
    

/*
    cout << "sizeof ColorNames: "<<sizeof(ColorNames)/sizeof(float)/10 << endl;


    if (ColorNames[123][1]!= ColorNamesNew[1][123]) cout << "ColorNames !=" << endl; 
    else cout << "ColorNames ==" << endl; 


    if (ColorNames[12345][3]!= ColorNamesNew[3][12345]) cout << "ColorNames != 2" << endl; 
    else cout << "ColorNames == 2" << endl; 


    streambuf* coutBuf = cout.rdbuf();
    ofstream of("test.h");
    streambuf* fileBuf = of.rdbuf();

    cout.rdbuf(fileBuf);

//    Mat cn(sizeof(ColorNames)/sizeof(float)/10,10,CV_32FC1,(void *)&ColorNames[0][0]);
    
    cout << "const float ColorNames[10][32768]={" << endl;
    for(int i=0; i<10; i++){
        cout << "{";
        for(int j=0; j<32767; j++){
            cout<< ColorNames[j][i] <<",";
            if(j%1024==0){
                cout << endl;
            }
        }
        cout << ColorNames[32767][i]
             << "}," << endl;
    }
    of.flush();
    of.close();
    cout.rdbuf(coutBuf);
  */

    Map<MatrixF> map(nullptr,0,0);

    new (& map) Map<MatrixF>(cA,3,2);

    cout << "map" << map << endl;

    MatrixF m1(2,2);
    MatrixF m2;
    cout << "m1: "<<(float *) m1.data() << endl;
    m2 = m1;
    cout << "m2: "<<(float *) m2.data() << endl;



}