/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "tracker.hpp"
#include <complex>
#include <cmath>
#include <string>
#include <iostream>
#include "fftw3.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include "featureColorName.hpp"


namespace cv{
  void*  fftwf_mallocWrapper(size_t n);
} /* namespace cv */

using namespace std;
using namespace Eigen;
typedef Matrix<float,Dynamic,Dynamic,RowMajor> MatrixF;
/*---------------------------
|  TrackerKCF
|---------------------------*/
namespace cv{

  /*
 * Prototype
 */
  class TrackerKCFImpl : public TrackerKCF {
  public:
    //  CAVEAT
    //  constructor and destructor function do not guarantee multi-thread safety!
    //  trakcer instance must new and delete in the same thread;
    TrackerKCFImpl( const TrackerKCF::Params &parameters = TrackerKCF::Params() );
    ~TrackerKCFImpl();
    bool resetImpl( const Mat& img, Rect2d& boundingBox, int temp_len);
    void read( const FileNode& /*fn*/ );
    void write( FileStorage& /*fs*/ ) const;
    //  CAVEAT
    //  wisdom operation method do not guarantee multi-thread safety!
    static string importWisdom();
    static bool   importWisdom(std::string& file);
    static void   clearWisdom();
    static string wisdomFile;

  protected:
     /*
    * basic functions and vars
    */
    bool initImpl( const Mat& /*image*/, const Rect2d& boundingBox );
    void release();
    bool updateImpl( const Mat& image, Rect2d& boundingBox );

    TrackerKCF::Params params;

    /*
    * KCF functions and vars
    */
    void createHanningWindow(OutputArray dest, const cv::Size winSize, const int type) ;
    void inline featureVectorInit(int row, int col, int cn);
    void inline fftwInit(int row, int col, int cn);

    void inline fftTool(const vector<Mat> & src, vector<Mat> & dest);
    void inline fftTool(const Mat src, Mat & dest);
    void inline ifftTool(const Mat src, Mat & dest);
    
    void inline pixelWiseMult(const std::vector<Mat> src1, const std::vector<Mat>  src2, const int flags, const bool conjB=false) ;
    void inline sumChannels(std::vector<Mat> src, Mat & dest) ;
    void inline updateProjectionMatrix(float pca_rate, int compressed_sz);
    void inline compress(Map<MatrixF> & sm, Map<MatrixF> & dm);
    void inline map2vector(Map<MatrixF> &m, vector<Mat> &vc);
    bool getSubWindow(const Mat & img, const Rect roi, Mat& patch);
    void extractCN(Mat patch_data);
    void pickIndex(int row,int col) ;

    void denseLinearKernel( const vector<Mat> & x_v, const vector<Mat> & y_v,
                            Mat & kf_data,
                            std::vector<Mat> & xf_data, std::vector<Mat> & yf_data, 
                            bool symmetric);

    void calcResponse(const Mat alphaf_data, const Mat kf_data, Mat & response_data, Mat & spec_data) ;

//    void calcAlphaf(int frame, float interp_factor, float lambda) __attribute__((noinline));
    void calcAlphaf(int frame, float interp_factor, float lambda);


  private:
    float output_sigma;
    Rect2d roi;

    Mat img_Patch;
    Mat patch_data_tmp;
    Mat hann; 	//hann window filter
    // hog extract
    vector<Mat> hog_index_vu,hog_index_vf;

    Mat y; 	// training response and its FFT
    Mat yf;
    Mat kf;	// dense gaussian kernel and its FFT
    Mat kf_lambda; // kf+lambda    

    vector<Mat> xcv;
    vector<Mat> zcv;

    Mat new_alphaf, alphaf;	 // training coefficients
    Mat response; // detection result
    // pre-defined Mat variables for optimization of private functions
    Mat spec;
    std::vector<Mat> vxf,vyf;

    float scalarf;
    fftwf_plan fftPlan;
    fftwf_plan ifftPlan;

    // eigen matrix for PCA
    vector<Mat> xv;
    Map<MatrixF> hannMap;
    Map<MatrixF> xMap,xcMap;
    Map<MatrixF> zMap,z_tmp_map,zcMap;

    MatrixF old_cov_matrix,new_covar_matrix, project_matrix;

    float resize_scale;
    int frame;
  };

  /*
 * Constructor
 */
  Ptr<TrackerKCF> TrackerKCF::create(const TrackerKCF::Params &parameters){
      return Ptr<TrackerKCFImpl>(new TrackerKCFImpl(parameters));
  }
  Ptr<TrackerKCF> TrackerKCF::create(){
      return Ptr<TrackerKCFImpl>(new TrackerKCFImpl());
  }

  string TrackerKCFImpl::importWisdom(){
    string file;
    char * wisdomPath = getenv("wisdom");
    if (wisdomPath == nullptr) {
      cout << "env wisdom is null, import from current work dir" 
           << endl;
      file = string("wisdom");
    } else {
      file = string(wisdomPath) + string("/wisdom");
    }
    if(0==fftwf_import_wisdom_from_filename(file.c_str())){
      cout << "fftw wisdom file import failed!" <<endl;
      exit(0);
    } else {
      cout << "wisdom import success" << endl;
    }
    return file;
  }
  
  bool TrackerKCFImpl::importWisdom(std::string &file){
    if(0==fftwf_import_wisdom_from_filename(file.c_str())){
      cout << "fftw wisdom file import failed!" <<endl;
      return false;
    } else {
      cout << "wisdom import success" << endl;
      wisdomFile = file;
      return true;
    }
  }

  string TrackerKCFImpl::wisdomFile = TrackerKCFImpl::importWisdom();

  void TrackerKCFImpl::clearWisdom(){
    fftwf_forget_wisdom();
  }

  TrackerKCFImpl::TrackerKCFImpl( const TrackerKCF::Params &parameters ) :
      // eigen map init
      params( parameters ), 
      hannMap(nullptr,0,0), 
      xMap(nullptr,0,0),xcMap(nullptr,0,0),
      zMap(nullptr,0,0),zcMap(nullptr,0,0),z_tmp_map(nullptr,0,0)
  {
    isInit = false;
  }

  TrackerKCFImpl::~TrackerKCFImpl(){
    if(isInit == true)
      release();
  }

  void TrackerKCFImpl::read( const cv::FileNode& fn ){
    params.read( fn );
  }

  void TrackerKCFImpl::write( cv::FileStorage& fs ) const {
    params.write( fs );
  }

  /*
   * Initialization:
   * - creating hann window filter
   * - ROI padding
   * - creating a gaussian response for the training ground-truth
   * - perform FFT to the gaussian response
   */
  bool TrackerKCFImpl::initImpl( const Mat& image, const Rect2d& boundingBox ){
    frame=0;
    roi.x = cvRound(boundingBox.x);
    roi.y = cvRound(boundingBox.y);
    roi.width = cvRound(boundingBox.width);
    roi.height = cvRound(boundingBox.height);

    if(roi.width > roi.height){
      roi.y -= (roi.width - roi.height)/2;
      roi.height = roi.width;
    } else {
      roi.x -= ( roi.height - roi.width )/2;
      roi.width = roi.height;
    }

    resize_scale = (float)roi.width * (1+params.pad_scale) / params.template_len;
    roi.x/=resize_scale;
    roi.y/=resize_scale;
    roi.width/=resize_scale;
    roi.height/=resize_scale;
    // add padding to the roi
    roi.x-=roi.width/2*params.pad_scale;
    roi.y-=roi.height/2*params.pad_scale;
    //roi.width = roi.width*(1+params.pad_scale);
    //roi.height = roi.height * (1+params.pad_scale);
    roi.width  = params.template_len;
    roi.height = params.template_len;

    //calclulate output sigma
    output_sigma=std::sqrt(static_cast<float>(roi.width*roi.height))*params.output_sigma_factor;
    output_sigma=-0.5f/(output_sigma*output_sigma);

    // initialize the hann window filter
    createHanningWindow(hann, roi.size(), CV_32F);
    // eigen hann
    new (& hannMap) Map<MatrixF>((float *)hann.data,hann.rows,hann.cols);

    // feature vector init
    featureVectorInit(roi.height, roi.width, 10);
    // fftw buffer matrix init
    fftwInit(roi.height, roi.width, params.compressed_size);
    // create gaussian response
    y = 0;

    for(int i=0;i<int(roi.height);i++){
      for(int j=0;j<int(roi.width);j++){
        y.at<float>(i,j) =
                static_cast<float>((i-roi.height/2+1)*(i-roi.height/2+1)+(j-roi.width/2+1)*(j-roi.width/2+1));
      }
    }

    y*=(float)output_sigma;
    cv::exp(y,y);
    // perform fourier transfor to the gaussian response
    fftTool(y,yf);

    //return true only if roi has intersection with the image
    if((roi & Rect2d(0,0,  image.cols / resize_scale ,
                           image.rows / resize_scale )) == Rect2d()) {
      return false;
    } 
      
    return true;
  }
  /*
    release resource (include heap memory for feature vector,
                              aligned memory for fftw) 
    requested when tracker init 
  */
  void TrackerKCFImpl::release(){
    // free feature vector
    free((void *)xMap.data());
    free((void *)zMap.data());
    free((void *)z_tmp_map.data());
    // free fftw aligned memory
    fftwf_destroy_plan(fftPlan);
    fftwf_destroy_plan(ifftPlan);

    for(size_t i = 0; i < vxf.size(); i++){
      fftwf_free((void *) vyf[i].data);
      fftwf_free((void *) vxf[i].data);
    }
    fftwf_free( y.data);
    fftwf_free( yf.data);
    fftwf_free( spec.data);
    fftwf_free( response.data);
    fftwf_free((void *) xcMap.data());
    fftwf_free((void *) zcMap.data());
  }
  /*
    reset tracker state, 
    input arguments image 
  */

  bool TrackerKCFImpl::resetImpl( const Mat& img,  Rect2d& boundingBox, int temp_len){
    if(isInit==true)
      release();
    else
      isInit = true;
    if(temp_len!=0)
      params.template_len = temp_len;
    return initImpl(img,boundingBox);
  }
  /*
   * Main part of the KCF algorithm
   */
  bool TrackerKCFImpl::updateImpl( const Mat& image, Rect2d& boundingBox ) {

    double minVal, maxVal;	// min-max response
    Point minLoc,maxLoc;	// min-max location

    Mat img = image; 
    // check the channels of the input image, grayscale is preferred
    CV_Assert(img.channels() == 3);

    // resize the image whenever needed
    resize(image,img,Size(img.cols/resize_scale,img.rows/resize_scale),0,0,INTER_LINEAR_EXACT);

    // detection part
    if(frame>0){
      // extract and pre-process the patch
      // get non compressed descriptors
      if(!getSubWindow(img,roi, img_Patch)){
        cout << "getSubWindow for new roi fail" << endl;
        return false;
      }
      //compress the features and the KRSL model
      compress(xMap,xcMap);
      map2vector(xcMap,xcv);
    
      compress(zMap,zcMap);
      map2vector(zcMap,zcv);
      
      //compute the gaussian kernel
      denseLinearKernel(xcv,zcv,kf,vxf,vyf,false);
      // calculate filter response
      calcResponse(alphaf, kf, response, spec);
      // extract the maximum response
      minMaxLoc( response, &minVal, &maxVal, &minLoc, &maxLoc );
      if (maxVal < params.detect_thresh)
      {
          cout << "maxVal < params.detect_thresh" << endl;
          return false;
      }
      roi.x+=(maxLoc.x-roi.width/2+1);
      roi.y+=(maxLoc.y-roi.height/2+1);
    }

    // update the bounding box
    boundingBox.x= (roi.x+roi.width/(1+params.pad_scale)/2*params.pad_scale)*resize_scale;
    boundingBox.y=(roi.y+roi.height/(1+params.pad_scale)/2*params.pad_scale)*resize_scale;
    boundingBox.width = roi.width*resize_scale/(1+params.pad_scale);
    boundingBox.height = roi.height*resize_scale/(1+params.pad_scale);
    // extract the patch for learning purpose
    // get non compressed descriptors
    if(!getSubWindow(img, roi, img_Patch)){
      cout << "getSubWindow for after update" << endl;
      return false;
    }
    //update the training data
    if(frame==0){
        // eigen matrix
      zMap = xMap;
    }else{
      // eigen matrix
      zMap =(1.0-params.interp_factor)*zMap+params.interp_factor*xMap;
    }
    // feature compression
    updateProjectionMatrix(params.pca_learning_rate,params.compressed_size);
    compress(xMap,xcMap);
    map2vector(xcMap,xcv);
    // initialize some required Mat variables
    if(frame==0){
      new_alphaf = Mat_<Vec2f >(yf.rows, yf.cols);
      // prepare for fftwf forward and backword context
    }
    // Kernel Regularized Least-Squares, calculate alphas
    denseLinearKernel(xcv,xcv,kf,vxf,vyf,true);
    calcAlphaf(frame,params.interp_factor,params.lambda);

    frame++;
    return true;
  }


  /*-------------------------------------
  |  implementation of the KCF functions
  |-------------------------------------*/

  /*
   * hann window filter
   */
  void TrackerKCFImpl::createHanningWindow(OutputArray dest, const cv::Size winSize, const int type) {
      CV_Assert( type == CV_32FC1 || type == CV_64FC1 );

      dest.create(winSize, type);
      Mat dst = dest.getMat();

      int rows = dst.rows, cols = dst.cols;

      AutoBuffer<float> _wc(cols);
      float * const wc = (float *)_wc;

      const float coeff0 = 2.0f * (float)CV_PI / (cols - 1);
      const float coeff1 = 2.0f * (float)CV_PI / (rows - 1);
      for(int j = 0; j < cols; j++)
        wc[j] = 0.5f * (1.0f - cos(coeff0 * j));

      if(dst.depth() == CV_32F){
        for(int i = 0; i < rows; i++){
          float* dstData = dst.ptr<float>(i);
          float wr = 0.5f * (1.0f - cos(coeff1 * i));
          for(int j = 0; j < cols; j++)
            dstData[j] = (float)(wr * wc[j]);
        }
      }else{
        for(int i = 0; i < rows; i++){
          double* dstData = dst.ptr<double>(i);
          double wr = 0.5f * (1.0f - cos(coeff1 * i));
          for(int j = 0; j < cols; j++)
            dstData[j] = wr * wc[j];
        }
      }

      // perform batch sqrt for SSE performance gains
      //cv::sqrt(dst, dst); //matlab do not use the square rooted version
  }

  void * fftwf_mallocWrapper(size_t n) {
    void * res = fftwf_malloc(n); 
    assert(res != nullptr);
    return res;
  }

  void inline TrackerKCFImpl::featureVectorInit(int row, int col, int cn){
      // hog extract
      hog_index_vf.resize(3);
      hog_index_vu.resize(3);
      for(int i = 0; i< 3; i++){
        hog_index_vf[i].create(row,col,CV_32FC1);
        hog_index_vu[i].create(row,col,CV_16UC1);
      }
        
      size_t s = row * col;
      // eigen matrix vector
      xv.resize(cn);
      float * p = (float *)malloc(sizeof(float)*s*cn);
      assert(p != nullptr);
      for(int i = 0; i< cn; i++)
        xv[i] = Mat(row, col, CV_32FC1, (void *)(p+i*s));
      new (&xMap)Map<MatrixF>(p,cn,row*col);

      p = (float *)malloc(sizeof(float)*s*cn);
      assert(p != nullptr);
      new (&zMap)Map<MatrixF>(p,cn,row*col);

      p = (float *)malloc(sizeof(float)*s*cn);
      assert(p != nullptr);
      new ( &z_tmp_map ) Map<MatrixF> (p,cn,row*col);

  }

  void inline TrackerKCFImpl::fftwInit(int row, int col, int cn) {

    scalarf = row * col;
    xcv.resize(cn);
    zcv.resize(cn);
    vxf.resize(cn);
    vyf.resize(cn);
    for(int i = 0; i< cn ; i++){
      vyf[i]    = Mat(row,(col/2+1),CV_32FC2,fftwf_mallocWrapper(sizeof (fftwf_complex) * row * (col/2+1) ));
      vxf[i]    = Mat(row,(col/2+1),CV_32FC2,fftwf_mallocWrapper(sizeof (fftwf_complex) * row * (col/2+1) ));
    }
    // eigen matrix
    float * p;
    p = (float *)fftwf_mallocWrapper(sizeof(float) * row * col * cn);
    new ( &xcMap ) Map<MatrixF> (p,cn,row*col);
    p = (float *)fftwf_mallocWrapper(sizeof(float) * row * col * cn);
    new ( &zcMap ) Map<MatrixF> (p,cn,row*col);

    y        = Mat(row,col,CV_32FC1,fftwf_mallocWrapper(sizeof (float) * row * col ));
    yf       = Mat(row,col/2+1,CV_32FC2,fftwf_mallocWrapper(sizeof (fftwf_complex) * row * (col/2+1) ));
    fftPlan  = fftwf_plan_dft_r2c_2d(row, col, (float *) y.data, (fftwf_complex *) yf.data,
                                        FFTW_WISDOM_ONLY|FFTW_PATIENT);
    if(fftPlan == nullptr ){
      cout << "fftwf create fft plan failed!" << endl;
      exit(1);
    }

    spec     = Mat(row,col/2+1,CV_32FC2,fftwf_mallocWrapper(sizeof (fftwf_complex) * row * (col/2+1) ));
    response = Mat(row,col,CV_32FC1,fftwf_mallocWrapper(sizeof (float) * row * col ));
    ifftPlan    = fftwf_plan_dft_c2r_2d(row, col, (fftwf_complex *) spec.data, (float *) response.data, 
                                         FFTW_WISDOM_ONLY|FFTW_PATIENT);

    if(ifftPlan == nullptr){
      cout << "fftwf create ifft plan failed!" << endl;
      exit(1);
    }

  }

  /*
   * wrapper to fourier transform function with fftw
   */

  void inline TrackerKCFImpl::fftTool(const Mat src, Mat & dest) {
    assert(src.channels() == 1);
    fftwf_execute_dft_r2c(fftPlan, (float *) src.data, (fftwf_complex *) dest.data);
  }

  void inline TrackerKCFImpl::fftTool(const vector<Mat> & src, std::vector<Mat> & dest) {
    for(int i=0;i<src.size();i++)
      fftTool(src[i],dest[i]);
  }

  void inline TrackerKCFImpl::ifftTool(const Mat src, Mat & dest) {
    assert(src.channels() == 2);
    fftwf_execute_dft_c2r(ifftPlan, (fftwf_complex *) src.data, (float *) dest.data);
  }

  /*
   * Point-wise multiplication of two Multichannel Mat data
   */
  void inline TrackerKCFImpl::pixelWiseMult(const std::vector<Mat> src1, const std::vector<Mat>  src2, const int flags, const bool conjB) { 
    // eigen matrix
    int len = src1[0].rows*src1[0].cols;
    for(unsigned i=0;i<src1.size();i++){
      Map<ArrayXcf> a1((std::complex<float> *) src1[i].data ,len);
      Map<ArrayXcf> a2((std::complex<float> *) src2[i].data ,len);
      if(conjB)
        a1 = a1 * a2.conjugate();
      else
        a1 = a1 * a2;
    }
    
  }

  /*
   * Combines all channels in a multi-channels Mat data into a single channel
   */
  void inline TrackerKCFImpl::sumChannels(std::vector<Mat> src, Mat & dest) {
    dest=src[0].clone();
    for(unsigned i=1;i<src.size();i++){
      dest+=src[i];
    }

    // eigen matrix
    int len = dest.rows*dest.cols;
    Map<ArrayXcf> a((std::complex<float> *) dest.data ,len);
    for(unsigned i=1;i<src.size();i++){
      Map<ArrayXcf> b((std::complex<float> *) src[i].data ,len);
      a = a + b;
    }

  }


  /*
   * obtains the projection matrix using PCA
   */
  void inline TrackerKCFImpl::updateProjectionMatrix(float pca_rate, int compressed_sz) {

    // eigen matrix
    z_tmp_map = zMap;
    float * p = (float *) z_tmp_map.data();
    Map<ArrayXf> feat_vec(nullptr,0);
    for (int i=0; i<10; i++){
      new (& feat_vec) Map<ArrayXf>(p+i*z_tmp_map.cols(),z_tmp_map.cols());
      feat_vec -= feat_vec.mean();
    }
    // calc covariance matrix
    new_covar_matrix = 1.0/(z_tmp_map.cols()-1) * z_tmp_map * z_tmp_map.transpose();

    if(old_cov_matrix.cols() ==0) old_cov_matrix = new_covar_matrix;
    // calc PCA
    BDCSVD<MatrixF> svd((1.0-pca_rate)*old_cov_matrix+pca_rate*new_covar_matrix,ComputeFullU);
    project_matrix = svd.matrixU().leftCols(compressed_sz);
    MatrixF proj_vars_matrix = MatrixF::Identity(compressed_sz,compressed_sz);
    for(int i=0;i<compressed_sz;i++)
      proj_vars_matrix(i,i)=svd.singularValues()(i);
    old_cov_matrix = (1.0-pca_rate)*old_cov_matrix + pca_rate*project_matrix*proj_vars_matrix*project_matrix.transpose();

  }

  /*
   * compress the features
   */
  void inline TrackerKCFImpl::compress(Map<MatrixF> & sm, Map<MatrixF> & dm)  {
    // eigen matrix
    dm = project_matrix.transpose() * sm;

  }

  void inline TrackerKCFImpl::map2vector(Map<MatrixF> &m, vector<Mat> &vc){
    float * p = m.data();
    for(size_t i = 0; i<vc.size(); i++ ){
      vc[i] = Mat(xv[0].rows,xv[0].cols,CV_32FC1,(void *)p);
      p += m.cols();
    }
  }

  /*
   * obtain the patch and apply hann window filter to it
   */
  bool TrackerKCFImpl::getSubWindow(const Mat & img, const Rect _roi, Mat& patch) {

    Rect region=_roi;
    // return false if roi is outside the image
    if((roi & Rect2d(0,0, img.cols, img.rows)) == Rect2d() ){
      return false;
    }
    // extract patch inside the image
    if(_roi.x<0){region.x=0;region.width+=_roi.x;}
    if(_roi.y<0){region.y=0;region.height+=_roi.y;}

    if(_roi.x+_roi.width>img.cols)region.width=img.cols-_roi.x;
    if(_roi.y+_roi.height>img.rows)region.height=img.rows-_roi.y;

    if(region.width>img.cols)region.width=img.cols;
    if(region.height>img.rows)region.height=img.rows;

    patch=img(region).clone();

    // add some padding to compensate when the patch is outside image border
    int addTop,addBottom, addLeft, addRight;
    addTop=region.y-_roi.y;
    addBottom=(_roi.height+_roi.y>img.rows?_roi.height+_roi.y-img.rows:0);
    addLeft=region.x-_roi.x;
    addRight=(_roi.width+_roi.x>img.cols?_roi.width+_roi.x-img.cols:0);
    copyMakeBorder(patch,patch,addTop,addBottom,addLeft,addRight,BORDER_REPLICATE);
    if(patch.rows==0 || patch.cols==0)return false;

    // extract the desired descriptors
    extractCN(patch);
    // eigen generate hog feature vector
    Map<MatrixF> x_temp(nullptr,0,0);
    for(int i = 0; i< 10; i++){
      new (& x_temp) Map<MatrixF>((float *)xv[i].data,patch.rows,patch.cols);
      x_temp.array() = x_temp.array() * hannMap.array();
    }
    return true;

  }

   void TrackerKCFImpl::pickIndex(int row,int col)  {
    unsigned index;
    for(int k=0;k<10;k++)
      for(int i=0;i<row;i++)
        for(int j=0;j<col;j++)
        {
          index = hog_index_vu[0].at<unsigned short int>(i,j);
          xv[k].at<float>(i,j) = ColorNames[k][index];
        }
  }
  /* Convert BGR to ColorNames
   */
  void TrackerKCFImpl::extractCN(Mat patch_data) {
    // hog extract
    patch_data_tmp.create(patch_data.rows,patch_data.cols,CV_32FC3);
    patch_data.convertTo(patch_data_tmp,CV_32FC3);
    patch_data_tmp = patch_data_tmp / 8;
    split(patch_data_tmp,hog_index_vf);
    for(size_t k = 0; k<3; k++ ){
      for(int i = 0; i < patch_data_tmp.rows ; i++)
        for(int j = 0; j < patch_data_tmp.cols; j++)
          hog_index_vu[k].at<unsigned short int>(i,j)=floor(hog_index_vf[k].at<float>(i,j));
    }
    hog_index_vu[1] = 32 *      hog_index_vu[1];
    hog_index_vu[0] = 32 * 32 * hog_index_vu[0];
    hog_index_vu[0] += hog_index_vu[1];
    hog_index_vu[0] += hog_index_vu[2];

    pickIndex(patch_data_tmp.rows,patch_data_tmp.cols);

  }

  /*
   *  dense linear kernel function
   */
  void TrackerKCFImpl::denseLinearKernel(
                                        const vector<Mat> & x_v, const vector<Mat> & y_v,
                                        Mat & kf_data,
                                        std::vector<Mat> & xf_data, std::vector<Mat> & yf_data, 
                                        bool symmetric)  {

    if(symmetric == true){
      fftTool(x_v,xf_data);
      pixelWiseMult(xf_data, xf_data, 0, true);
      sumChannels(xf_data,kf_data);
      //sumChannels(xyf_v,kf_data);
      
      kf_data = kf_data / (x_v[0].rows* x_v[0].cols * x_v.size());
    } else {
      fftTool(x_v,xf_data);
      fftTool(y_v,yf_data);
      pixelWiseMult(xf_data, yf_data, 0, true);
      sumChannels(xf_data,kf_data);
      //sumChannels(xyf_v,kf_data);
      kf_data = kf_data / (x_v[0].rows* x_v[0].cols * x_v.size());
    }
  }



  /*
   * calculate the detection response
   */
  void TrackerKCFImpl::calcResponse(const Mat alphaf_data, const Mat kf_data, Mat & response_data, Mat & spec_data) {
    // eigen matrix
    int len = kf_data.rows*kf_data.cols;
    Map<ArrayXcf> a((std::complex<float> *) alphaf_data.data ,len);
    Map<ArrayXcf> b((std::complex<float> *) kf_data.data ,len);
    Map<ArrayXcf> c((std::complex<float> *) spec_data.data ,len);
    c = a * b;

    ifftTool(spec_data,response_data);
    response_data = response_data / scalarf;

  }

  void TrackerKCFImpl::calcAlphaf(int frame, float interp_factor, float lambda) {
    // eigen matrix
    int len = kf.rows*kf.cols;
    kf_lambda.create(kf.rows,kf.cols,CV_32FC2);
    Map<ArrayXcf> a((std::complex<float> *) kf.data ,len);
    Map<ArrayXcf> b((std::complex<float> *) kf_lambda.data ,len);
    Map<ArrayXcf> c((std::complex<float> *) yf.data ,len);
    Map<ArrayXcf> d((std::complex<float> *) new_alphaf.data ,len);
    Map<ArrayXcf> e((std::complex<float> *) alphaf.data ,len);
    b = a + lambda;
    d = c / b ;

    if(frame==0){
      alphaf=new_alphaf.clone();
    }else{
      e = (1.0-interp_factor) * e + interp_factor * d;
    }

  }

  /*----------------------------------------------------------------------*/

  /*
 * Parameters
 */
  TrackerKCF::Params::Params(){
      detect_thresh = 0.5f;
      lambda=0.0001f;
      interp_factor=0.085f;
      output_sigma_factor=1.0f / 16.0f;

      compressed_size=2;
      pca_learning_rate=0.15f;

      template_len = 64;
      pad_scale = 0.5f;
  }

  void TrackerKCF::Params::read( const cv::FileNode& fn ){
      *this = TrackerKCF::Params();

      if (!fn["detect_thresh"].empty())
          fn["detect_thresh"] >> detect_thresh;

      if (!fn["lambda"].empty())
          fn["lambda"] >> lambda;

      if (!fn["interp_factor"].empty())
          fn["interp_factor"] >> interp_factor;

      if (!fn["output_sigma_factor"].empty())
          fn["output_sigma_factor"] >> output_sigma_factor;

      if (!fn["compressed_size"].empty())
          fn["compressed_size"] >> compressed_size;

      if (!fn["pca_learning_rate"].empty())
          fn["pca_learning_rate"] >> pca_learning_rate;

      if (!fn["template_len"].empty())
          fn["template_len"] >> template_len;

      if (!fn["pad_scale"].empty())
          fn["pad_scale"] >> pad_scale;
  }

  void TrackerKCF::Params::write( cv::FileStorage& fs ) const{
    fs << "detect_thresh" << detect_thresh;
    fs << "lambda" << lambda;
    fs << "interp_factor" << interp_factor;
    fs << "output_sigma_factor" << output_sigma_factor;
    fs << "compressed_size" << compressed_size;
    fs << "pca_learning_rate" << pca_learning_rate;
    fs << "template_len" << template_len;
    fs << "pad_scale" << pad_scale;
  }
} /* namespace cv */
