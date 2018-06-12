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

#ifndef __OPENCV_TRACKER_HPP__
#define __OPENCV_TRACKER_HPP__

//#include "cvconfig.h"
#include "opencv2/core/cvdef.h"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include <typeinfo>

#include "opencv2/core.hpp"
#include "opencv2/imgproc/types_c.h"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
#include <time.h>
#include <stdlib.h>

/*
 * Partially based on:
 * ====================================================================================================================
 *   - [AAM] S. Salti, A. Cavallaro, L. Di Stefano, Adaptive Appearance Modeling for Video Tracking: Survey and Evaluation
 *  - [AMVOT] X. Li, W. Hu, C. Shen, Z. Zhang, A. Dick, A. van den Hengel, A Survey of Appearance Models in Visual Object Tracking
 *
 * This Tracking API has been designed with PlantUML. If you modify this API please change UML files under modules/tracking/doc/uml
 *
 */

namespace cv
{

//! @addtogroup tracking
//! @{

/************************************ Tracker Base Class ************************************/

/** @brief Base abstract class for the long-term tracker:
 */
class CV_EXPORTS_W Tracker : public virtual Algorithm
{
 public:

  virtual ~Tracker();

  /** @brief Initialize the tracker with a known bounding box that surrounded the target
    @param image The initial frame
    @param boundingBox The initial bounding box

    @return True if initialization went succesfully, false otherwise
     */
  CV_WRAP bool init( InputArray image, const Rect2d& boundingBox );

  /** @brief Update the tracker, find the new most likely bounding box for the target
    @param image The current frame
    @param boundingBox The bounding box that represent the new target location, if true was returned, not
    modified otherwise

    @return True means that target was located and false means that tracker cannot locate target in
    current frame. Note, that latter *does not* imply that tracker has failed, maybe target is indeed
    missing from the frame (say, out of sight)
     */
  CV_WRAP bool update( InputArray image, CV_OUT Rect2d& boundingBox );

  CV_WRAP bool reset( InputArray image, Rect2d& boundingBox, int temp_len=0);

  virtual void read( const FileNode& fn )=0;
  virtual void write( FileStorage& fs ) const=0;

  long id;

 protected:

  virtual bool initImpl( const Mat& image, const Rect2d& boundingBox ) = 0;
  virtual bool updateImpl( const Mat& image, Rect2d& boundingBox ) = 0;
  virtual bool resetImpl( const Mat& image, Rect2d& boundingBox, int temp_len ) = 0;

  bool isInit;

};

/** @brief KCF is a novel tracking framework that utilizes properties of circulant matrix to enhance the processing speed.
 * This tracking method is an implementation of @cite KCF_ECCV which is extended to KCF with color-names features (@cite KCF_CN).
 * The original paper of KCF is available at <http://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf>
 * as well as the matlab implementation. For more information about KCF with color-names features, please refer to
 * <http://www.cvl.isy.liu.se/research/objrec/visualtracking/colvistrack/index.html>.
 */
class CV_EXPORTS_W TrackerKCF : public Tracker
{
public:
  /**
  * \brief Feature type to be used in the tracking grayscale, colornames, compressed color-names
  * The modes available now:
  -   "GRAY" -- Use grayscale values as the feature
  -   "CN" -- Color-names feature
  */
  enum MODE {
    GRAY   = (1 << 0),
    CN     = (1 << 1),
    CUSTOM = (1 << 2)
  };

  struct CV_EXPORTS Params
  {
    /**
    * \brief Constructor
    */
    Params();

    /**
    * \brief Read parameters from a file
    */
    void read(const FileNode& /*fn*/);

    /**
    * \brief Write parameters to a file
    */
    void write(FileStorage& /*fs*/) const;

    float detect_thresh;         //!<  detection confidence threshold
    float lambda;                //!<  regularization
    float interp_factor;         //!<  linear interpolation factor for adaptation
    float output_sigma_factor;   //!<  spatial bandwidth (proportional to target)
    float pca_learning_rate;     //!<  compression learning rate
    int compressed_size;          //!<  feature size after compression
    int template_len;             // correlation template size
    float pad_scale;              // padding scale to origin rectangel
  };

  /** @brief Constructor
  @param parameters KCF parameters TrackerKCF::Params
  */
  static Ptr<TrackerKCF> create(const TrackerKCF::Params &parameters);

  CV_WRAP static Ptr<TrackerKCF> create();

  virtual ~TrackerKCF() {}
};

} /* namespace cv */

#endif
