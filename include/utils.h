#ifndef TRTX_YOLOV5_UTILS_H_
#define TRTX_YOLOV5_UTILS_H_

#include <dirent.h>
#include <opencv2/opencv.hpp>

cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h);
int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names);

#endif  // TRTX_YOLOV5_UTILS_H_

