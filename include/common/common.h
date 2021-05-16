#ifndef YOLOV5_COMMON_H_
#define YOLOV5_COMMON_H_

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "yolov5cfg.h"
#include "yolov4cfg.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)
using namespace nvinfer1;

cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h);

cv::Rect get_rect(cv::Mat& img, float bbox[4]);

float iou(float lbox[4], float rbox[4]);
namespace yolov5 {
	bool cmp(const yolov5::Detection& a, const yolov5::Detection& b);

	void nms(std::vector<yolov5::Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5);
	void nms_all(std::vector<yolov5::Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5);
}

namespace yolov4 {
	bool cmp(const yolov4::Detection& a, const yolov4::Detection& b);

	void nms(std::vector<yolov4::Detection>& res, float *output, float nms_thresh = NMS_THRESH);
}

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file);

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps);

ILayer* convBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname);

ILayer* focus(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int ksize, std::string lname);

ILayer* bottleneckCSP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname);
ILayer* C3(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int n, bool shortcut, int g, float e, std::string lname);
ILayer* SPP(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int c1, int c2, int k1, int k2, int k3, std::string lname);

std::vector<float> getAnchors(std::map<std::string, Weights>& weightMap);

IPluginV2Layer* addYoLoLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, std::string lname, std::vector<IConvolutionLayer*> dets);
int get_width(int x, float gw, int divisor = 8);
int get_depth(int x, float gd);
int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names);
#endif

