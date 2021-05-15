#ifndef YOLOV5NET_H
#define YOLOV5NET_H
//#include "common.hpp"
#include "NvInferRuntime.h"
#include "NvInfer.h"
#include "logging.h"
#include "yololayer.h"

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "yololayer.h"
using namespace nvinfer1;

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
static const char* INPUT_BLOB_NAME = "data";
static const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;
void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize);
	bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& img_dir);
	void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, bool& is_p6, float& gd, float& gw, std::string& wts_name);
namespace DeepBlueDet {
}
#endif // YOLOV5NET_H
