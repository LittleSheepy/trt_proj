#ifndef YOLOV5_H
#define YOLOV5_H
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
namespace DeepBlueDet {
    //#define USE_FP16  // comment out this if want to use FP32
	#define DEVICE 0  // GPU id
	#define NMS_THRESH 0.4
	#define CONF_THRESH 0.5
	#define BATCH_SIZE 1
	#define NET m  // s m l x
	#define NETSTRUCT(str) createEngine_##str
	#define CREATENET(net) NETSTRUCT(net)
	#define STR1(x) #x
	#define STR2(x) STR1(x)

	// stuff we know about the network and the input/output blobs
	static const int INPUT_H = Yolo::INPUT_H;
	static const int INPUT_W = Yolo::INPUT_W;
	static const int CLASS_NUM = Yolo::CLASS_NUM;
	// we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
	static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1; 
	static const char* INPUT_BLOB_NAME = "data";
	static const char* OUTPUT_BLOB_NAME = "prob";
	static Logger gLogger;

	class YOLOV5
	{
	public:
		YOLOV5();
		~YOLOV5();
		void Init();
		void doInference();
		std::vector<Yolo::Detection> predict(cv::Mat& img);
        std::vector<std::string> predict_str(cv::Mat& img);
	public:
		IRuntime*			m_runtime	= nullptr;					// 
		ICudaEngine*		m_engine	= nullptr;					// 
		IExecutionContext*	m_context	= nullptr;					// 
		cudaStream_t		m_stream	= nullptr;					// 
        void*				buffers[2];								//
		float		input[BATCH_SIZE * 3 * INPUT_H * INPUT_W];		// 图片数据
		float		output[BATCH_SIZE * OUTPUT_SIZE];				// 输出数据
		int					inputIndex;
		int					outputIndex;
	};
}
#endif // YOLOV5_H
