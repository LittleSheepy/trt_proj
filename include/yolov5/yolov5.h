#ifndef YOLOV5_H
#define YOLOV5_H
//#include "common.hpp"
#include "NvInferRuntime.h"
#include "NvInfer.h"
#include "logging.h"

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
//#include "yololayer.h"
#include "yolov5cfg.h"
using namespace nvinfer1;
using namespace yolov5;
namespace ObjDet {
	class YOLOV5
	{
	public:
		YOLOV5();
		~YOLOV5();
		void Init(std::string& engine_name);
		ICudaEngine* YOLOV5::build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name);
		ICudaEngine* YOLOV5::build_engine_p6(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name);
		int LoadEngine();
		void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, bool& is_p6, float& gd, float& gw, std::string& wts_name);
		void doInference();
		int serialize(std::string& wts_name, std::string& engine_name, bool& is_p6, float& gd, float& gw);
		//void deserialize();
		std::vector<yolov5::Detection> predict(cv::Mat& img);
        std::vector<std::string> predict_str(cv::Mat& img);
	public:
		IRuntime*			m_runtime	= nullptr;					// 
		ICudaEngine*		m_engine	= nullptr;					// 
		IExecutionContext*	m_context	= nullptr;					// 
		cudaStream_t		m_stream	= nullptr;					// 
		std::string			m_engine_name = "";
        void*				buffers[2];								//

		float				input[BATCH_SIZE * 3 * yolov5::INPUT_H * yolov5::INPUT_W];		// 图片数据
		float				output[BATCH_SIZE * yolov5::OUTPUT_SIZE];				// 输出数据
		int					inputIndex;
		int					outputIndex;
	};
}
#endif // YOLOV5_H

