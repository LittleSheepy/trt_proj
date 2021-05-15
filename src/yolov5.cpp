/*******************************************************************
** 文件名:	AIServer.cpp
** 版  权:	(C) littlesheepy 2020 - All Rights Reserved
** 创建人:	littlesheepy
** 日  期:	12/02/2020
** 版  本:	1.0
** 描  述:
** 应  用:
**************************** 修改记录 ******************************
** 修改人:
** 日  期:
** 描  述:
********************************************************************/
#include <chrono>
#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include "yolov5.h"
#include "common.h"
#include "utils.h"
//#include "cfg.h"
using namespace nvinfer1;
using namespace Yolo;

//extern std::string IMG_NAME;
namespace DeepBlueDet {
	YOLOV5::YOLOV5()
	{

	}
	YOLOV5::~YOLOV5()
	{
		// Release stream and buffers
		cudaStreamDestroy(this->m_stream);
		CHECK(cudaFree(this->buffers[inputIndex]));
		CHECK(cudaFree(this->buffers[outputIndex]));
		// Destroy the engine
		m_context->destroy();
		m_engine->destroy();
		m_runtime->destroy();
	}

	void YOLOV5::Init()
	{
		cudaSetDevice(DEVICE);
		// cfg
		//Config cfg("cfg.txt");

		// create a model using the API directly and serialize it to a stream
		char *trtModelStream{ nullptr };
		size_t size{ 0 };
		std::string engine_name = STR2(NET);
		engine_name = "DeepBlueDet_" + engine_name + ".engine";
		std::ifstream file(engine_name, std::ios::binary);
		if (file.good()) {
			file.seekg(0, file.end);
			size = file.tellg();
			file.seekg(0, file.beg);
			trtModelStream = new char[size];
			assert(trtModelStream);
			file.read(trtModelStream, size);
			file.close();
		}
		m_runtime = createInferRuntime(gLogger);
		assert(m_runtime != nullptr);
		this->m_engine = m_runtime->deserializeCudaEngine(trtModelStream, size);
		assert(this->m_engine != nullptr);
		this->m_context = m_engine->createExecutionContext();
		assert(this->m_context != nullptr);
		delete[] trtModelStream;
		assert(m_engine->getNbBindings() == 2);
		// In order to bind the buffers, we need to know the names of the input and output tensors.
		// Note that indices are guaranteed to be less than IEngine::getNbBindings()
		inputIndex = m_engine->getBindingIndex(INPUT_BLOB_NAME);
		outputIndex = m_engine->getBindingIndex(OUTPUT_BLOB_NAME);
		assert(inputIndex == 0);
		assert(outputIndex == 1);
		// Create GPU buffers on device
		buffers[2];
		CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
		CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
		// Create stream
		CHECK(cudaStreamCreate(&m_stream));
	}


    void YOLOV5::doInference() {
        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CHECK(cudaMemcpyAsync(buffers[0], this->input, BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, m_stream));
        this->m_context->enqueue(BATCH_SIZE, this->buffers, m_stream, nullptr);
        CHECK(cudaMemcpyAsync(this->output, this->buffers[1], BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
        cudaStreamSynchronize(m_stream);
    }
    int cmpr(Yolo::Detection& x1, Yolo::Detection& x2){
        return x1.bbox[0] < x2.bbox[0];
    }
	std::vector<Yolo::Detection> YOLOV5::predict(cv::Mat& img) {
		float* data = this->input;
		cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox BGR to RGB
		int i = 0;
		for (int row = 0; row < INPUT_H; ++row) {
			uchar* uc_pixel = pr_img.data + row * pr_img.step;
			for (int col = 0; col < INPUT_W; ++col) {
				data[i] = (float)uc_pixel[2] / 255.0;
				data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
				data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
				uc_pixel += 3;
				++i;
			}
		}
		// Run inference
		auto start = std::chrono::system_clock::now();
		doInference();
		auto end = std::chrono::system_clock::now();
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
		std::vector<std::vector<Yolo::Detection>> batch_res(1);
		for (int b = 0; b < 1; b++) {
			auto& res = batch_res[b];
			nms_all(res, &this->output[0], CONF_THRESH, NMS_THRESH);
		}
        sort(batch_res[0].begin(), batch_res[0].end(), cmpr);
		return batch_res[0];
	}

    std::vector<std::string> YOLOV5::predict_str(cv::Mat& img) {
        auto res = predict(img);
        std::vector<std::string> rec_result_list;
        std::cout << "size:" <<res.size() << std::endl;
        std::string str = "";
		std::string label_list[] = { "0","1","2", "3","4", "5","6", "7","8", "9",
							 "A", "b","C", "d","E", "F","J","h","P","t",
							 "L", "U", "u","o", "_" };
		cv::Mat img_clone;
		img_clone = img.clone();
        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect(img_clone, res[j].bbox);
            cv::rectangle(img_clone, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
			//cv::putText(img_clone, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
			cv::putText(img_clone, std::string(label_list[(int)res[j].class_id]), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
			if (j > 0){
                if ( (res[j].bbox[0] - res[j-1].bbox[0]) > 1.5 * res[j].bbox[2]){
                    str.append(std::string(" "));
                }
            }
            str.append(std::string(label_list[(int)res[j].class_id]));
        }
		//cv::imwrite("../images/" + IMG_NAME, img_clone);
        std::cout << "str:"<<str<<std::endl;
        rec_result_list.push_back(str);

		return rec_result_list;
    }
}
