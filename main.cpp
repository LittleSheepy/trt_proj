#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include <opencv2/opencv.hpp>
//#include "logging.h"
#include "common.h"
#include "cuda_utils.h"
#include "yolov4.h"
#include "yolov4cfg.h"
#include "yolov5.h"
#include "yolov5cfg.h"

using namespace nvinfer1;
using namespace std;
using namespace ObjDet;
using namespace yolov5;

//static nvinfer1::PluginRegistrar<YoloPluginCreator> pluginRegistrarYoloPluginCreator{};
bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& img_dir) {
	//if (argc < 4) return false;
	if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
		wts = std::string(argv[2]);
		engine = std::string(argv[3]);
		auto net = std::string(argv[4]);
		if (net[0] == 's') {
			gd = 0.33;
			gw = 0.50;
		}
		else if (net[0] == 'm') {
			gd = 0.67;
			gw = 0.75;
		}
		else if (net[0] == 'l') {
			gd = 1.0;
			gw = 1.0;
		}
		else if (net[0] == 'x') {
			gd = 1.33;
			gw = 1.25;
		}
		else if (net[0] == 'c' && argc == 7) {
			gd = atof(argv[5]);
			gw = atof(argv[6]);
		}
		else {
			return false;
		}
		if (net.size() == 2 && net[1] == '6') {
			is_p6 = true;
		}
	}
	else if (std::string(argv[1]) == "-d" && argc >= 3) {
		engine = std::string(argv[2]);
		if (argc > 3) {
			img_dir = std::string(argv[3]);
		}
	}
	else {
		return false;
	}
	return true;
}

int yolov4_main(int argc, char** argv) {
	YOLOV4 * yolov4 = new YOLOV4();
	yolov4->Init();
	yolov4->LoadEngine();

	cv::VideoCapture capture;
	bool res = capture.open(0);
	while (true) {
		cv::Mat img;
		bool red_res = capture.read(img);
		auto res = yolov4->predict(img);
		cout << img.cols << "," << img.rows << endl;

		cv::Mat img_lab(img.rows, img.cols + 200, CV_8UC3);
		//auto& res = batch_res[0];
		std::cout << "size:" << res.size() << std::endl;
		std::string str = "";
		std::string label_list[] = { "0","1","2", "3","4", "5","6", "7","8", "9",
							 "A", "b","C", "d","E", "F","J","h","P","t",
							 "L", "U", "u","o", "_" };
		for (size_t j = 0; j < res.size(); j++) {
			cv::Rect r = get_rect(img, res[j].bbox);
			cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
			cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
			//cv::putText(img, std::string(label_list[(int)res[j].class_id]), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
			//str.append(std::string(label_list[(int)res[j].class_id]));
		}
		img.copyTo(img_lab(cv::Rect(0, 0, img.cols, img.rows)));

		cv::putText(img_lab, str, cv::Point(650, 200), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0xFF, 0xFF, 0xFF), 4);
		std::cout << "str:" << str << std::endl;
		cv::imshow("ÉãÏñÍ· ", img_lab);
		cv::waitKey(30);
	}


	return 1;
}

int yolov5_1(int argc, char** argv) {
	std::string wts_name = "";
	std::string engine_name = "";
	bool is_p6 = false;
	float gd = 0.0f, gw = 0.0f;
	std::string img_dir = "";
	if (!parse_args(argc, argv, wts_name, engine_name, is_p6, gd, gw, img_dir)) {
		std::cerr << "arguments not right!" << std::endl;
		std::cerr << "./yolov5 -s [.wts] [.engine] [s/m/l/x/s6/m6/l6/x6 or c/c6 gd gw]  // serialize model to plan file" << std::endl;
		std::cerr << "./yolov5 -d [.engine] ../samples  // deserialize plan file and run inference" << std::endl;
		return -1;
	}
	YOLOV5 * yolov5 = new YOLOV5();
	yolov5->Init(engine_name);
	if (!wts_name.empty()) {
		yolov5->serialize(wts_name, engine_name, is_p6, gd, gw);
		return 0;
	}
	yolov5->LoadEngine();
	if (!img_dir.empty()) {
		std::vector<std::string> file_names;
		if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
			std::cerr << "read_files_in_dir failed." << std::endl;
			return -1;
		}
		for (int f = 0; f < (int)file_names.size(); f++) {

			cv::Mat img = cv::imread(img_dir + "/" + file_names[f]);
			if (img.empty()) continue;
			auto res = yolov5->predict(img);
			for (size_t j = 0; j < res.size(); j++) {
				cv::Rect r = get_rect(img, res[j].bbox);
				cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
				cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
			}
			cv::imwrite("_" + file_names[f], img);
		}
	}
	else {
		cv::VideoCapture capture;
		bool res = capture.open(0);
		while (true) {
			cv::Mat img;
			bool red_res = capture.read(img);
			auto res = yolov5->predict(img);
			cout << img.cols << "," << img.rows << endl;

			cv::Mat img_lab(img.rows, img.cols + 200, CV_8UC3);
			//auto& res = batch_res[0];
			std::cout << "size:" << res.size() << std::endl;
			std::string str = "";
			std::string label_list[] = { "0","1","2", "3","4", "5","6", "7","8", "9",
								 "A", "b","C", "d","E", "F","J","h","P","t",
								 "L", "U", "u","o", "_" };
			for (size_t j = 0; j < res.size(); j++) {
				cv::Rect r = get_rect(img, res[j].bbox);
				cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
				cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
				//cv::putText(img, std::string(label_list[(int)res[j].class_id]), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
				//str.append(std::string(label_list[(int)res[j].class_id]));
			}
			img.copyTo(img_lab(cv::Rect(0, 0, img.cols, img.rows)));

			cv::putText(img_lab, str, cv::Point(650, 200), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0xFF, 0xFF, 0xFF), 4);
			std::cout << "str:" << str << std::endl;
			cv::imshow("ÉãÏñÍ· ", img_lab);
			cv::waitKey(30);
		}
	}

}

int main(int argc, char** argv) {
	yolov4_main(argc, argv);
}
