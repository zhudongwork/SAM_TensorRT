#include "samDecoder.h"
#include <chrono>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{


	const std::string engine_encoder_file = "../models/mobile_sam_encoder.engine";
	const std::string engine_decoder_file = "../models/mobile_sam_decoder.engine";
    
    SamDecoder sam(engine_encoder_file, engine_decoder_file);

    float x = 514.0;
    float y = 391.0;
    cv::Mat frame = cv::imread("../dog.jpg");
    cv::resize(frame, frame, cv::Size(1024, 1024));
   

    cv::Rect init_box = cv::selectROI("dst", frame, false, false);
    auto center_x = init_box.x + init_box.width / 2;
    auto center_y = init_box.y + init_box.height / 2;
    cv::Mat outputMaskSam;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    outputMaskSam = sam.getMask(frame, center_x, center_y);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Mobile SAM run time taken : " << duration.count() << " ms" << std::endl;
    
    cv::cvtColor(outputMaskSam, outputMaskSam, cv::COLOR_GRAY2BGR);
    cv::Mat dst;
    cv::addWeighted(frame, 0.7, outputMaskSam, 0.3, 0, dst);
    cv::imshow("dst", dst);
    cv::waitKey(0);
    
    return 0;

}
