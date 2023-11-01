
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include <stdexcept> 

#include "samEncoder.h"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"
#include <assert.h>
#include <unordered_map>
#include "cuda_runtime.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
typedef unsigned char uint8;


class SamDecoder
{
public:
    SamDecoder(const std::string& encoder_engine_file, const std::string& decoder_engine_file);
    ~SamDecoder();
  
    const cv::Mat getMask(const cv::Mat &image, const float x, const float y);

private:
	class Logger : public nvinfer1::ILogger
	{
	public:
		void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept
		{
			switch (severity)
			{
			case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
				std::cerr << "kINTERNAL_ERROR: " << msg << std::endl;
				break;
			case nvinfer1::ILogger::Severity::kERROR:
				std::cerr << "kERROR: " << msg << std::endl;
				break;
			case nvinfer1::ILogger::Severity::kWARNING:
				std::cerr << "kWARNING: " << msg << std::endl;
				break;
			case nvinfer1::ILogger::Severity::kINFO:
				std::cerr << "kINFO: " << msg << std::endl;
				break;
			case nvinfer1::ILogger::Severity::kVERBOSE:
				std::cerr << "kVERBOSE: " << msg << std::endl;
				break;
			default:
				break;
			}
		}
	};

private:
    
    SamDecoder(const SamDecoder &);

    bool IsExists(const std::string& file)
	{
		std::fstream f(file.c_str());
		return f.is_open();
	}

	void SaveRtModel(const std::string& path)
	{
		std::ofstream outfile(path, std::ios_base::out | std::ios_base::binary);
		outfile.write((const char*)m_gie_model_stream->data(), m_gie_model_stream->size());
		outfile.close();
	}

    bool LoadModel();
	bool LoadOnnxModel();
	bool LoadTRTModel();
	bool deserializeCudaEngine(const void* blob, std::size_t size);
	bool mallocInputOutput();

    void Forward();
	const cv::Mat postProcess();
	void preprocess(const std::vector<float> &embedding, float x, float y);

    SamEncoder sam_encoder; 

	const std::string m_encoder_engine_file;
	const std::string m_decoder_engine_file;

	Logger m_loger;
	cudaStream_t m_stream;

	nvinfer1::IRuntime* m_runtime;
	nvinfer1::ICudaEngine* m_engine;
	nvinfer1::IExecutionContext* m_context;
	nvinfer1::IHostMemory* m_gie_model_stream{ nullptr };

	std::vector<void*> m_buffers;

	float* d_input_image_embeddings_ptr;
	float* d_input_point_coords_ptr;
	float* d_input_point_labels_ptr;
	float* d_input_mask_input_ptr;
	float* d_input_has_mask_input_ptr;


	float* d_output_low_res_masks_ptr;
	float* d_output_iou_predictions_ptr;
	float* d_output_masks_ptr;
	
	float* h_output_masks_ptr;

    int m_max_batchsize = 1;
    int m_batchsize = 1;

    SamDecoder &operator=(const SamDecoder &);
    static SamDecoder *instance;

};
