/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <vector>

#ifdef _MSC_VER
#define NOMINMAX
#endif

#include <algorithm>
#include <cmath>
#include <string>

#include <pthread.h>
#include <sys/time.h>
#include <sched.h>
#include <sys/resource.h>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include "common.h"
#include "tengine_c_api.h"

#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1

#define VXDEVICE  "VX"

#define MODEL_COLS 416
#define MODEL_ROWS 416

#define DEBUG_OPTION 0

#define CAMERA_WEIGHT 1920
#define CAMERA_HIGHT 1080

static unsigned int tmpVal;

using namespace std;

typedef struct
{
    float x, y, w, h;
} box;

typedef struct
{
    box bbox;
    int classes;
    float* prob;
    float* mask;
    float objectness;
    int sort_class;
} detection;

typedef struct layer
{
    int layer_type;
    int batch;
    int total;
    int n, c, h, w;
    int out_n, out_c, out_h, out_w;
    int classes;
    int inputs;
    int outputs;
    int* mask;
    float* biases;
    float* output;
    int coords;
} layer;

const int classes = 80;
const float thresh = 0.5;
const float hier_thresh = 0.5;
const float nms = 0.45;
const int numBBoxes = 5;
const int relative = 1;
const int yolov3_numAnchors = 6;
const int yolov2_numAnchors = 5;

//coco datasheet
const char *coco_names[80] = {"person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

// yolov3
float biases[18] = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};
// tiny
float biases_tiny[12] = {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
// yolov2
float biases_yolov2[10] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};

pthread_mutex_t mutex4q;

char* model_file = nullptr;

int repeat_count = DEFAULT_REPEAT_COUNT;
int num_thread = DEFAULT_THREAD_COUNT;
char *video_device = NULL;

void get_input_data_cv(const cv::Mat& sample, uint8_t* input_data, int img_h, int img_w, float input_scale, int zero_point, int swapRB = 0)
{
	cv::Mat img;
	if (sample.channels() == 4)
	{
		cv::cvtColor(sample, img, cv::COLOR_BGRA2BGR);
	}
	else if (sample.channels() == 1)
	{
		cv::cvtColor(sample, img, cv::COLOR_GRAY2BGR);
	}
	else if (sample.channels() == 3 && swapRB == 1)
	{
		cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);
	}
	else
	{
		img = sample;
	}

	cv::Mat img2(img_h,img_w,CV_8UC3,cv::Scalar(0,0,0));
	int h,w;

	if((float)img.cols/(float)img_w > (float)img.rows/(float)img_h){
		h=1.0*img_w/img.cols*img.rows;
		w=img_w;
		cv::resize(img, img, cv::Size(w,h));
	}else{
		w=1.0*img_h/img.rows*img.cols;
		h=img_h;
		cv::resize(img, img, cv::Size(w,h));
	}

	int top = (img_h - h)/2;
	int bat = (img_h - h + 1)/2;
	int left = (img_w - w)/2;
	int right = (img_w - w + 1)/2;


	cv::copyMakeBorder(img,img2,top,bat,left,right,cv::BORDER_CONSTANT,cv::Scalar(0,0,0));

	uint8_t* img_data = img2.data;
	int hw = img_h * img_w;

	for (int h = 0; h < img_h; h++)
	{
		for (int w = 0; w < img_w; w++)
		{
			for (int c = 0; c < 3; c++)
			{
				int udata = *img_data;
				if (udata > 255)
					udata = 255;
				else if (udata < 0)
					udata = 0;
				input_data[c * hw + h * img_w + w] = udata;
				img_data++;
			}
		}
	}
}


layer make_darknet_layer(int batch, int w, int h, int net_w, int net_h, int n, int total, int classes, int layer_type)
{
    layer l = {0};
    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n * (classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.inputs = l.w * l.h * l.c;

    l.biases = ( float* )calloc(total * 2, sizeof(float));
    if (layer_type == 0)
    {
        l.mask = ( int* )calloc(n, sizeof(int));
        if (9 == total)
        {
            for (int i = 0; i < total * 2; ++i)
            {
                l.biases[i] = biases[i];
            }
            if (l.w == net_w / 32)
            {
                int j = 6;
                for (int i = 0; i < l.n; ++i)
                    l.mask[i] = j++;
            }
            if (l.w == net_w / 16)
            {
                int j = 3;
                for (int i = 0; i < l.n; ++i)
                    l.mask[i] = j++;
            }
            if (l.w == net_w / 8)
            {
                int j = 0;
                for (int i = 0; i < l.n; ++i)
                    l.mask[i] = j++;
            }
        }
        if (6 == total)
        {
            for (int i = 0; i < total * 2; ++i)
            {
                l.biases[i] = biases_tiny[i];
            }
            if (l.w == net_w / 32)
            {
                int j = 3;
                for (int i = 0; i < l.n; ++i)
                    l.mask[i] = j++;
            }
            if (l.w == net_w / 16)
            {
                int j = 0;
                for (int i = 0; i < l.n; ++i)
                    l.mask[i] = j++;
            }
        }
    }
    else if (1 == layer_type)
    {
        l.coords = 4;
        for (int i = 0; i < total * 2; ++i)
        {
            l.biases[i] = biases_yolov2[i];
        }
    }
    l.layer_type = layer_type;
    l.outputs = l.inputs;
    l.output = ( float* )calloc(batch * l.outputs, sizeof(float));

    return l;
}

void free_darknet_layer(layer l)
{
    if (NULL != l.biases)
    {
        free(l.biases);
        l.biases = NULL;
    }
    if (NULL != l.mask)
    {
        free(l.mask);
        l.mask = NULL;
    }
    if (NULL != l.output)
    {
        free(l.output);
        l.output = NULL;
    }
}

static int entry_index(layer l, int batch, int location, int entry)
{
    int n = location / (l.w * l.h);
    int loc = location % (l.w * l.h);
    return batch * l.outputs + n * l.w * l.h * (4 + l.classes + 1) + entry * l.w * l.h + loc;
}

void logistic_cpu(float* input, int size)
{
    for (int i = 0; i < size; ++i)
    {
        input[i] = 1.f / (1.f + expf(-input[i]));
    }
}

void forward_darknet_layer_cpu(const float* input, layer l)
{
    memcpy(( void* )l.output, ( void* )input, sizeof(float) * l.inputs * l.batch);
    if (0 == l.layer_type)
    {
        for (int b = 0; b < l.batch; ++b)
        {
            for (int n = 0; n < l.n; ++n)
            {
                int index = entry_index(l, b, n * l.w * l.h, 0);
                logistic_cpu(l.output + index, 2 * l.w * l.h);
                index = entry_index(l, b, n * l.w * l.h, 4);
                logistic_cpu(l.output + index, (1 + l.classes) * l.w * l.h);
            }
        }
    }
}

int yolo_num_detections(layer l, float thresh)
{
    int i, n, b;
    int count = 0;
    for (b = 0; b < l.batch; ++b)
    {
        for (i = 0; i < l.w * l.h; ++i)
        {
            for (n = 0; n < l.n; ++n)
            {
                int obj_index = entry_index(l, b, n * l.w * l.h + i, 4);
                if (l.output[obj_index] > thresh)
                    ++count;
            }
        }
    }
    return count;
}

int num_detections(vector<layer> layers_params, float thresh)
{
    int i;
    int s = 0;
    for (i = 0; i < ( int )layers_params.size(); ++i)
    {
        layer l = layers_params[i];
        if (0 == l.layer_type)
            s += yolo_num_detections(l, thresh);
        else if (1 == l.layer_type)
            s += l.w * l.h * l.n;
    }
#if DEBUG_OPTION
    fprintf(stderr, "%s,%d\n", __func__, s);
#endif
    return s;
}

detection* make_network_boxes(vector<layer> layers_params, float thresh, int* num)
{
    layer l = layers_params[0];
    int i;
    int nboxes = num_detections(layers_params, thresh);
    if (num)
        *num = nboxes;
    detection* dets = ( detection* )calloc(nboxes, sizeof(detection));

    for (i = 0; i < nboxes; ++i)
    {
        dets[i].prob = ( float* )calloc(l.classes, sizeof(float));
    }
    return dets;
}

void correct_yolo_boxes(detection* dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w = 0;
    int new_h = 0;
    if ((( float )netw / w) < (( float )neth / h))
    {
        new_w = netw;
        new_h = (h * netw) / w;
    }
    else
    {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    for (i = 0; i < n; ++i)
    {
        box b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / (( float )new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / (( float )new_h / neth);
        b.w *= ( float )netw / new_w;
        b.h *= ( float )neth / new_h;
        if (!relative)
        {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

box get_yolo_box(float* x, float* biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0 * stride]) / lw;
    b.y = (j + x[index + 1 * stride]) / lh;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;

    return b;
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int* map, int relative,
                        detection* dets)
{
    int i, j, n, b;
    float* predictions = l.output;
    int count = 0;
    for (b = 0; b < l.batch; ++b)
    {
        for (i = 0; i < l.w * l.h; ++i)
        {
            int row = i / l.w;
            int col = i % l.w;
            for (n = 0; n < l.n; ++n)
            {
                int obj_index = entry_index(l, b, n * l.w * l.h + i, 4);
                float objectness = predictions[obj_index];
                if (objectness <= thresh)
                    continue;
                int box_index = entry_index(l, b, n * l.w * l.h + i, 0);

                dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw,
                                                neth, l.w * l.h);
                dets[count].objectness = objectness;
                dets[count].classes = l.classes;
                for (j = 0; j < l.classes; ++j)
                {
                    int class_index = entry_index(l, b, n * l.w * l.h + i, 4 + 1 + j);
                    float prob = objectness * predictions[class_index];
                    dets[count].prob[j] = (prob > thresh) ? prob : 0;
                }
                ++count;
            }
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}

void correct_region_boxes(detection* dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w = 0;
    int new_h = 0;
    if ((( float )netw / w) < (( float )neth / h))
    {
        new_w = netw;
        new_h = (h * netw) / w;
    }
    else
    {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    for (i = 0; i < n; ++i)
    {
        box b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / (( float )new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / (( float )new_h / neth);
        b.w *= ( float )netw / new_w;
        b.h *= ( float )neth / new_h;
        if (!relative)
        {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

box get_region_box(float* x, float* biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0 * stride]) / w;
    b.y = (j + x[index + 1 * stride]) / h;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    return b;
}

void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int* map, float tree_thresh,
                           int relative, detection* dets)
{
    int i, j, n;
    float* predictions = l.output;

    for (i = 0; i < l.w * l.h; ++i)
    {
        int row = i / l.w;
        int col = i % l.w;
        for (n = 0; n < l.n; ++n)
        {
            int index = n * l.w * l.h + i;
            for (j = 0; j < l.classes; ++j)
            {
                dets[index].prob[j] = 0;
            }
            int obj_index = entry_index(l, 0, n * l.w * l.h + i, l.coords);
            int box_index = entry_index(l, 0, n * l.w * l.h + i, 0);
            int mask_index = entry_index(l, 0, n * l.w * l.h + i, 4);
            float scale = predictions[obj_index];
            dets[index].bbox = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h, l.w * l.h);
            dets[index].objectness = scale > thresh ? scale : 0;
            if (dets[index].mask)
            {
                for (j = 0; j < l.coords - 4; ++j)
                {
                    dets[index].mask[j] = l.output[mask_index + j * l.w * l.h];
                }
            }
            // int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1);
            if (dets[index].objectness)
            {
                for (j = 0; j < l.classes; ++j)
                {
                    int class_index = entry_index(l, 0, n * l.w * l.h + i, l.coords + 1 + j);
                    float prob = scale * predictions[class_index];
                    dets[index].prob[j] = (prob > thresh) ? prob : 0;
                }
            }
        }
    }
    correct_region_boxes(dets, l.w * l.h * l.n, w, h, netw, neth, relative);
}

void fill_network_boxes(vector<layer> layers_params, int img_w, int img_h, int net_w, int net_h, float thresh,
                        float hier, int* map, int relative, detection* dets)
{
    int j;
    for (j = 0; j < ( int )layers_params.size(); ++j)
    {
        layer l = layers_params[j];
        if (0 == l.layer_type)
        {
            int count = get_yolo_detections(l, img_w, img_h, net_w, net_h, thresh, map, relative, dets);
            dets += count;
        }
        else
        {
            get_region_detections(l, img_w, img_h, net_w, net_h, thresh, map, hier, relative, dets);
            dets += l.w * l.h * l.n;
        }
    }
}

detection* get_network_boxes(vector<layer> layers_params, int img_w, int img_h, int net_w, int net_h, float thresh,
                             float hier, int* map, int relative, int* num)
{
    // make network boxes
    detection* dets = make_network_boxes(layers_params, thresh, num);

    // fill network boxes
    fill_network_boxes(layers_params, img_w, img_h, net_w, net_h, thresh, hier, map, relative, dets);
    return dets;
}

// release detection memory
void free_detections(detection* dets, int nboxes)
{
    int i;
    for (i = 0; i < nboxes; ++i)
    {
        free(dets[i].prob);
    }
    free(dets);
}

int nms_comparator(const void* pa, const void* pb)
{
    detection a = *( detection* )pa;
    detection b = *( detection* )pb;
    float diff = 0;
    if (b.sort_class >= 0)
    {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    }
    else
    {
        diff = a.objectness - b.objectness;
    }
    if (diff < 0)
        return 1;
    else if (diff > 0)
        return -1;
    return 0;
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0)
        return 0;
    float area = w * h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b) / box_union(a, b);
}

void do_nms_sort(detection* dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i)
    {
        if (dets[i].objectness == 0)
        {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k)
    {
        for (i = 0; i < total; ++i)
        {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator);
        for (i = 0; i < total; ++i)
        {
            if (dets[i].prob[k] == 0)
                continue;
            box a = dets[i].bbox;
            for (j = i + 1; j < total; ++j)
            {
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh)
                {
                    dets[j].prob[k] = 0;
				}
            }
        }
    }
}


void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-d video_device] [-r repeat_count] [-t thread_count]\n");
}

int check_file_exist(const char* file_name)
{
    FILE* fp = fopen(file_name, "r");
    if (!fp)
    {
        fprintf(stderr, "Input file not existed: %s\n", file_name);
        return 0;
    }
    fclose(fp);
    return 1;
}


static cv::Scalar obj_id_to_color(int obj_id) {
	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
	int const offset = obj_id * 123457 % 6;
	int const color_scale = 150 + (obj_id * 123457) % 100;
	cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
	color *= color_scale;
	return color;
}



void postpress_graph(const cv::Mat& sample, graph_t graph, int output_node_num,int net_w, int net_h,int numBBoxes,int total_numAnchors,int layer_type,char *window_name)
{
	cv::Mat frame = sample;

	vector<layer> layers_params;
	layers_params.clear();

	for (int i = 0; i < output_node_num; ++i)
	{
		tensor_t out_tensor = get_graph_output_tensor(graph, i, 0);    //"detection_out"
		int out_dim[4];
		get_tensor_shape(out_tensor, out_dim, 4);
		layer l_params;
		int out_w = out_dim[3];
		int out_h = out_dim[2];
		l_params = make_darknet_layer(1, out_w, out_h, net_w, net_h, numBBoxes, total_numAnchors, classes, layer_type);
		layers_params.push_back(l_params);

		/* dequant output data */
		float output_scale = 0.f;
		int output_zero_point = 0;
		get_tensor_quant_param(out_tensor, &output_scale, &output_zero_point, 1);
		int count = get_tensor_buffer_size(out_tensor) / sizeof(uint8_t);
		uint8_t* data_uint8 = ( uint8_t* )get_tensor_buffer(out_tensor);
		float* data_fp32 = ( float* )malloc(sizeof(float) * count);
		for (int c = 0; c < count; c++)
		{
			data_fp32[c] = (( float )data_uint8[c] - ( float )output_zero_point) * output_scale;
		}

		forward_darknet_layer_cpu(data_fp32, l_params);

		free(data_fp32);
	}

	int nboxes = 0;
	// get network boxes
	detection* dets =
		get_network_boxes(layers_params, frame.cols, frame.rows, net_w, net_h, thresh, hier_thresh, 0, relative, &nboxes);

	if (nms != 0)
	{
		do_nms_sort(dets, nboxes, classes, nms);
	}

	int i, j;
	for (i = 0; i < nboxes; ++i)
	{
		int cls = -1;
		char cvTest[64];
		for (j = 0; j < classes; ++j)
		{
			if (dets[i].prob[j] > 0.5)
			{
				if (cls < 0)
				{
					cls = j;
				}
#if DEBUG_OPTION
				fprintf(stderr, "%d: %.0f%%\n", cls, dets[i].prob[j] * 100);
				sprintf(cvTest,"%s:%.0f%%",coco_names[cls],dets[i].prob[j]*100);
#endif
				sprintf(cvTest,"%s",coco_names[cls]);
			}

		}
		if (cls >= 0)
		{
			box b = dets[i].bbox;
			int left = (b.x - b.w / 2.) * frame.cols;
			int top = (b.y - b.h / 2.) * frame.rows;
			if (top < 30) {
				top = 30;
				left +=10;
			}

#if DEBUG_OPTION
			fprintf(stderr, "left = %d,top = %d\n", left, top);
#endif
			cv::Rect rect(left, top, b.w*frame.cols, b.h*frame.rows);
			cv::Rect rect1(left, top-20, 100, 20);

			int baseline;
			cv::Size text_size = cv::getTextSize(cvTest, cv::FONT_HERSHEY_COMPLEX,0.5,1,&baseline);

			cv::rectangle(frame,rect,obj_id_to_color(cls),1,8,0);
			cv::rectangle(frame,rect1,obj_id_to_color(cls),-1);
			cv::putText(frame,cvTest,cvPoint(left+((100-text_size.width)/2),top-5),cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(0,0,0),1);
		}


		if (dets[i].mask)
			free(dets[i].mask);
		if (dets[i].prob)
			free(dets[i].prob);
	}
	free(dets);

	for (int i = 0; i < (int)layers_params.size(); i++)
	{
		layer l = layers_params[i];
		if (l.output)
			free(l.output);
		if (l.biases)
			free(l.biases);
		if (l.mask)
			free(l.mask);
	}

	cv::imshow(window_name, frame);
	cv::waitKey(1);
	frame.release();
}

static void *thread_camera(void *parameter)
{

	int layer_type = 0;
	int numBBoxes = 3;
	int total_numAnchors = 9;
	int net_w = MODEL_COLS;
	int net_h = MODEL_ROWS;

	struct timeval tmsStart, tmsEnd;

	char *window_name = (char *)"CameraWindow";

	string str = video_device;
	string res=str.substr(10);
	
	cv::VideoCapture cap(stoi(res));
	cap.set(CV_CAP_PROP_FRAME_WIDTH, CAMERA_WEIGHT);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, CAMERA_HIGHT);
	
	if (!cap.isOpened()) {
		cout << "capture device failed to open!" << endl;
		cap.release();
	}

	setpriority(PRIO_PROCESS, pthread_self(), -15);

	cv::namedWindow(window_name);


	/* inital tengine */
	init_tengine();
	fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

	/* load npu backend */
	if (load_tengine_plugin(VXDEVICE, "libvxplugin.so", "vx_plugin_init") < 0)
	{
		fprintf(stderr, "Load vx plugin failed.\n");
		exit(-1);
	}

	/* set inference context with npu */
	context_t vx_context = create_context("vx", 1);
	add_context_device(vx_context, VXDEVICE);

	/* create graph, load tengine model xxx.tmfile */
	graph_t graph = create_graph(vx_context, "tengine", model_file);
	if (graph == nullptr)
	{
		fprintf(stderr, "Create graph failed.\n");
		fprintf(stderr, "errno: %d \n", get_tengine_errno());
		exit(-1);
	}


	/* set the input shape to initial the graph, and prerun graph to infer shape */
	int img_size = net_h * net_w * 3;
	int dims[] = {1, 3, net_h, net_w};    // nchw

	std::vector<uint8_t> input_data(img_size);

	tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
	if (input_tensor == nullptr)
	{
		fprintf(stderr, "Get input tensor failed\n");
		exit(-1);
	}

	if (set_tensor_shape(input_tensor, dims, 4) < 0)
	{
		fprintf(stderr, "Set input tensor shape failed\n");
		exit(-1);
	}

	if (prerun_graph(graph) < 0)
	{
		fprintf(stderr, "Prerun graph failed\n");
		exit(-1);
	}

	float input_scale = 0.f;
	int input_zero_point = 0;


	get_tensor_quant_param(input_tensor, &input_scale, &input_zero_point, 1);
#if DEBUG_OPTION
	cout << "input data:" << input_scale << input_zero_point << endl;
#endif
	if (set_tensor_buffer(input_tensor, input_data.data(), img_size) < 0)
	{
		fprintf(stderr, "Set input tensor buffer failed\n");
		exit(-1);
	}

	int output_node_num = get_graph_output_node_number(graph);

	while(1){
		cv::Mat frame(CAMERA_HIGHT,CAMERA_WEIGHT,CV_8UC3);
		pthread_mutex_lock(&mutex4q);
		if (!cap.read(frame)) {
			pthread_mutex_unlock(&mutex4q);
			cout<<"Capture read error"<<std::endl;
			break;
		}

		pthread_mutex_unlock(&mutex4q);

//#if DEBUG_OPTION
		gettimeofday(&tmsStart, 0);
//#endif

		get_input_data_cv(frame,input_data.data(),net_w,net_h,input_scale, input_zero_point,0);

		/* run graph */
		double min_time = __DBL_MAX__;
		double max_time = -__DBL_MAX__;
		double total_time = 0.;
		for (int i = 0; i < repeat_count; i++)
		{
			double start = get_current_time();
			if (run_graph(graph, 1) < 0)
			{
				fprintf(stderr, "Run graph failed\n");
				exit(-1);
			}
			double end = get_current_time();
			double cur = end - start;
			total_time += cur;
			min_time = std::min(min_time, cur);
			max_time = std::max(max_time, cur);
		}
#if DEBUG_OPTION
		fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count,
				num_thread, total_time / repeat_count, max_time, min_time);
		fprintf(stderr, "--------------------------------------\n");
#endif	
	
		/* process the detection result */
		int output_node_num = get_graph_output_node_number(graph);
	
		postpress_graph(frame,graph,output_node_num,net_w,net_h,numBBoxes,total_numAnchors,layer_type,window_name);
//#if DEBUG_OPTION
		gettimeofday(&tmsEnd, 0);
		tmpVal = 1000 * (tmsEnd.tv_sec - tmsStart.tv_sec) + (tmsEnd.tv_usec - tmsStart.tv_usec) / 1000;
		printf("FPS:%d\n",1000/(tmpVal+8));
//#endif
		frame.release();
	}
	/* release tengine */
	for (int i = 0; i < output_node_num; ++i)
	{
		tensor_t out_tensor = get_graph_output_tensor(graph, i, 0);
		release_graph_tensor(out_tensor);
	}


	release_graph_tensor(input_tensor);
	postrun_graph(graph);
	destroy_graph(graph);
	release_tengine();
	return 0;
}

int main(int argc, char* argv[])
{
	int i=0;
	pthread_t tid[2];

    int res;
    while ((res = getopt(argc, argv, "m:d:r:t:h:")) != -1)
    {
        switch (res)
        {
            case 'm':
                model_file = optarg;
				break;
			case 'd':
				video_device = optarg;
                break;
            case 'r':
                repeat_count = std::strtoul(optarg, nullptr, 10);
                break;
            case 't':
                num_thread = std::strtoul(optarg, nullptr, 10);
                break;
            case 'h':
                show_usage();
                return 0;
            default:
                break;
        }
    }

    /* check files */
    if (nullptr == model_file)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file)){
        return -1;
	}

	pthread_mutex_init(&mutex4q,NULL);

	if (0 != pthread_create(&tid[0], NULL, thread_camera, NULL)) {
		fprintf(stderr, "Couldn't create thread func\n");
		return -1;
	}

	while(1)
	{
		for(i=0; i<(int)sizeof(tid)/(int)sizeof(tid[0]); i++)
		{
			pthread_join(tid[i], NULL);
		}
		sleep(1);
	}

    return 0;
}
