#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <iomanip>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include "nn_sdk.h"
#include "nn_util.h"
//#include "nn_demo.h"

using namespace std;
using namespace cv;

static const char *sdkversion = "v1.6.2";
static void *context = NULL;
img_classify_out_t *cls_out = NULL;
aml_config config;

void help(){
	cout << "SDK version:" << sdkversion << endl;
	cout << "Useage ./image_classify_244 < path to image_classify nb file>  < path to jpeg file> " << endl;
}

void process_top5_(float *buf, unsigned int num){

	int i = 0,j = 0;
   	unsigned int MaxClass[5]={0};
	float fMaxProb[5]={0.0};
	float *pfMaxProb = fMaxProb;
	unsigned int *pMaxClass = MaxClass;

	for (j = 0; j < 5; j++){
		for (i=0; i<(int)num; i++){
        	if ((i == (int)*(pMaxClass+0)) || (i == (int)*(pMaxClass+1)) || (i == (int)*(pMaxClass+2)) ||
					(i == (int)*(pMaxClass+3)) || (i == (int)*(pMaxClass+4))){
				continue;
			}
			if (buf[i] > *(pfMaxProb+j)){
				*(pfMaxProb+j) = buf[i];
				*(pMaxClass+j) = i;
			}
		}
	}
	for(i=0; i<5; i++){
		if(cls_out == NULL){
			cout << setw(3) << MaxClass[i] << ": " << setw(12) << setfill(' ') << setprecision(6) << fMaxProb[i] << endl;
		}else{
			cls_out->score[i] = fMaxProb[i];
			cls_out->topClass[i] = MaxClass[i];
		}
	}

}

float Float16ToFloat32(const signed short* src , float* dst ,int lenth)
{
    signed int t1;
    signed int t2;
    signed int t3;
    float out;
    int i;
    for (i = 0 ;i < lenth ;i++)
    {
        t1 = src[i] & 0x7fff;                       // Non-sign bits
        t2 = src[i] & 0x8000;                       // Sign bit
        t3 = src[i] & 0x7c00;                       // Exponent

        t1 <<= 13;                              // Align mantissa on MSB
        t2 <<= 16;                              // Shift sign bit into position

        t1 += 0x38000000;                       // Adjust bias

        t1 = (t3 == 0 ? 0 : t1);                // Denormals-as-zero

        t1 |= t2;
        *((unsigned int*)&out) = t1;                 // Re-insert sign bit
        dst[i] = out;

    }
    return out;
}


float *dtype_To_F32(nn_output * outdata ,int sz)
{                                                                                                    
    int stride, fl, i, zeropoint;
    float scale;
    unsigned char *buffer_u8 = NULL;
    signed char *buffer_int8 = NULL;
    signed short *buffer_int16 = NULL;
    float *buffer_f32 = NULL;

    buffer_f32 = (float *)malloc(sizeof(float) * sz );

    if (outdata->out[0].param->data_format == NN_BUFFER_FORMAT_UINT8)
    {
        stride = (outdata->out[0].size)/sz;
        scale = outdata->out[0].param->quant_data.affine.scale;
        zeropoint =  outdata->out[0].param->quant_data.affine.zeroPoint;

        buffer_u8 = (unsigned char*)outdata->out[0].buf;
        for (i = 0; i < sz; i++)
        {
            buffer_f32[i] = (float)(buffer_u8[stride * i] - zeropoint) * scale;
        }
    }

    else if (outdata->out[0].param->data_format == NN_BUFFER_FORMAT_INT8)
    {
        buffer_int8 = (signed char*)outdata->out[0].buf;
        if (outdata->out[0].param->quant_data.dfp.fixed_point_pos >= 0)
        {
            fl = 1 << (outdata->out[0].param->quant_data.dfp.fixed_point_pos);
            for (i = 0; i < sz; i++)
            {
                buffer_f32[i] = (float)buffer_int8[i] * (1.0/(float)fl);
            }
        }
        else
        {
            fl = 1 << (-outdata->out[0].param->quant_data.dfp.fixed_point_pos);
            for (i = 0; i < sz; i++)
                buffer_f32[i] = (float)buffer_int8[i] * ((float)fl);
        }
    }

    else if (outdata->out[0].param->data_format == NN_BUFFER_FORMAT_INT16)
    {
        buffer_int16 =  (signed short*)outdata->out[0].buf;
        if (outdata->out[0].param->quant_data.dfp.fixed_point_pos >= 0)
        {
            fl = 1 << (outdata->out[0].param->quant_data.dfp.fixed_point_pos);
            for (i = 0; i < sz; i++)
            {   
                buffer_f32[i] = (float)((buffer_int16[i]) * (1.0/(float)fl));
            }
        }
        else
        {   
            fl = 1 << (-outdata->out[0].param->quant_data.dfp.fixed_point_pos);
            for (i = 0; i < sz; i++)
                buffer_f32[i] = (float)((buffer_int16[i]) * ((float)fl));
        }
    }
    else if (outdata->out[0].param->data_format == NN_BUFFER_FORMAT_FP16 )
    {   
        buffer_int16 = (signed short*)outdata->out[0].buf;
        
        Float16ToFloat32(buffer_int16 ,buffer_f32 ,sz);
    }
    
    else if (outdata->out[0].param->data_format == NN_BUFFER_FORMAT_FP32)
    {   
        memcpy(buffer_f32, outdata->out[0].buf, sz);
    }
    else
    {   
        printf("Error: currently not support type, type = %d\n", outdata->out[0].param->data_format);
    }
    return buffer_f32;
}


int create_network(char *nbfile){

	memset(&config,0,sizeof(aml_config));
	config.path = (const char *)nbfile;
	config.nbgType = NN_NBG_FILE;
	config.modelType = TENSORFLOW;
	context = aml_module_create(&config);	
	return 0;
}

int preprocess_network(char *jpegpath){

	int ret = 0;
	nn_input inData;
	const char *jpeg_path = NULL;
	unsigned char *rawdata = NULL;
	jpeg_path = (const char *)jpegpath;
	rawdata = get_jpeg_rawData(jpeg_path,224,224);
	inData.input_index = 0; //this value is index of input,begin from 0
	inData.size = 224*224*3;
	inData.input = rawdata;
	inData.input_type = RGB24_RAW_DATA;
	ret = aml_module_input_set(context,&inData);

	if(rawdata != NULL){
		free(rawdata);
		rawdata = NULL;
		return -1;
	}

	return ret;
}

int postpress_network(){
	
	int i,sz=1;
	aml_output_config_t outconfig;
	float *buffer = NULL;
	nn_output *pout = NULL;	

	outconfig.mdType = CUSTOM_NETWORK;
	outconfig.format = AML_OUTDATA_RAW;

	pout =(nn_output *)aml_module_output_get(context,outconfig);
	
	for (i = 0; i < (int)pout->out[0].param->num_of_dims; i++)
		sz *= pout->out[0].param->sizes[i];
	
	buffer = dtype_To_F32(pout,sz);
	process_top5_(buffer,pout->out[0].size/sizeof(float));

	return 0;
}

int main(int argc,char **argv){

	int ret = 0;
	if (strcmp(argv[1], "--help") == 0){
		help();
		return 0;
	}
	if(argc < 3){
		help();
		return -1;
	}
	
	create_network(argv[1]);
	preprocess_network(argv[2]);
	postpress_network();
	ret = aml_module_destroy(context);

	return ret;

}

