/************************************************************************/
// The purpose of this file is to provide a GPU implementation of the 
// heat transfer simulation using MATLAB.
//
// Author: Jason Lowden
// Date: October 20, 2013
//
// File: KMeans.h
/************************************************************************/
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include <cuda_texture_types.h>
#include <cmath>
#include <cuda.h>
#include "HeatTransfer.h"

texture<float,cudaTextureType2D,cudaReadModeElementType> heatTexture ;

__global__ void HTkernel(float* data_d, int size, float heatSpeed){

	int idy = threadIdx.x + blockIdx.x * blockDim.x;
	int idx = threadIdx.y + blockIdx.y * blockDim.y;

	float T_center=0,T_top=0,T_bottom=0,T_left=0,T_right=0;

	if(idx>0 && idx<size-1 && idy>0 && idy<size-1){
		T_top=tex2D(heatTexture,idx,idy-1);
		T_left=tex2D(heatTexture,idx-1,idy);
		T_center=tex2D(heatTexture,idx,idy);
		T_right=tex2D(heatTexture,idx+1,idy);
		T_bottom=tex2D(heatTexture,idx,idy+1);
		data_d[idy*size+idx]=(T_center + (heatSpeed*(( T_top + T_bottom + T_left + T_right)- (4 * T_center))));
	}
}


bool UpdateHeatMap(float* dataIn, float* dataOut, int size, float heatSpeed, int numIterations)
{
    // Error return value
    //cudaError_t status;
	int bytes = size*size*sizeof(float);

	float * data_d;
	
	cudaMalloc((void**) &data_d,bytes);
    

	cudaChannelFormatDesc channel = cudaCreateChannelDesc(32,0,0,0, cudaChannelFormatKindFloat);
	cudaArray* aray;
    cudaMallocArray(&aray, &channel, size, size, 0);

	cudaMemcpyToArray(aray, 0, 0, dataIn, bytes, cudaMemcpyHostToDevice);

	heatTexture.filterMode=cudaFilterModePoint;
	heatTexture.addressMode[0]=cudaAddressModeClamp;
	heatTexture.addressMode[1]=cudaAddressModeClamp;
	heatTexture.normalized=false;
	cudaMemcpy(data_d,dataIn,bytes,cudaMemcpyHostToDevice);
	//cudaMemcpyToArray(aray, 0, 0, dataIn, bytes, cudaMemcpyHostToDevice);

    for(int i = 0; i< numIterations;i++){
	cudaBindTextureToArray(&heatTexture,aray,&channel);

	dim3 dimBlock (16,16);

	int gridx=(int)ceil((float)size/16);
	int gridy=(int)ceil((float)size/16);

	dim3 dimGrid (gridx,gridy);

	HTkernel<<<dimGrid,dimBlock>>>( data_d,  size,  heatSpeed);
	cudaThreadSynchronize();
	
	cudaUnbindTexture(&heatTexture);
	cudaMemcpyToArray(aray, 0, 0, data_d, bytes, cudaMemcpyDeviceToDevice);
	}

    cudaMemcpy(dataOut,data_d,bytes,cudaMemcpyDeviceToHost);
	cudaFree(data_d);
	cudaFreeArray(aray);
	return true;
}