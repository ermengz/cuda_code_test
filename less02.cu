#include <stdio.h>

void addvec_cpu(){

}

__global__ void addvec_gpu(const float* dx,const float *dy,float *dz,const int number)
{
    // gobal thread idx
    // 特别注意！！1dim的grid和block,也会有block的dim
    int gthreaid = threadIdx.x + blockIdx.x * blockDim.x;   
    if (gthreaid < number)
    {
        dz[gthreaid] = dx[gthreaid] + dy[gthreaid];
    }

}

int main(int argc, char** argv){

    constexpr int NUM = 100;

    // 内存分配
    int size = sizeof(float) * NUM;
    float * h_x = (float*) malloc(size);
    float * h_y = (float*) malloc(size);
    float * result = (float*) malloc(size);

    for(int i=0;i<NUM;i++){
        h_x[i] =  1;
        h_y[i] = 2;
    }

    // gpu 内存分配
    float * d_x=NULL;
    float * d_y=NULL;
    float * d_z = NULL;
    cudaError_t err = cudaMalloc((void**)&d_x,size);
    cudaMalloc((void**)&d_y,size);
    cudaMalloc((void**)&d_z,size);

    // copy to device
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    int blocksize = 4;
    int gridsize = (NUM + blocksize -1 ) / blocksize;
    printf("blocksize = %d, gridsize=%d \n",blocksize,gridsize);
    addvec_gpu<<<gridsize, blocksize>>>(d_x,d_y,d_z,NUM);

    cudaDeviceSynchronize();

    cudaMemcpy(result, d_z, size, cudaMemcpyDeviceToHost);  // 注意dst和src的位置，cpu数据在前面

    // 验证
    float error = 0;
    for(int i=0;i<NUM;i++){
        // if(i<10)printf("result[%d]=(%.2f)\n",i,result[i]);
        error += (result[i] -  3);
    }

    if (error > 0.000001)
    {
        printf("calculate wrong \n");
    }
    else{
        printf("calculate right. \n");
    }
    
    // 释放内存
    free(h_x);
    free(h_y);
    free(result);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;

}