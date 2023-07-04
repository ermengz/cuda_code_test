
#include "stdio.h"

#define CHECK(call)                                                     \
do{                                                                     \
    cudaError_t err_code = call;                                        \
    if (err_code != cudaSuccess){                                       \
        printf("error code: {%d}.\n",err_code);                         \
        printf("error string: %s.\n", cudaGetErrorString(err_code));    \
        printf("__File__: %s\n", __FILE__);                             \
        printf("__LINE__:%d\n",__LINE__);                               \
        exit(-1);                                                       \
    }                                                                   \
}while(0)                                                               \


void matrix_sqrt_cpu(int *da, int *db, int* dr, int matrix_size){

    for (int r=0;r<matrix_size;r++){
        for(int c=0;c<matrix_size;c++){
            int temp=0;
            for (int step=0; step< matrix_size; step++){
                temp += da[r*matrix_size + step] * db[step*matrix_size+c];
            }
            dr[r*matrix_size+c] = temp;
        }
    }     

}

__global__ void matrix_sqrt_gpu(int *da, int *db, int* dr, int matrix_size){
    // 2d 的block 和 2d 的grid
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if(r>0 && r<matrix_size && c>0 &&c<matrix_size){

        int temp=0;
        for (int step=0; step< matrix_size; step++){
            temp += da[r*matrix_size + step] * db[step*matrix_size+c];
        }
        dr[r*matrix_size+c] = temp;

    }
}

int main(){

    // 定义矩阵尺寸
    const int matrix_size = 1000;
    int memsize = sizeof(int) * matrix_size * matrix_size;

    // cpu 内存
    int *ma = (int*)malloc(memsize);    // 矩阵平方
    int *mr = (int*) malloc(memsize);

    for (int r = 0; r <matrix_size; r++){
        for (int c=0; c< matrix_size;c++){
            ma[r*matrix_size +c] = rand()%1024;
        }
    }

    // gpu分配内存
    int * da = 0;
    int * dr = 0;
    CHECK(cudaMalloc((void**)&da, memsize));   // 如果不加检查机制，如此的错误，排查起来比较麻烦
    CHECK(cudaMalloc((void**)&dr, memsize));   // 如果不加检查机制，如此的错误，排查起来比较麻烦

    // 事件
    cudaEvent_t start, gpu_stop, cpu_stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&gpu_stop));
    CHECK(cudaEventCreate(&cpu_stop));

    cudaEventRecord(start);
    // copy to gpu
    CHECK(cudaMemcpy(da, ma, memsize, cudaMemcpyHostToDevice)); // 如果不加检查机制，如此的错误，排查起来比较麻烦100
    // // 以下，定义为宏函数
    // cudaError_t err = cudaGetLastError();
    // printf("error code: {%d}.\n",err);
    // printf("error string: %s.\n", cudaGetErrorString(err));
    // printf("__File__: %s\n", __FILE__);
    // printf("__LINE__:%d\n",__LINE__);

    // cuda 矩阵乘法
    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid(int((matrix_size + threadsPerBlock.x -1) / threadsPerBlock.x),
                        int((matrix_size + threadsPerBlock.y -1)/threadsPerBlock.y));
    matrix_sqrt_gpu<<<blocksPerGrid, threadsPerBlock>>>(da, da, dr, matrix_size);

    // 此处为何可以不用cudaDeviceSynchronize()？？？ 
    // 因为下面有一个cudaMemCpy()的拷贝操作，cuda会判断，结果计算完成后，才会copy！！！
    // 而如果核函数后面没有copy或其他的同步机制函数，而是进行下一个核函数的计算，则需要加同步函数！！！！
    // cudaDeviceSynchronize();

    // copy to gpu
    CHECK(cudaMemcpy(mr, dr, memsize, cudaMemcpyDeviceToHost));

    // cudaEventSynchronize();
    cudaEventRecord(gpu_stop);
    
    // 
    matrix_sqrt_cpu(ma, ma, mr, matrix_size);
    cudaEventRecord(cpu_stop);

    float gpu_time=0;
    float cpu_time=0;
    cudaEventElapsedTime(&gpu_time, start, gpu_stop);
    cudaEventElapsedTime(&cpu_time, gpu_stop, cpu_stop);
    printf("gpu time: %.2f ms\n", gpu_time);
    printf("cpu time: %.2f ms \n", cpu_time);
// gpu time: 21.65 ms
// cpu time: 4313.76 ms 
// 
    // 资源回收
    cudaEventDestroy(start);
    cudaEventDestroy(gpu_stop);
    cudaEventDestroy(cpu_stop);

    free(ma);
    free(mr);
    cudaFree(da);
    cudaFree(dr);
    printf("done\n");

    return 0;
}