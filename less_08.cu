
// shared_mem 归约操作对向量进行求和

#include <stdio.h>

#define N 1000000
#define BLOCKSIZE 32
#define GRIDSIZE 32
// 定义的gridsize* blocksize 远远小于N， 所以sm执行的时候，block内的线程每个线程，不单单只处理一个数据；
// 例如：N=1024时,gridsize=32,blocksize=16,；所以每个线程处理的数据量 = 1024/(32*16) = 2

__managed__ int matrix_a[N];
__managed__ int result_gpu[1]={0};

__global__ void vector_sum_gpu(const int * ma, int * result, int count)
{
    // 分配共享内存
    __shared__ int smem[BLOCKSIZE];

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    // 将数据copy到共享内存中，
    int temp=0;
    for(int s=index; s<count; s+=gridDim.x*blockDim.x){
        // if(s)
        temp += ma[s];
    }
    smem[threadIdx.x] = temp;
    __syncthreads();

    // 二分法累加: block内计算
    int tmp=0;
    for(int total_thread=BLOCKSIZE/2; total_thread>=1;total_thread /=2){
        // 注意for循环的结束条件>=1，等于号也要加上
        // 取值
        if(threadIdx.x < total_thread){
            tmp = smem[threadIdx.x] + smem[threadIdx.x + total_thread];
        }
        // __syncthreads();             // 此处价格不加同步都可以
        // 赋值
        if(threadIdx.x < total_thread){
            smem[threadIdx.x] = tmp;
        }
        __syncthreads();                // 这里的同步一定要加
    }
    // 
    // 跨block累加
    if(blockIdx.x * blockDim.x < count){
        if(threadIdx.x == 0){
            atomicAdd(result, smem[0]);
        }
    }
}

int main(int argc, char** argv)
{
    
    int result_cpu=0;
    // 初始化
    for(int x=0; x<N; x++){
        matrix_a[x]= rand()%64; //1; //
    }

    // 计算事件
    cudaEvent_t start, stop_gpu, stop_cpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_cpu);
    cudaEventCreate(&stop_gpu);

    // 开始计时
    cudaEventRecord(start);
    cudaEventSynchronize(start);

    int blocksize = BLOCKSIZE;
    int gridsize = GRIDSIZE;
    for(int i=0;i<20;i++){
        result_gpu[0] =0;
        vector_sum_gpu<<<gridsize, blocksize>>>(matrix_a, result_gpu, N);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    // compute in cpu

    for (int x=0; x<N ; x++){
        result_cpu += matrix_a[x];
    }

    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);

    // compare result between result_cpu and result_gpu
    printf("Result: %s\ngpu_result:%d\ncpu_result:%d\n", result_cpu==result_gpu[0]?"Pass":"ERROR", result_gpu[0], result_cpu);

    float time_cpu = 0;
    float time_gpu = 0;
    cudaEventElapsedTime(&time_cpu, stop_gpu, stop_cpu);
    cudaEventElapsedTime(&time_gpu, start, stop_gpu);
    printf("time_cpu: %.2f\n",time_cpu);
    printf("time_gpu: %.2f\n",time_gpu/20);

    cudaEventDestroy(start);
    cudaEventDestroy(stop_gpu);
    cudaEventDestroy(stop_cpu);

    return 0;
}