
// 利用共享内存优化矩阵转置

// 需求：矩阵转置

#include "stdio.h"
#include "math.h"

// 定义矩阵大小
#define ROW 2000
#define COL 1500
// ROW * COL --> COL* ROW

#define BLOCK_SIZE 32

// 统一内存管理
__managed__ int maxtri_a[ROW][COL];
__managed__ int maxtri_b_gpu[COL][ROW];
__managed__ int  maxtri_b_cpu[COL][ROW];

__global__ void matrix_transpose_gpu(int a[ROW][COL], int b[COL][ROW], int row, int col){
    // thread的全局索引
    int r = threadIdx.y + blockIdx.y * blockDim.y;
    int c = threadIdx.x + blockIdx.x * blockDim.x;

    // 矩阵范围内的线程
    if(r < row && c < col){
        b[c][r] = a[r][c];
    }
}
// matrix_transpose_gpu函数实现时，频繁的在全局内存中读取值，效率较低。
// 所以依赖共享内存__shared__的高速缓存机制。

// 利用共享内存优化矩阵转置
__global__ void matrix_shared_transpose_gpu(int a[ROW][COL], int b[COL][ROW], int row, int col){
    // 分配共享内存
    __shared__ int sharedm[BLOCK_SIZE+1][BLOCK_SIZE+1];

    int r = threadIdx.y + blockIdx.y * blockDim.y;
    int c = threadIdx.x + blockIdx.x * blockDim.x;

    // 数据拷贝到共享内存
    if(r<row && c<col){
        sharedm[threadIdx.y][threadIdx.x] = a[r][c];
    }
    __syncthreads();

    // 转置后，分块数据中的x\y的索引(转换到b中的全局索引)
    int y1 = threadIdx.y + blockIdx.x * blockDim.x;
    int x1 = threadIdx.x + blockIdx.y * blockDim.y;
    if (x1 < row && y1<col){
        b[y1][x1] = sharedm[threadIdx.x][threadIdx.y];
    }



}

void matrix_transpose_cpu(int a[ROW][COL], int b[COL][ROW], int row, int col)
{
    for(int r = 0; r < row; r++){
        for(int c=0; c < col; c++){
            b[c][r] = a[r][c];
        }
    }
}

int main(){

    // 矩阵初始化
    for(int r = 0; r< ROW; r++){
        for(int c=0; c<COL; c++){
            maxtri_a[r][c] = rand()%1024;
        }
    }
    
    // event 计算耗时
    cudaEvent_t start;
    cudaEvent_t stop_gpu;
    cudaEvent_t stop_cpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_cpu);
    cudaEventCreate(&stop_gpu);

    // GPU计算
    dim3 gridsize(int((COL + BLOCK_SIZE-1)/BLOCK_SIZE),int((ROW+BLOCK_SIZE-1)/BLOCK_SIZE));
    dim3 blocksize(BLOCK_SIZE, BLOCK_SIZE);
    cudaEventRecord(start);
    cudaEventSynchronize(start);
    for (int i=0;i<20;i++){
        matrix_transpose_gpu<<<gridsize, blocksize>>>(maxtri_a, maxtri_b_gpu, ROW, COL);
        // matrix_shared_transpose_gpu<<<gridsize, blocksize>>>(maxtri_a, maxtri_b_gpu, ROW, COL);
        cudaDeviceSynchronize();
        
    }
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    
    matrix_transpose_cpu(maxtri_a, maxtri_b_cpu,ROW, COL);


    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);

    // 统计耗时
    float time_cpu=0, time_gpu=0;
    cudaEventElapsedTime(&time_gpu, start, stop_gpu);
    cudaEventElapsedTime(&time_cpu, stop_gpu, stop_cpu);

    // 比较结果
    bool error = false;
    for(int r = 0; r< COL; r++){
        for(int c=0; c<ROW; c++){
            if (abs(maxtri_b_cpu[r][c] - maxtri_b_gpu[r][c]) > (1.0e-10)){
                error = true;
            }
        }
    }
    printf("result %s\n", error? "Error": "Pass");
    printf("time cpu: %.2f\ntime gpu: %.2f\n", time_cpu, time_gpu/20);

    cudaEventDestroy(start);
    cudaEventDestroy(stop_cpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}