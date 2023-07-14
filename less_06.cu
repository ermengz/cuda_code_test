
// 统一内存机制、shared_memory机制优化矩阵乘法
// a[][] * b[][] = c[][]
// 
//                         b00 b01 b02 b03
//                         b10 b11 b12 b13
//                         b20 b21 b22 b23
//                         b30 b31 b32 b33
//
// a00 a01 a02 a03         c00 c01 c02 c03
// a10 a11 a12 a13         c10 c11 c12 c13     block(1, 0) -> shared memory
// a20 a21 a22 a23         c20 c21 c22 c23     c20 c21
// a30 a31 a32 a33         c30 c31 c32 c33     c30 c31
//
//                              b00 b01->  sub_b_step_0
//                              b10 b11
//
//                              b20 b21->  sub_b_step_1
//                              b30 b31
// sub_a_step_0 sub_a_step_1    sub_c
// a20 a21      a22 a23         c20 c21
// a30 a31      a32 a33         c30 c31
//
// sub_c = sub_a_step_0 * sub_b_step_0 + sub_a_step_1 * sub_b_step_1;
//
// for(int step =0; step < N/block_size; step++ )
//      load sub_a_step to shared memory;
//      load sub_b_step to shared memory;
//      tmp += sub_a_step_on_sharedmemory * sub_b_step_on_sharedmemory;
// sub_c = tmp;
//
// cudaMalloc -> global memory
// data global memory -> shared memory
// threads shared memory -> register
// shared memory SM(stream multi-processor) same block same shared memory
//
// c21 = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31
// a00 a01 a02 a03 a10 a11 a12 a13 a20 a21 a22 a23 a30 a31 a32 a33
// 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
// b00 b01 b02 b03 b10 b11 b12 b13 b20 b21 b22 b23 b30 b31 b32 b33


#include "stdio.h"


#define BLOCK_SIZE 2

// 常规的矩阵乘法
__global__ void matrix_mutil_gpu(const int *ma, const int *mb,  int *mc, const int sm, const int sn,const int sk)
{
    // 全局线程索引
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    // 需要加判断的原因：分配的线程数[gridDim * threadDim]会大于实际的矩阵！！
    if( r<sm &&  c<sk){
        int temp=0;
        for (int step=0; step<sn; step++){
            
            temp += ma[r*sn + step] * mb[step*sk+c];
        }
        mc[r*sk+c] = temp;
    }

}

// 本段理解参考：https://zhuanlan.zhihu.com/p/434513198

// 如果矩阵a和矩阵b的值太大，矩阵c的结果值，会频繁的操作对ma、ma存取操作。
// cudamalloc存储的数据是在全局内存中，即板卡内存中，存取速度慢；
// 所以，考虑将数据存到sm的共享内存中，但是共享内存容量有限，则需要对矩阵进行分块计算
// 本函数与上一个函数相比，只是在中间的计算上有 是否分块的区别；
// 重点在，查找第几个子矩阵，并将子矩阵，通过线程的索引，换算回矩阵a,矩阵b的全局索引值！！！！！
__global__ void matrix_mutil_gpu_block(const int *ma, const int *mb,  int *mc, const int sm, const int sn,const int sk)
{
    // 分配共享内存，共享内存是指同一个block内共享； 即同一个block内只分配一次，共同使用。
    // 从这里也可以看出，核函数是按block为单元的线程id
    __shared__ int sub_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int sub_b[BLOCK_SIZE][BLOCK_SIZE];

    // 全局线程索引
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    // 将数据分块计算。
    int temp=0;
    int idx = 0;
    for(int step=0; step < sn/BLOCK_SIZE; step++){
        // ma的子矩阵，拷贝到sub_a 
        int sub_a_x = step * BLOCK_SIZE + threadIdx.x;   // X轴 blocksize的第几块block
        int sub_a_y = r;                                 // 结果行轴，与ma的行轴一样
        idx = sub_a_y * sn + sub_a_x; // ma 的子矩阵值，在全局的线程索引！！！
        if (sub_a_x >=sn || sub_a_y>=sm){
            sub_a[threadIdx.y][threadIdx.x] = 0;
        }
        else{
            sub_a[threadIdx.y][threadIdx.x] = ma[idx];
            printf("r=%d, c=%d, ma=%d\n",r,c,ma[idx]);
        }

        // mb对于的子矩阵，拷贝到sub_b
        int sub_b_x = c;                             // 结果列轴，与ma的列轴一样
        int sub_b_y =step*BLOCK_SIZE + threadIdx.y;   // Y轴 blocksize的第几块
        idx = sub_b_y * sk + sub_b_x;   
        if(sub_b_x>=sk || sub_b_y >=sn){
            sub_b[threadIdx.y][threadIdx.x]  =0;
        }           
        else{
            sub_b[threadIdx.y][threadIdx.x] = mb[idx];
            printf("r=%d, c=%d, mb=%d\n",r,c,mb[idx]);
        }
        // 当__syncthreads被调用时，在同一个线程块中每个线程都必须等待直至该线程块中所有其他线程都已经达到这个同步点。
        // 在栅栏之前所有线程产生的所有全局内存和共享内存 访问，将会在栅栏后对线程块中所有其他的线程可见.
        __syncthreads();
        // 共享内存内计算
        for(int i=0;i<BLOCK_SIZE;i++){
            temp += sub_a[threadIdx.y][i] * sub_b[i][threadIdx.x];
            printf("r=%d, c=%d, sub_a=%d, sub_b=%d\n",r,c,sub_a[threadIdx.y][i], sub_b[i][threadIdx.x]);
        }
        __syncthreads();

    }

    if( r<sm &&  c<sk){
        mc[r*sk+c] = temp;
    }

}


void matrix_mutil_cpu(const int *ma,const int *mb, int *mc,const int sm,const int sn,const int sk)
{
    for (int r=0; r < sm; r++)
    {
        for (int c=0; c < sk; c++)
        {
            int temp=0;
            for (int step=0; step< sn; step++)
            {
                temp += ma[r*sn + step] * mb[step*sk+c];
            }
            mc[r*sk + c] = temp;

        }
    }
}

// Matrix_a M*N
// Matrix_b N*K
// 结果 Matrix_c M*K
// 定义矩阵大小
constexpr int M = 2;
constexpr int N = 4;
constexpr int K = 2;
// 统一内存，分配空间, 不允许放在host侧的函数内
__managed__ int matrixa[M*N];
__managed__ int matrixb[N*K];
__managed__ int mc_gpu[M*K];
__managed__ int mc_cpu[M*K];

// 矩阵乘法
int main(int arc, char** argv){

    // 初始化数据
    for (int r = 0; r < M; r++)
    {
        for(int c=0; c<N; c++){
            matrixa[r*N +c] = r*BLOCK_SIZE+c; //rand()%1024;
        }
    }
    for (int r = 0; r < N; r++)
    {
        for(int c=0; c<K; c++){
            matrixb[r*K +c] = r*BLOCK_SIZE+c;// rand()%1024;
        }
    }

    // 核函数计算
    dim3 threadPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blockPerGrid((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE);
    // matrix_mutil_gpu<<<blockPerGrid, threadPerBlock>>>(matrixa, matrixb, mc_gpu, M, N, K);
    matrix_mutil_gpu_block<<<blockPerGrid, threadPerBlock>>>(matrixa, matrixb, mc_gpu, M, N, K);
    
    cudaDeviceSynchronize();// win 系统要加同步机制函数
    // cpu矩阵乘法
    matrix_mutil_cpu(matrixa, matrixb, mc_cpu, M, N, K);

    // 比较结果
    bool error = false;
    for (int r=0; r<M; r++){
        for (int c = 0; c < K; c++)
        {
            // if (abs(mc_cpu[r*K+c] - mc_gpu[r*K+c]) > (1.0e-10)){
            if (fabs(mc_cpu[r*K+c] - mc_gpu[r*K+c]) > (1.0e-10)){
                // printf("cpu=%d,gpu=%d,",mc_cpu[r*K+c] , mc_gpu[r*K+c]);
                error = true;
            }
        }
    }

    printf("result: %s\n", error? "Error":"PASS");

    return 0;

}