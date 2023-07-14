


## 1. 核函数

[code](less01.cu)

核函数规范：
+ 写在cu文件
+ __global__ 标志
+ 无返回值 
+ 不支持可变参数列表
+ 核函数内部只能访问设备的内存

## GPU执行规范
每个线程执行相同的核函数，通过索引的不同，取不同的值进行计算。

线程，块，栅格概念理解

1. 线程是核函数执行的最小单元，一定数量的线程组成一个block，多个block组成一个grid。
2. 根据硬件的不同，如P4，T4，A100等，一个block内部的最大thread数量不相同，如，max_thread=1024 。
3. 核函数的执行数量，即控制thread、block、grid的组织方式，即由程序控制。block\grid均可设置为1维2维3维的变量，grid和block设置之后，执行的单元是线程。从这里可以理解核函数grid和block的设置逻辑和计算方式，参考[博客](https://zhuanlan.zhihu.com/p/151676261?utm_id=0)。
```c++
// 核函数定义
__global__ void kernel_foo(){
    // 此处计算线程的全局id为：
    // 3 dim grid block 
    // int .....
    // 计算的思想：将3d的方体数据，展平为1d的顺序方式。
}
//核函数执行grid和block设置
// 1d grid block
int gridsize = 2;
int blocksize = 3;
// 2d grid block
dim3 gridsize(2,2);
dim3 blocksize(3,3);
// 3d grid block
dim3 gridsize(2,2,2);
dim3 blocksize(3,3,3);
kernel_foo<<<gridsize, blocksize>>>();

```
4. 根据设置的参数，重点理解核函数中，执行的全局线程数id的计算方式：思想就是将多维展平为一维的排列

thread -> block -> grid

## 2. 向量计算

向量是一维的，设置gridsize和blocksize时，即使是1d的，在求线程的全局id时，也要注意blockDim的变换。
```c
__global__ void kernel_foo(){
    // 即使是一维的，也要注意block的idx
    int g_thread_id = threadIdx.x + blockDim.x * blockIdx.x;
}
```

## 3. 矩阵计算

2 dim 的矩阵计算方式： 矩阵乘，行乘以列。理解左右矩阵的取值方式:

```c++
// 方阵乘法
a00 a01 a02 a03      b00 b01 b02 b03          c00 c01 c02 c03   
a10 a11 a12 a13  @   b10 b11 b12 b13     ==》 c10 c11 c12 c13     
a20 a21 a22 a23      b20 b21 b22 b23          c20 c21 c22 c23   
a30 a31 a32 a33      b30 b31 b32 b33          c30 c31 c32 c33   

矩阵在内存中，按行存储，为顺序的内存方式
数据：a00 a01 a02 a03 a10 a11 a12 a13 a20 a21 a22 a23 a30 a31 a32 a33
索引：0   1   2   3   4   5   6   7   8   9   10  11  12  14  14  15
数据：b00 b01 b02 b03 b10 b11 b12 b13 b20 b21 b22 b23 b30 b31 b32 b33

c21 = a20*b01 + a21*b11 + a22*b21 + a23*b31
两个矩阵，数据的索引顺序是相同的。

结果的索引：index = y * size + x
c21对于的y=2,x=1
行列乘积和时step : 0 ~ 3  // 计算时，行列的对应值索引
所以，a的取值方式为：
A矩阵索引：index = y*size + step
B矩阵索引： index = step*size + x
以结果索引为导向，求矩阵的行列取值方式。
```

1. 重点理解行列乘法时，由结果值对应的行列数，去映射求前后两个矩阵的索引方式。
2. 核函数时，特别要注意索引的范围值。
3. cuda内存分配时，要取地址`cudaMalloc((void**) &dx, memsize);` 返回指向内存指针的指针。

## sobel 图像边缘检测

关键点：
1. 理解-核函数内分配的变量，存储空间是在SM内的寄存器register内，所以，变量是能省则省；
2. 理解-滤波时，由结果的索引，反推回输入图像的值索引。

## API 检查机制 和 事件的使用

1. 由于GPU是在设备上进行的内存分配、异步执行，所以出现的很多错误,cpu端是捕获不到的；所以需要专门进行CUDA API接口的检查机制；
2. GPU的异步运行机制，导致在进行程序计时的时候，无法准确卡点，这需要事件机制

事件类型：
```c++
cudaEvent_t start, gpu_stop, cpu_stop;
CHECK(cudaEventCreate(&start));
cudaEventRecord(start);
...
cudaEventRecord(gpu_stop);
...
cudaEventRecord(cpu_stop);

float gpu_time=0;
cudaEventElapsedTime(&gpu_time, start, gpu_stop);

```

## 优化矩阵乘法
+ 统一内存优化
+ shared_memory优化

统一内存， cuda的内存优化机制，是需要分配一次空间；则既可以存cpu的数据，也可以存GPU的数据，内部cpu与GPU数据还能自动同步拷贝；
shared_memory 内存，即为SM内的GPU存储空间，空间不大，**存取的效率高**。而且**同一个block内的核函数**，能共享shared_memory。

共享内存的机制，就是同一个block使用相同的空间，且在chip上的内存中，存取速度快。但是空间太小，所以，需要将矩阵乘，进行分块乘再累加的优化方式。


```c++
// cudamalloc存储的数据是在全局内存中，即板卡内存中，存取速度慢；
// 所以，考虑将数据存到sm的共享内存中，但是共享内存容量有限，则需要对矩阵进行分块计算
// 本函数与上一个函数相比，只是在中间的计算上有 是否分块的区别；
// 重点在，查找第几个子矩阵，并将子矩阵，通过线程的索引，换算回矩阵a,矩阵b的全局索引值！！！！！

// 重点理解分好blocksize大小的块矩阵，怎么拷贝到共享矩阵中；因为分块矩阵要对于到全局的一维索引，而共享矩阵则要计算索引对应的行列
```
A*B=C的矩阵乘法思路：
+ 矩阵A 按block的分块操作，将块数据拷贝到__shared__矩阵sub_a;
+ 矩阵B，按block的分块操作，将块数据拷贝到__shared__矩阵sub_b;
+ 获取sub_a 与 sub_b的分块数据时，都是通过threadIdx的关系转换获取，这个是重点！！！

Note: 理解block中，每个线程执行相同的核函数，但是一个线程内，已经将整个block数据copy到shared内种，从而使得同一个block内的所有线程，共享这部分数据。怎么共享的？其他线程怎么知道这个block数据已经copy到sharedmem里了？,这种**单线程对应block的关系**要梳理梳理！！！
**同一个block中的线程，申请的share_memory内存，会指向同一个内存块！！！**这就解释了上面提到的，多个线程会不会同时开辟多块内存的问题。

```__syncthreads()``` 当__syncthreads被调用时，在同一个线程块中每个线程都必须等待直至该线程块中所有其他线程都已经达到这个同步点。到达这个同步点之后，往后执行，而且同block内的线程，共享block内的全局内存和共享内存的访存权限。


## shared_mem 优化矩阵转置
1. 分块转置的思维。与分块乘积类似。
2. 步骤：a.T = b
    + 按blocksize * blocksize的大小，分配共享内存
    + 将a中的数据，copy到shared_mem 中。这里要注意，
    + 将shared_mem数据，赋值到b中。通过threadIdx.x\y以及blockDim.x\y参数，以block的转置关系，将threadidx换算到b的block值的全局索引


## shared_memory的归约操作、原子操作

1. 