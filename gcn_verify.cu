#include <math.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <algorithm>
using namespace std;
typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;
int e_num = 0;
int F0 = 0, F1 = 0;
// coo graph
vector<vector<int>> edge_index;
vector<vector<float>> edge_val;
vector<int> degree;
vector<int> raw_graph;

//csr graph
int *nodes_index;
int *edges;
float *edges_value;

//layer
float *X0, *W1, *X1, *X1_inter;
//layer on gpu
char *d_mem;
float *d_X0, *d_W1, *d_X1, *d_X1_inter;

//csr graph on gpu
int *d_index, *d_edges;
float *d_edges_val;

void readGraph(char *fname) {
    ifstream infile(fname);
    int source;
    int end;
    infile >> v_num >> e_num;
    while (!infile.eof()) {
        infile >> source >> end;
        if (infile.peek() == EOF) break;
        raw_graph.push_back(source);
        raw_graph.push_back(end);
    }
}

void raw_graph_to_AdjacencyList_to_csr() {
    int src;
    int dst;
    edge_index.resize(v_num);
    edge_val.resize(v_num);
    degree.resize(v_num, 0);

    for (int i = 0; i < raw_graph.size() / 2; i++) {
        src = raw_graph[2 * i];
        dst = raw_graph[2 * i + 1];
        edge_index[dst].push_back(src);
        degree[src]++;
    }


}

void edgeNormalization() {
    for (int i = 0; i < v_num; i++) {
        for (int j = 0; j < edge_index[i].size(); j++) {
            float val = 1 / sqrt(degree[i]) / sqrt(degree[edge_index[i][j]]);
            edge_val[i].push_back(val);
        }
    }
}

void readFloat(char *fname, float *&dst, int num) {
    dst = (float *) malloc(num * sizeof(float));
    FILE *fp = fopen(fname, "rb");
    fread(dst, num * sizeof(float), 1, fp);
    fclose(fp);
}

void initFloat(float *&dst, int num) {
    dst = (float *) malloc(num * sizeof(float));
    memset(dst, 0, num * sizeof(float));
}

void initGPUMemory() {
    // 计算总的内存需求
    size_t totalSize = v_num * F0 * sizeof(float) + 
                   F0 * F1 * sizeof(float) + 
                   v_num * F1 * sizeof(float) + 
                   (v_num + 1) * sizeof(int) + 
                   e_num * sizeof(int) + 
                   e_num * sizeof(float) + 
                   v_num * F1 * sizeof(float);

    // 分配所需的总内存
    cudaMalloc(&d_mem, totalSize);

    // printf("Total size: %zd\n", totalSize);

    size_t offset = 0;

    // 使用偏移量设置指针
    d_X0 = reinterpret_cast<float *>(d_mem + offset);
    offset += v_num * F0 * sizeof (float);

    d_W1 = reinterpret_cast<float *>(d_mem + offset);
    offset += F0 * F1 * sizeof(float);

    d_X1_inter = reinterpret_cast<float *>(d_mem + offset);
    offset += v_num * F1 * sizeof(float);

    d_index = reinterpret_cast<int *>(d_mem + offset);
    offset += (v_num + 1) * sizeof(int);

    d_edges = reinterpret_cast<int *>(d_mem + offset);
    offset +=  e_num * sizeof(int);

    d_edges_val = reinterpret_cast<float *>(d_mem + offset);
    offset += e_num * sizeof(float);

    d_X1 = reinterpret_cast<float *>(d_mem + offset);
    offset += v_num * F1 * sizeof(float);
}

void Preprocessing() {

    nodes_index = (int *) malloc((v_num + 1) * sizeof(int));

    int sum = 0;
    for (int i = 0; i < v_num; i++) {
        nodes_index[i] = sum;
        sum += degree[i];
    }
    nodes_index[v_num] = sum;

    edges = (int *) malloc(e_num * sizeof(int));
    for (int i = 0; i < v_num; i++) {
        memcpy(edges + nodes_index[i], edge_index[i].data(), sizeof(int) * edge_index[i].size());
    }

    edges_value = (float *) malloc(e_num * sizeof(float));
    for (int i = 0; i < v_num; i++) {
        memcpy(edges_value + nodes_index[i], edge_val[i].data(), sizeof(float) * edge_val[i].size());
    }


}

#define TILE_WIDTH 16
#define BLOCK_SIZE 1

__global__ void XW_blockized_(int in_dim, int out_dim, float *in_X, float *out_X, float *W, int v_num) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float tmp = 0.0;

    for (int ph = 0; ph < ceil((float)in_dim / TILE_WIDTH); ++ph) {
        if (row < v_num && ph * TILE_WIDTH + tx < in_dim) {
            ds_A[ty][tx] = in_X[row * in_dim + ph * TILE_WIDTH + tx];
        } else {
            ds_A[ty][tx] = 0.0;
        }

        if (ph * TILE_WIDTH + ty < in_dim && col < out_dim) {
            ds_B[ty][tx] = W[(ph * TILE_WIDTH + ty) * out_dim + col];
        } else {
            ds_B[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            tmp += ds_A[ty][k] * ds_B[k][tx];
        }

        __syncthreads();
    }

    if (row < v_num && col < out_dim) {
        out_X[row * out_dim + col] = tmp;
    }
}

__global__ void logSoftmax_AX_parallalized_(int dim, float *in_X, float *out_X, int *index, int *edges, float *edges_val, int v_num) {

    int vid = blockIdx.x;
    int tid = threadIdx.x;

    if (vid >= v_num) return;

    extern __shared__ float shared_mem[];
    float *shared_out_X = shared_mem;
    shared_out_X[tid] = 0;

    int *nbrs = &edges[index[vid]];
    float *nbrs_val = &edges_val[index[vid]];

    int degree = index[vid + 1] - index[vid];

    __syncthreads();
    
    for (int j = 0; j < degree; j++) {
        int nbr = nbrs[j];
        shared_out_X[tid] += in_X[nbr * dim + tid] * nbrs_val[j];

        __syncthreads();
    }
    
    float *partial_max_val = shared_mem + dim * sizeof (float);
    partial_max_val[tid] = shared_out_X[tid];

    __syncthreads();

    for (int stride = dim / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_max_val[tid] = max(partial_max_val[tid], partial_max_val[tid + stride]);
        }

        __syncthreads();
    }

    float max_val = partial_max_val[0];

    float *partial_sum = shared_mem + 2 * dim * sizeof (float);
    partial_sum[tid] = expf(shared_out_X[tid] - max_val);

    __syncthreads();

    for (int stride = dim / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }

        __syncthreads();
    }

    float sum = partial_sum[0];
    sum = logf(sum);

    shared_out_X[tid] = shared_out_X[tid] - max_val - sum;

    // 将共享内存的数据写回全局内存
    out_X[dim * vid + tid] = shared_out_X[tid];
}

void freeGPUMemory() {
    cudaFree(d_mem);
}

void freeCPUMemory() {
    free(nodes_index);
    free(edges);
    free(edges_value);
}

void GCN() {
    initGPUMemory();

    cudaStream_t memcpy_stream_X0W;
    cudaStreamCreate(&memcpy_stream_X0W);

    cudaMemcpyAsync(d_X0, X0, v_num * F0 * sizeof(float), cudaMemcpyHostToDevice, memcpy_stream_X0W);
    cudaMemcpyAsync(d_W1, W1, F0 * F1 * sizeof(float), cudaMemcpyHostToDevice, memcpy_stream_X0W);
   
    Preprocessing(); 

    cudaStream_t memcpy_stream_graph;
    cudaStreamCreate(&memcpy_stream_graph);
    
    cudaMemcpyAsync(d_index, nodes_index, (v_num + 1) * sizeof(int), cudaMemcpyHostToDevice, memcpy_stream_graph);
    cudaMemcpyAsync(d_edges, edges, e_num * sizeof(int), cudaMemcpyHostToDevice, memcpy_stream_graph);
    cudaMemcpyAsync(d_edges_val, edges_value, e_num * sizeof(float), cudaMemcpyHostToDevice, memcpy_stream_graph);

    cudaStreamSynchronize(memcpy_stream_X0W);

    XW_blockized_<<<dim3(ceil((float)F1 / TILE_WIDTH), ceil((float)v_num / TILE_WIDTH)), 
          dim3(TILE_WIDTH, TILE_WIDTH)>>>
       (F0, F1, d_X0, d_X1_inter, d_W1, v_num);

    cudaStreamSynchronize(memcpy_stream_graph);
    
    logSoftmax_AX_parallalized_<<<v_num, 
                     F1, 
                     3 * F1 * sizeof (float)>>>
                  (F1, d_X1_inter, d_X1, d_index, d_edges, d_edges_val, v_num);

    cudaMemcpy(X1, d_X1, sizeof(float) * v_num * F1, cudaMemcpyDeviceToHost);

    cudaStreamDestroy(memcpy_stream_X0W);
    cudaStreamDestroy(memcpy_stream_graph);
    
    freeGPUMemory();
    freeCPUMemory();
}

float MaxRowSum(float *X, int dim) {

    float max = -__FLT_MAX__;

    for (int i = 0; i < v_num; i++) {
        float sum = 0;
        for (int j = 0; j < dim; j++) {
            sum += X[i * dim + j];
        }
        if (sum > max) max = sum;
    }
    return max;
}

void freeFloats() {
    free(X0);
    free(W1);
    free(X1);
    free(X1_inter);
}

int main(int argc, char **argv) {
    // Do NOT count the time of reading files, malloc, and memset
    F0 = atoi(argv[1]);
    F1 = atoi(argv[2]);

    readGraph(argv[3]);

    readFloat(argv[4], X0, v_num * F0);

    readFloat(argv[5], W1, F0 * F1);

    initFloat(X1, v_num * F1);
    initFloat(X1_inter, v_num * F1);

    raw_graph_to_AdjacencyList_to_csr();
    edgeNormalization();

    cudaFree(0);

    TimePoint start = chrono::steady_clock::now();

    GCN();
    // Time point at the end of the computation
    TimePoint end = chrono::steady_clock::now();
    chrono::duration<double> l_durationSec = end - start;
    double l_timeMs = l_durationSec.count() * 1e3;

    // Compute the max row sum for result verification
    float max_sum = MaxRowSum(X1, F1);
    // The max row sum and the computing time should be print
    printf("verify\n");
    printf("%f\n", max_sum);
    printf("%f\n", l_timeMs);

    freeFloats();
}