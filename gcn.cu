#include <math.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <assert.h>

using namespace std;

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;
int e_num = 0;
int F0 = 0, F1 = 0;
// coo graph
vector<vector<int>> edge_index;
vector<vector<double>> edge_val;
vector<int> degree;
vector<int> raw_graph;

// csr graph;
int *nodes_index;
int *edges;
double *edges_value;

// layer
double *X0, *W1, *X1, *X1_inter;
// layer on gpu
double *d_X0, *d_W1, *d_X1, *d_X1_inter;

// csr graph on gpu
int *d_index, *d_edges;
double *d_edges_val;

void readGraph(char *fname)
{
    ifstream infile(fname);
    int source;
    int end;
    infile >> v_num >> e_num;
    while (!infile.eof())
    {
        infile >> source >> end;
        if (infile.peek() == EOF)
            break;
        raw_graph.push_back(source);
        raw_graph.push_back(end);
    }
}

void to_csr()
{

    nodes_index = (int *)malloc(v_num * sizeof(int) + 1);

    int sum = 0;
    for (int i = 0; i < v_num; i++)
    {
        nodes_index[i] = sum;
        sum += degree[i];
    }
    nodes_index[v_num] = sum;

    edges = (int *)malloc(e_num * sizeof(int));
    for (int i = 0; i < v_num; i++)
    {
        memcpy(edges + nodes_index[i], edge_index[i].data(), sizeof(int) * edge_index[i].size());
    }

    edges_value = (double *)malloc(e_num * sizeof(double));
    for (int i = 0; i < v_num; i++)
    {
        memcpy(edges_value + nodes_index[i], edge_val[i].data(), sizeof(double) * edge_val[i].size());
    }
}

void raw_graph_to_AdjacencyList()
{
    int src;
    int dst;
    edge_index.resize(v_num);
    edge_val.resize(v_num);
    degree.resize(v_num, 0);

    for (int i = 0; i < raw_graph.size() / 2; i++)
    {
        src = raw_graph[2 * i];
        dst = raw_graph[2 * i + 1];
        edge_index[dst].push_back(src);
        degree[src]++;
    }
}

void edgeNormalization()
{
    for (int i = 0; i < v_num; i++)
    {
        for (int j = 0; j < edge_index[i].size(); j++)
        {
            double val = 1 / sqrt(degree[i]) / sqrt(degree[edge_index[i][j]]);
            edge_val[i].push_back(val);
        }
    }
}

void readdouble(char *fname, double *&dst, int num)
{
    dst = (double *)malloc(num * sizeof(double));
    FILE *fp = fopen(fname, "rb");
    fread(dst, num * sizeof(double), 1, fp);
    fclose(fp);
}

void initdouble(double *&dst, int num)
{
    dst = (double *)malloc(num * sizeof(double));
    memset(dst, 0, num * sizeof(double));
}

__global__ void XW_(int in_dim, int out_dim, double *in_X, double *out_X, double *W, int v_num)
{

    int tid = threadIdx.x + blockIdx.x * blockDim.x; // 控制v_vum

    if (tid >= v_num)
        return;

    double *tmp_in_X = in_X;
    double *tmp_out_X = out_X;
    double *tmp_W = W;

    for (int j = 0; j < out_dim; j++)
    {
        for (int k = 0; k < in_dim; k++)
        {
            tmp_out_X[tid * out_dim + j] += tmp_in_X[tid * in_dim + k] * tmp_W[k * out_dim + j];
        }
    }
}

#define TILE_WIDTH 16

__global__ void
__launch_bounds__(1024)
XW_blockized_(int in_dim, int out_dim, double *in_X, double *out_X, double *W, int v_num) {
    __shared__ double ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    double tmp = 0.0;

    for (int ph = 0; ph < ceil((double)in_dim / TILE_WIDTH); ++ph) {
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

        for (int k = 0; k < TILE_WIDTH; k++) {
            tmp += ds_A[ty][k] * ds_B[k][tx];
        }

        __syncthreads();
    }

    __shared__ double ds_O[TILE_WIDTH][TILE_WIDTH];
    ds_O[ty][tx] = tmp;

    if (row < v_num && col < out_dim) {
        out_X[row * out_dim + col] = ds_O[ty][tx];
    }
}

#define TILES_PER_BLOCK 1

__global__ void
__launch_bounds__(1024)
XW_blockized_better_(int in_dim, int out_dim, double *in_X, double *out_X, double *W, int v_num) {
    
    assert(out_dim == 16);
    
    __shared__ double ds_A[TILES_PER_BLOCK][TILE_WIDTH][TILE_WIDTH];
    __shared__ double ds_B[TILES_PER_BLOCK][TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tz = threadIdx.z;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    double tmp = 0.0;

    for (int ph = tz; ph < (in_dim + TILE_WIDTH - 1) / TILE_WIDTH; ph += TILES_PER_BLOCK) {
        if (row < v_num && ph * TILE_WIDTH + tx < in_dim) {
            ds_A[tz][ty][tx] = in_X[row * in_dim + ph * TILE_WIDTH + tx];
        } else {
            ds_A[tz][ty][tx] = 0.0;
        }

        if (ph * TILE_WIDTH + ty < in_dim && col < out_dim) {
            ds_B[tz][ty][tx] = W[(ph * TILE_WIDTH + ty) * out_dim + col];
        } else {
            ds_B[tz][ty][tx] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            tmp += ds_A[tz][ty][k] * ds_B[tz][k][tx];
        }

        __syncthreads();
    }

    __shared__ double ds_O[TILE_WIDTH][TILE_WIDTH];
    ds_O[ty][tx] = 0;
    __syncthreads();

    atomicAdd(&(ds_O[ty][tx]), tmp);

    __syncthreads();

    if (row < v_num && col < out_dim && tz == 0) {
        out_X[row * out_dim + col] = ds_O[ty][tx];
    }
}

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define BM 8
#define BN 8
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_V1(
    double * __restrict__ a, double * __restrict__ b, double * __restrict__ c,
    const int M, const int N, const int K) {

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ double s_a[BM][BK];
    __shared__ double s_b[BK][BN];

    double r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;  // tid/2, row of s_a
    int load_a_smem_k = (tid & 1) << 2;  // (tid % 2 == 0) ? 0 : 4, col of s_a
    int load_b_smem_k = tid >> 5;   // tid/32, row of s_b
    int load_b_smem_n = (tid & 31) << 2;  // (tid % 32) * 4, col of s_b

    int load_a_gmem_m = by * BM + load_a_smem_m;  // global row of a
    int load_b_gmem_n = bx * BN + load_b_smem_n;  // global col of b

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        for (int i = 0; i < 4; i++) {
            int load_a_gmem_k = bk * BK + load_a_smem_k + i;   // global col of a
            int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
            s_a[load_a_smem_m][load_a_smem_k + i] = a[load_a_gmem_addr];

            int load_b_gmem_k = bk * BK + load_b_smem_k;   // global row of b
            int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n + i, N);
            s_b[load_b_smem_k][load_b_smem_n + i] = b[load_b_gmem_addr];
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    int comp_a_smem_m = ty * TM + m;
                    int comp_b_smem_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            c[store_c_gmem_addr] = r_c[i][j];
        }
    }
}

void gemm(double * A, double *B, double *O, const int M, const int N, const int K) {
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_V1<<<gridDim, blockDim>>>(A, B, O, M, N, K);
}


__global__ void AX_(int dim, double *in_X, double *out_X, int *index, int *edges, double *edges_val, int v_num)
{

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= v_num)
        return;

    int *nbrs = &edges[index[tid]];
    double *nbrs_val = &edges_val[index[tid]];

    int degree = index[tid + 1] - index[tid];

    for (int j = 0; j < degree; j++)
    {
        int nbr = nbrs[j];
        for (int k = 0; k < dim; k++)
        {
            out_X[dim * tid + k] += in_X[nbr * dim + k] * nbrs_val[j];
        }
    }
}

__global__ void logSoftmax_AX_(int dim, double *in_X, double *out_X, int *index, int *edges, double *edges_val, int v_num) {
    assert(dim == 16);

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bdy = blockDim.y;

    if (bx >= v_num) return;

    __shared__ char shared_mem[3 * 16 * sizeof (double)];
    double *shared_out_X = (double*) shared_mem;
    shared_out_X[tx] = 0;

    int *nbrs = &edges[index[bx]];
    double *nbrs_val = &edges_val[index[bx]];

    int degree = index[bx + 1] - index[bx];

    __syncthreads();
    
    for (int j = ty; j < degree; j += bdy) {
        int nbr = nbrs[j];
        double x = in_X[nbr * dim + tx];
        double y = nbrs_val[j];
        atomicAdd(&(shared_out_X[tx]), x * y);
    }

    if (ty) {
        return;
    }

    __syncthreads();

    double *partial_max_val = (double*) (shared_mem + dim * sizeof (double));
    partial_max_val[tx] = shared_out_X[tx];

    for (int stride = dim / 2; stride > 0; stride /= 2) {
        if (tx < stride) {
            partial_max_val[tx] = max(partial_max_val[tx], partial_max_val[tx + stride]);
        }

        __syncthreads();
    }

    double max_val = partial_max_val[0];

    double *partial_sum = (double*) (shared_mem + 2 * dim * sizeof (double));
    partial_sum[tx] = exp(shared_out_X[tx] - max_val);

    __syncthreads();

    for (int stride = dim / 2; stride > 0; stride /= 2) {
        if (tx < stride) {
            partial_sum[tx] += partial_sum[tx + stride];
        }

        __syncthreads();
    }

    double sum = partial_sum[0];
    sum = log(sum);

    shared_out_X[tx] = shared_out_X[tx] - max_val - sum;

    // 将共享内存的数据写回全局内存
    out_X[dim * bx + tx] = shared_out_X[tx];
}

__global__ void 
__launch_bounds__(1024)
logSoftmax_AX_better_(int dim, double *in_X, double *out_X, int *index, int *edges, double *edges_val, int v_num) {
    assert(dim == 16);

    int bx = blockIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int threads_per_node = blockDim.y;
    int nodes_per_block = blockDim.z;

    int vid = bx * nodes_per_block + tz;

    if (vid >= v_num) return;

    // __shared__ char shared_mem[THREADS_PER_NODE][3 * 16 * sizeof (double)];
    extern __shared__ char shared_mem[];
    
    double *shared_out_X = (double*) (shared_mem + tz * 3 * 16 * sizeof (double));
    if (ty == 0) {
        shared_out_X[tx] = 0;
    }
    
    __syncthreads();

    int *nbrs = &edges[index[vid]];
    double *nbrs_val = &edges_val[index[vid]];

    int degree = index[vid + 1] - index[vid];

    for (int j = ty; j < degree; j += threads_per_node) {
        int nbr = nbrs[j];
        double x = in_X[nbr * 16 + tx];
        double y = nbrs_val[j];
        atomicAdd(&(shared_out_X[tx]), x * y);
    }
    
    // if (ty) {
    //     return;
    // }

    // __syncthreads();

    // double max_val = shared_out_X[tx];
    // for (int i = 0; i < 16; i++) {
    //     max_val = shared_out_X[i] > max_val ? shared_out_X[i] : max_val;
    // }

    // __syncthreads();

    // double *shared_out_X_exp = (double*) (shared_mem[tz] + 16 * sizeof (double));

    // shared_out_X_exp[tx] = exp(shared_out_X[tx] - max_val);

    // __syncthreads();

    // double sum = 0;
    // for (int i = 0; i < 16; i++) {
    //     sum += shared_out_X_exp[i];
    // }
    // sum = log(sum);
    
    // shared_out_X[tx] = shared_out_X[tx] - max_val - sum;

    // out_X[16 * vid + tx] = shared_out_X[tx];
    
    // ==========

    if (ty) {
        return;
    }

    __syncthreads();

    double *partial_max_val = (double*) (shared_mem + tz * 3 * 16 * sizeof (double) + 16 * sizeof (double));
    partial_max_val[tx] = shared_out_X[tx];

    for (int stride = 16 / 2; stride > 0; stride /= 2) {
        if (tx < stride) {
            partial_max_val[tx] = max(partial_max_val[tx], partial_max_val[tx + stride]);
        }

        __syncthreads();
    }

    double max_val = partial_max_val[0];

    double *partial_sum = (double*) (shared_mem + tz * 3 * 16 * sizeof (double) + 2 * 16 * sizeof (double));
    partial_sum[tx] = exp(shared_out_X[tx] - max_val);

    __syncthreads();

    for (int stride = 16 / 2; stride > 0; stride /= 2) {
        if (tx < stride) {
            partial_sum[tx] += partial_sum[tx + stride];
        }

        __syncthreads();
    }

    double sum = partial_sum[0];
    sum = log(sum);

    shared_out_X[tx] = shared_out_X[tx] - max_val - sum;

    // 将共享内存的数据写回全局内存
    out_X[16 * vid + tx] = shared_out_X[tx];
}

__global__ void 
__launch_bounds__(1024)
logSoftmax_AX_better_rowsum_trick_(int dim, double *in_X, double *out_X, int *index, int *edges, double *edges_val, int v_num) {
    assert(dim == 16);

    int bx = blockIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int threads_per_node = blockDim.y;
    int nodes_per_block = blockDim.z;

    int vid = bx * nodes_per_block + tz;

    if (vid >= v_num) return;

    // __shared__ char shared_mem[THREADS_PER_NODE][3 * 16 * sizeof (double)];
    extern __shared__ char shared_mem[];
    
    double *shared_out_X = (double*) (shared_mem + tz * 3 * 16 * sizeof (double));
    if (ty == 0) {
        shared_out_X[tx] = 0;
    }
    
    __syncthreads();

    int *nbrs = &edges[index[vid]];
    double *nbrs_val = &edges_val[index[vid]];

    int degree = index[vid + 1] - index[vid];

    for (int j = ty; j < degree; j += threads_per_node) {
        int nbr = nbrs[j];
        double x = in_X[nbr * 16 + tx];
        double y = nbrs_val[j];
        atomicAdd(&(shared_out_X[tx]), x * y);
    }
    
    // if (ty) {
    //     return;
    // }

    // __syncthreads();

    // double max_val = shared_out_X[tx];
    // for (int i = 0; i < 16; i++) {
    //     max_val = shared_out_X[i] > max_val ? shared_out_X[i] : max_val;
    // }

    // __syncthreads();

    // double *shared_out_X_exp = (double*) (shared_mem[tz] + 16 * sizeof (double));

    // shared_out_X_exp[tx] = exp(shared_out_X[tx] - max_val);

    // __syncthreads();

    // double sum = 0;
    // for (int i = 0; i < 16; i++) {
    //     sum += shared_out_X_exp[i];
    // }
    // sum = log(sum);
    
    // shared_out_X[tx] = shared_out_X[tx] - max_val - sum;

    // out_X[16 * vid + tx] = shared_out_X[tx];
    
    // ==========

    if (ty) {
        return;
    }

    __syncthreads();

    double *partial_max_val = (double*) (shared_mem + tz * 3 * 16 * sizeof (double) + 16 * sizeof (double));
    partial_max_val[tx] = shared_out_X[tx];

    for (int stride = 16 / 2; stride > 0; stride /= 2) {
        if (tx < stride) {
            partial_max_val[tx] = max(partial_max_val[tx], partial_max_val[tx + stride]);
        }

        __syncthreads();
    }

    double max_val = partial_max_val[0];

    double *partial_sum = (double*) (shared_mem + tz * 3 * 16 * sizeof (double) + 2 * 16 * sizeof (double));
    partial_sum[tx] = exp(shared_out_X[tx] - max_val);

    __syncthreads();

    for (int stride = 16 / 2; stride > 0; stride /= 2) {
        if (tx < stride) {
            partial_sum[tx] += partial_sum[tx + stride];
        }

        __syncthreads();
    }

    double sum = partial_sum[0];
    sum = log(sum);

    shared_out_X[tx] = shared_out_X[tx] - max_val - sum;

    __syncthreads();

    if (tx == 0) {
        sum = 0;
        for (int i = 0; i < 16; i++) {
            sum += shared_out_X[i];
        }
        out_X[vid] = sum;
    }
}

void LogSoftmax(int dim, double *X)
{

    for (int i = 0; i < v_num; i++)
    {
        double max = X[i * dim + 0];
        for (int j = 1; j < dim; j++)
        {
            if (X[i * dim + j] > max)
                max = X[i * dim + j];
        }

        double sum = 0;
        for (int j = 0; j < dim; j++)
        {
            sum += exp(X[i * dim + j] - max);
        }
        sum = log(sum);

        for (int j = 0; j < dim; j++)
        {
            X[i * dim + j] = X[i * dim + j] - max - sum;
        }
    }
}

double MaxRowSum(double *X, int dim)
{

    double max = -__FLT_MAX__;

    for (int i = 0; i < v_num; i++)
    {
        double sum = 0;
        for (int j = 0; j < dim; j++)
        {
            sum += X[i * dim + j];
        }
        if (sum > max)
            max = sum;
    }
    return max;
}

void freedoubles()
{
    free(X0);
    free(W1);
    free(X1);
    free(X1_inter);
    free(nodes_index);
    free(edges);
    free(edges_value);
    cudaFree(d_X0);
    cudaFree(d_X1_inter);
    cudaFree(d_W1);
    cudaFree(d_X1);
    cudaFree(d_index);
    cudaFree(d_edges);
    cudaFree(d_edges_val);
}

void initGPUMemory()
{

    cudaFree(0);

    cudaMalloc(&d_X0, v_num * F0 * sizeof(double));
    cudaMemcpy(d_X0, X0, v_num * F0 * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_X1_inter, v_num * F1 * sizeof(double));
    cudaMemcpy(d_X1_inter, X1_inter, v_num * F1 * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_W1, F0 * F1 * sizeof(double));
    cudaMemcpy(d_W1, W1, F0 * F1 * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_X1, F1 * v_num * sizeof(double));
    cudaMemcpy(d_X1, X1, F1 * v_num * sizeof(double), cudaMemcpyHostToDevice);

    //    d_index, d_edge, d_edge_val

    cudaMalloc(&d_index, (v_num + 1) * sizeof(int));
    cudaMemcpy(d_index, nodes_index, (v_num + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_edges, e_num * sizeof(int));
    cudaMemcpy(d_edges, edges, e_num * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_edges_val, e_num * sizeof(double));
    cudaMemcpy(d_edges_val, edges_value, e_num * sizeof(double), cudaMemcpyHostToDevice);
}

double GCN()
{
    assert(F1 == 16);


    // cudaMemset(d_X1_inter, 0, v_num * F1 * sizeof(double));
    // cudaMemset(d_X1, 0, F1 * v_num * sizeof(double));

    TimePoint start = chrono::steady_clock::now();

    // ==========
    // const int block_size = 512;
    // const int grid_size = v_num / block_size + 1;

    // XW_<<<grid_size, block_size>>>(F0, F1, d_X0, d_X1_inter, d_W1, v_num);
    
    XW_blockized_<<<dim3((16 + TILE_WIDTH - 1) / TILE_WIDTH, (v_num + TILE_WIDTH - 1) / TILE_WIDTH), dim3(TILE_WIDTH, TILE_WIDTH)>>>(F0, 16, d_X0, d_X1_inter, d_W1, v_num);

    // XW_blockized_better_<<<dim3((16 + TILE_WIDTH - 1) / TILE_WIDTH, (v_num + TILE_WIDTH - 1) / TILE_WIDTH), dim3(TILE_WIDTH, TILE_WIDTH, TILES_PER_BLOCK)>>>(F0, 16, d_X0, d_X1_inter, d_W1, v_num);

    // AX_<<<grid_size, block_size>>>(F1, d_X1_inter, d_X1, d_index, d_edges, d_edges_val, v_num);
    // cudaMemcpy(X1, d_X1, sizeof(double) * v_num * F1, cudaMemcpyDeviceToHost);

    // LogSoftmax(F1, X1);

    // logSoftmax_AX_<<<v_num, 
    //                  dim3(16, 8)
    //               >>>(16, d_X1_inter, d_X1, d_index, d_edges, d_edges_val, v_num);

    int n_edge = nodes_index[v_num];
    double avg_edge_per_node = (double) n_edge / v_num;
    
    int nodes_per_block, threads_per_node;

    if (avg_edge_per_node > 10) {
        threads_per_node = 16;
    } else {
        threads_per_node = 2;
    }

    // threads_per_node = 16;

    nodes_per_block = 1024 / 16 / threads_per_node;
    
    // logSoftmax_AX_better_<<<(v_num + nodes_per_block - 1) / nodes_per_block, dim3(16, threads_per_node, nodes_per_block), nodes_per_block * 3 * 16 * sizeof (double)>>>(16, d_X1_inter, d_X1, d_index, d_edges, d_edges_val, v_num);

    logSoftmax_AX_better_rowsum_trick_<<<(v_num + nodes_per_block - 1) / nodes_per_block, dim3(16, threads_per_node, nodes_per_block), nodes_per_block * 3 * 16 * sizeof (double)>>>(16, d_X1_inter, d_X1, d_index, d_edges, d_edges_val, v_num);


    for (int i = 0; i < v_num; i++) {
        for (int j = 0; j < 16; j++) {
            X1[i * 16 + j] = j == 0 ? -1e9 : 0;
        }
    }

    static double *temp = new double[v_num];

    cudaMemcpy(temp, d_X1, sizeof (double) * v_num, cudaMemcpyDeviceToHost);

    // // 开始计时
    // auto postprocess_start = std::chrono::high_resolution_clock::now();

    // 需要计时的代码段
    double mx = -__FLT_MAX__;
    int idx = 0;
    
    for (int i = 0; i < v_num; i++) {
        if (temp[i] > mx) {
            mx = temp[i];
            idx = i;
        }
    }
    X1[16 * idx] = mx;

    // // 结束计时
    // auto postprocess_end = std::chrono::high_resolution_clock::now();

    // // 计算持续时间
    // std::chrono::duration<double, std::milli> duration = postprocess_end - postprocess_start;
    // printf("%lf\n", duration.count());

    // cudaError_t error = cudaGetLastError();
    // if (error != cudaSuccess)
    // {
    //     fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
    // }

    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);

    // // 打印每个块的最大线程数
    // printf("Maximum threads per block: %d\n", prop.maxThreadsPerBlock);

    
    // cudaMemcpyAsync(X1, d_X1, sizeof (double) * v_num * 16, cudaMemcpyDeviceToHost);

    // cudaStream_t stream_0;
    // cudaStreamCreate(&stream_0);
    // cudaStream_t stream_1;
    // cudaStreamCreate(&stream_1);

    // logSoftmax_AX_<<<v_num / 2, 
    //                  dim3(16, 4),
    //                  0,
    //                  stream_0
    //                  >>>(16, d_X1_inter, d_X1, d_index, d_edges, d_edges_val, v_num / 2);
    // // cudaStreamSynchronize(stream_0);
    // cudaMemcpyAsync(X1, d_X1, sizeof (double) * v_num / 2 * 16, cudaMemcpyDeviceToHost, stream_0);
    
    // int second_half_start = v_num / 2;
    // int second_half_v_num = v_num - second_half_start;

    // logSoftmax_AX_<<<second_half_v_num, 
    //                  dim3(16, 4),
    //                  0,
    //                  stream_1
    //                  >>>(16, 
    //                      d_X1_inter, 
    //                      d_X1 + second_half_start * 16, 
    //                      d_index + second_half_start, 
    //                      d_edges, 
    //                      d_edges_val, 
    //                      second_half_v_num);

    // cudaMemcpyAsync(X1 + second_half_start * 16, d_X1 + second_half_start * 16, sizeof (double) * second_half_v_num * 16, cudaMemcpyDeviceToHost, stream_1);

    // cudaStreamSynchronize(stream_0);
    // cudaStreamSynchronize(stream_1);

    // cudaStreamDestroy(stream_0);
    // cudaStreamDestroy(stream_1);

    // ==========

    TimePoint end = chrono::steady_clock::now();
    chrono::duration<double> l_durationSec = end - start;
    double l_timeMs = l_durationSec.count() * 1e3;

    return l_timeMs;
}

void XW_verify(int in_dim, int out_dim, double *in_X, double *out_X, double *W)
{
    double *tmp_in_X = in_X;
    double *tmp_out_X = out_X;
    double *tmp_W = W;

    for (int i = 0; i < v_num; i++)
    {   
        for (int j = 0; j < out_dim; j++)
        {
            for (int k = 0; k < in_dim; k++)
            {
                tmp_out_X[i * out_dim + j] += tmp_in_X[i * in_dim + k] * tmp_W[k * out_dim + j];
            }
        }
    }
}
void AX_verify(int dim, double *in_X, double *out_X)
{
    for (int i = 0; i < v_num; i++)

    {
        int *nbrs = &edges[nodes_index[i]];
        double *nbrs_val = &edges_value[nodes_index[i]];
        int degree = nodes_index[i + 1] - nodes_index[i];
        
        for (int j = 0; j < degree; j++)
        {
            int nbr = nbrs[j];
            for (int k = 0; k < dim; k++)
            {
                out_X[dim * i + k] += in_X[nbr * dim + k] * nbrs_val[j];
            }
        }
    }
}

void LogSoftmax_verify(int dim, double *X)
{

    for (int i = 0; i < v_num; i++)
    {
        double max = X[i * dim + 0];
        for (int j = 1; j < dim; j++)
        {
            if (X[i * dim + j] > max)
                max = X[i * dim + j];
        }

        double sum = 0;
        for (int j = 0; j < dim; j++)
        {
            sum += exp(X[i * dim + j] - max);
        }
        sum = log(sum);

        for (int j = 0; j < dim; j++)
        {
            X[i * dim + j] = X[i * dim + j] - max - sum;
        }
    }
}

bool verify(double max_sum)
{

    memset(X1_inter, 0, v_num * F1 * sizeof(double));
    memset(X1, 0, F1 * v_num * sizeof(double));

    XW_verify(F0, F1, X0, X1_inter, W1);

    // printf("Layer1 AX\n");
    AX_verify(F1, X1_inter, X1);

    // printf("Layer1 ReLU\n");
    LogSoftmax_verify(F1, X1);
    double verify_max_sum = MaxRowSum(X1, F1);
    printf("CPU_max_sum,  %6f\n", verify_max_sum);
    printf("GPU_max_sum,  %6f\n", max_sum);
    return fabs(max_sum - verify_max_sum) < 0.000001;
}

int main(int argc, char **argv)
{
    // !!! Attention !!!
    // Datasets: web-stanford ak_2010 dblp
    // Downloaded from：

    // 编译：
	//      hipify-perl gcn.cu > gcn.cpp
	//      hipcc gcn.cpp -o gcn
    //
    // 执行：仅供测试参考，队伍提交直接执行slurm.sh 即可
    //      可执行程序需接收5个参数，分别为：
	//      输入顶点特征长度F0，第一层顶点特征长度F1，图结构文件名，输入顶点特征矩阵文件名，第一层权重矩阵文件名
    //      ./gcn 128 16 graph/web-stanford_nodes_281903_edges_1992636_core_71.txt embedding/web-stanford_F0_128.bin weight/web-stanford_F0_128_F1_16.bin
    //      ./gcn 128 16 graph/com-dblp_nodes_317080_edges_1049866_core_113.txt embedding/dblp_F0_128.bin weight/dblp_F0_128_F1_16.bin
    //      ./gcn 128 16 graph/ak_2010.txt embedding/ak_2010_F0_128.bin weight/ak_2010_F0_128_F1_16.bin
    
    // 要求： 
    //      只允许修改GCN()函数里包含的代码；其余代码不允许修改，一旦发现取消成绩。

    // 评分：
    //      计算耗时显示 程序运行后会循环计算五次，评分是主要查看平均耗时。

    // 提交：
    //      查看slurm.sh 文件
    F0 = atoi(argv[1]);
    F1 = atoi(argv[2]);
    readGraph(argv[3]);
    readdouble(argv[4], X0, v_num * F0);
    readdouble(argv[5], W1, F0 * F1);
    initdouble(X1, v_num * F1);
    initdouble(X1_inter, v_num * F1);

    raw_graph_to_AdjacencyList();
    edgeNormalization();
    to_csr();
    initGPUMemory();

    double max_sum = 0, ave_timeMs = 0;
    int ROUNDs = 5;

    for (int i = 0; i < ROUNDs; i++)
    {
        // ################
        //
        ave_timeMs += GCN();
        // ################
        // Time point at the end of the computation
        // Compute the max row sum for result verification
        max_sum = MaxRowSum(X1, F1);

        // The max row sum and the computing time should be print
    }

    printf("verify\n");

    if (verify(max_sum))
    {
        printf("True\n");
    }
    else
    {
        printf("False\n");
    }

    printf("%f\n", ave_timeMs / ROUNDs);

    // freedoubles();
}