#include <math.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <fstream>
#include <vector>
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

__global__ void XW_(int in_dim, int out_dim, float *in_X, float *out_X, float *W, int v_num) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x; //控制v_vum

    if (tid >= v_num) return;

    float *tmp_in_X = in_X;
    float *tmp_out_X = out_X;
    float *tmp_W = W;

    for (int j = 0; j < out_dim; j++) {
        for (int k = 0; k < in_dim; k++) {
            tmp_out_X[tid * out_dim + j] += tmp_in_X[tid * in_dim + k] * tmp_W[k * out_dim + j];
        }
    }
}

__global__ void AX_(int dim, float *in_X, float *out_X, int *index, int *edges, float *edges_val, int v_num) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= v_num) return;

    int *nbrs = &edges[index[tid]];
    float *nbrs_val = &edges_val[index[tid]];

    int degree = index[tid + 1] - index[tid];

    for (int j = 0; j < degree; j++) {
        int nbr = nbrs[j];
        for (int k = 0; k < dim; k++) {
            out_X[dim * tid + k] += in_X[nbr * dim + k] * nbrs_val[j];
        }
    }
}

void LogSoftmax(int dim, float *X) {

    for (int i = 0; i < v_num; i++) {
        float max = X[i * dim + 0];
        for (int j = 1; j < dim; j++) {
            if (X[i * dim + j] > max) max = X[i * dim + j];
        }

        float sum = 0;
        for (int j = 0; j < dim; j++) {
            sum += exp(X[i * dim + j] - max);
        }
        sum = log(sum);

        for (int j = 0; j < dim; j++) {
            X[i * dim + j] = X[i * dim + j] - max - sum;
        }
    }
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

void initGPUMemory() {
    cudaMalloc(&d_X0, v_num * F0 * sizeof(float));
    cudaMemcpy(d_X0, X0, v_num * F0 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_X1_inter, v_num * F1 * sizeof(float));
    cudaMemcpy(d_X1_inter, X1_inter, v_num * F1 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_W1, F0 * F1 * sizeof(float));
    cudaMemcpy(d_W1, W1, F0 * F1 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_X1, F1 * v_num * sizeof(float));
    cudaMemcpy(d_X1, X1, F1 * v_num * sizeof(float), cudaMemcpyHostToDevice);

//    d_index, d_edge, d_edge_val

    cudaMalloc(&d_index, (v_num + 1) * sizeof(int));
    cudaMemcpy(d_index, nodes_index, (v_num + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_edges, e_num * sizeof(int));
    cudaMemcpy(d_edges, edges, e_num * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_edges_val, e_num * sizeof(float));
    cudaMemcpy(d_edges_val, edges_value, e_num * sizeof(float), cudaMemcpyHostToDevice);
}

void Preprocessing() {

    nodes_index = (int *) malloc(v_num * sizeof(int) + 1);

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

void GCN() {
    Preprocessing();
    initGPUMemory();
    const int block_size = 512;
    const int grid_size = v_num / block_size + 1;
    XW_<<<grid_size, block_size>>>(F0, F1, d_X0, d_X1_inter, d_W1, v_num);
    cudaMemcpy(X1_inter, d_X1_inter, v_num * F1 * sizeof(float), cudaMemcpyDeviceToHost);

    AX_<<<grid_size, block_size>>>(F1, d_X1_inter, d_X1, d_index, d_edges, d_edges_val, v_num);
    cudaMemcpy(X1, d_X1, sizeof(float) * v_num * F1, cudaMemcpyDeviceToHost);

    LogSoftmax(F1, X1);
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


    TimePoint start = chrono::steady_clock::now();

    for (int i = 0; i < 100; i++) {
        GCN();
    }
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