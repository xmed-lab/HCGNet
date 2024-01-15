#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "interpolate_gpu.h"


__global__ void three_nn_kernel_fast(int b, int n, int m, const float *__restrict__ unknown, 
    const float *__restrict__ known, float *__restrict__ dist2, int *__restrict__ idx) {
    // unknown: (B, N, 3)
    // known: (B, M, 3)
    // output: 
    //      dist2: (B, N, 3)
    //      idx: (B, N, 3)
    
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= n) return;

    unknown += bs_idx * n * 3 + pt_idx * 3;
    known += bs_idx * m * 3;
    dist2 += bs_idx * n * 3 + pt_idx * 3;
    idx += bs_idx * n * 3 + pt_idx * 3;

    float ux = unknown[0];
    float uy = unknown[1];
    float uz = unknown[2];

    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
    int besti1 = 0, besti2 = 0, besti3 = 0;
    for (int k = 0; k < m; ++k) {
        float x = known[k * 3 + 0];
        float y = known[k * 3 + 1];
        float z = known[k * 3 + 2];
        float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
        if (d < best1) {
            best3 = best2; besti3 = besti2;
            best2 = best1; besti2 = besti1;
            best1 = d; besti1 = k;
        } 
        else if (d < best2) {
            best3 = best2; besti3 = besti2;
            best2 = d; besti2 = k;
        } 
        else if (d < best3) {
            best3 = d; besti3 = k;
        }
    }
    dist2[0] = best1; dist2[1] = best2; dist2[2] = best3;
    idx[0] = besti1; idx[1] = besti2; idx[2] = besti3;
}


void three_nn_kernel_launcher_fast(int b, int n, int m, const float *unknown, 
    const float *known, float *dist2, int *idx, cudaStream_t stream) {
    // unknown: (B, N, 3)
    // known: (B, M, 3)
    // output: 
    //      dist2: (B, N, 3)
    //      idx: (B, N, 3)

    cudaError_t err;
    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    three_nn_kernel_fast<<<blocks, threads, 0, stream>>>(b, n, m, unknown, known, dist2, idx);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void three_nn_grad_kernel_fast(int b, int n, int m, const float *__restrict__ grad_out, 
    const int *__restrict__ idx, const float *__restrict__ unknown_xyz, const float *__restrict__ known_xyz, 
    float *__restrict__ grad_xyz, float *__restrict__ grad_unknown) {
    // grad_out: dist (B, N, 3)
    // unknown_xyz: (B, N, 3)
    // idx: (B, N, 3)
    // output: (gradient to known points xyz)
    //      grad_xyz: (B, M, 3)
    int bs_idx = blockIdx.z;
    int pt_idx = blockIdx.y;
    int tp_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bs_idx >= b || pt_idx >= n || tp_idx >= 3) return;

    // dist = (x - x')^2 + (y - y')^2 + (z - z')^2
    grad_unknown += bs_idx * 3 * n + pt_idx * 3; // locate unknown point idx
    unknown_xyz += bs_idx * 3 * n + pt_idx * 3; // locate point idx
    grad_out += bs_idx * 3 * n + pt_idx * 3 + tp_idx; // locate dist, idx
    idx += bs_idx * 3 * n + pt_idx * 3 + tp_idx;
    grad_xyz += bs_idx * 3 * m + idx[0] * 3;
    known_xyz += bs_idx * 3 * m + idx[0] * 3;

    atomicAdd(grad_xyz + 0, 2 * (known_xyz[0] - unknown_xyz[0]) * grad_out[0]);
    atomicAdd(grad_xyz + 1, 2 * (known_xyz[1] - unknown_xyz[1]) * grad_out[0]);
    atomicAdd(grad_xyz + 2, 2 * (known_xyz[2] - unknown_xyz[2]) * grad_out[0]);

    atomicAdd(grad_unknown + 0, 2 * (unknown_xyz[0] - known_xyz[0]) * grad_out[0]);
    atomicAdd(grad_unknown + 1, 2 * (unknown_xyz[1] - known_xyz[1]) * grad_out[0]);
    atomicAdd(grad_unknown + 2, 2 * (unknown_xyz[2] - known_xyz[2]) * grad_out[0]);
}

void three_nn_grad_kernel_launcher_fast(int b, int n, int m, const float *grad_out, 
    const int *idx, const float *unknown_xyz, const float *known_xyz, 
    float *grad_xyz, float *grad_unknown, cudaStream_t stream) {
    cudaError_t err;
    dim3 blocks(DIVUP(3, THREADS_PER_BLOCK), n, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    three_nn_grad_kernel_fast<<<blocks, threads, 0, stream>>>(b, n, m, grad_out, idx, unknown_xyz, known_xyz, grad_xyz, grad_unknown);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void three_interpolate_kernel_fast(int b, int c, int m, int n, const float *__restrict__ points, 
    const int *__restrict__ idx, const float *__restrict__ weight, float *__restrict__ out) {
    // points: (B, C, M)
    // idx: (B, N, 3)
    // weight: (B, N, 3)
    // output:
    //      out: (B, C, N)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bs_idx >= b || c_idx >= c || pt_idx >= n) return;

    weight += bs_idx * n * 3 + pt_idx * 3;
    points += bs_idx * c * m + c_idx * m;
    idx += bs_idx * n * 3 + pt_idx * 3;
    out += bs_idx * c * n + c_idx * n;

    out[pt_idx] = weight[0] * points[idx[0]] + weight[1] * points[idx[1]] + weight[2] * points[idx[2]];
}

void three_interpolate_kernel_launcher_fast(int b, int c, int m, int n, 
    const float *points, const int *idx, const float *weight, float *out, cudaStream_t stream) {
    // points: (B, C, M)
    // idx: (B, N, 3)
    // weight: (B, N, 3)
    // output:
    //      out: (B, C, N)

    cudaError_t err;
    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    three_interpolate_kernel_fast<<<blocks, threads, 0, stream>>>(b, c, m, n, points, idx, weight, out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__global__ void three_interpolate_grad_kernel_fast(int b, int c, int n, int m, const float *__restrict__ grad_out, 
    const int *__restrict__ idx, const float *__restrict__ weight, const float *__restrict__ feats, float *__restrict__ grad_wts, float *__restrict__ grad_points) {
    // grad_out: (B, C, N)
    // weight: (B, N, 3)
    // feats: (B, C, M)
    // output:
    //      grad_wts: (B, N, 3)
    //      grad_points: (B, C, M)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bs_idx >= b || c_idx >= c || pt_idx >= n) return;
    
    grad_out += bs_idx * c * n + c_idx * n + pt_idx;
    idx += bs_idx * n * 3 + pt_idx * 3;
    weight += bs_idx * n * 3 + pt_idx * 3;
    feats += bs_idx * c * m + c_idx * m; // locate channel
    
    grad_wts += bs_idx * n * 3 + pt_idx * 3;
    grad_points += bs_idx * c * m + c_idx * m;

    atomicAdd(grad_points + idx[0], grad_out[0] * weight[0]);
    atomicAdd(grad_points + idx[1], grad_out[0] * weight[1]);
    atomicAdd(grad_points + idx[2], grad_out[0] * weight[2]);

    atomicAdd(grad_wts + 0, grad_out[0] * (feats + idx[0])[0]);
    atomicAdd(grad_wts + 1, grad_out[0] * (feats + idx[1])[0]);
    atomicAdd(grad_wts + 2, grad_out[0] * (feats + idx[2])[0]);
}

void three_interpolate_grad_kernel_launcher_fast(int b, int c, int n, int m, const float *grad_out, 
    const int *idx, const float *weight, const float *feats, float *grad_wts, float *grad_points, cudaStream_t stream) {
    // grad_out: (B, C, N)
    // weight: (B, N, 3)
    // output:
    //      grad_points: (B, C, M)

    cudaError_t err;
    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    three_interpolate_grad_kernel_fast<<<blocks, threads, 0, stream>>>(b, c, n, m, grad_out, idx, weight, feats, grad_wts, grad_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}