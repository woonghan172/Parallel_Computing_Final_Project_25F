#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h> 

__constant__ float d_EPS2;

typedef struct {
    float x, y, z;
} Vec3;

#define EPS2 1e-9f   // softening factor

static void pack_pos_mass(int N, const Vec3 *pos, const float *mass, float4 *posMass) {
    for (int i = 0; i < N; i++) {
        posMass[i].x = pos[i].x;
        posMass[i].y = pos[i].y;
        posMass[i].z = pos[i].z;
        posMass[i].w = mass[i]; 
    }
}

#define CUDA_CHECK(call)                                                     \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(_err));                              \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// ------------------------------
// Count number of bodies (lines)
// ------------------------------
int count_lines(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("fopen");
        exit(1);
    }

    int count = 0;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), fp)) {
        count++;
    }
    fclose(fp);
    return count;
}

// --------------------------------------------------------
// Function 1: Read mass + position inputs
// --------------------------------------------------------
int read_inputs(
        const char *mass_file,
        const char *coord_file,
        float **mass_out,
        Vec3 **pos_out)
{
    int N = count_lines(mass_file);
    int coord_lines = count_lines(coord_file);
    if (coord_lines != N) {
        fprintf(stderr, "Input size mismatch: %s has %d lines, %s has %d lines\n",
                mass_file, N, coord_file, coord_lines);
        exit(1);
    }

    float *mass = (float *)malloc(sizeof(float) * N);
    Vec3  *pos  = (Vec3  *)malloc(sizeof(Vec3)  * N);

    if (!mass || !pos) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Read masses
    FILE *fm = fopen(mass_file, "r");
    if (!fm) {
        perror("fopen mass file");
        exit(1);
    }
    for (int i = 0; i < N; i++) {
        if (fscanf(fm, "%f", &mass[i]) != 1) {
            fprintf(stderr, "Invalid mass at line %d\n", i + 1);
            exit(1);
        }
    }
    fclose(fm);

    // Read coordinates
    FILE *fc = fopen(coord_file, "r");
    if (!fc) {
        perror("fopen coordinate file");
        exit(1);
    }
    for (int i = 0; i < N; i++) {
        if (fscanf(fc, "%f %f %f",
                   &pos[i].x, &pos[i].y, &pos[i].z) != 3) {
            fprintf(stderr, "Invalid coordinate at line %d\n", i + 1);
            exit(1);
        }
    }
    fclose(fc);

    *mass_out = mass;
    *pos_out = pos;

    return N;
}

// --------------------------------------------------------
// Function: about the gems kernel
// --------------------------------------------------------
__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai) {
    float3 r;

    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + d_EPS2;

    // use rsqrtf for faster computation
    float invDist  = rsqrtf(distSqr);
    float invDist3 = invDist * invDist * invDist;

    float s = bj.w * invDist3; // bj.w = mass_j, G=1

    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}
// shared memory tiled version
__global__ void compute_forces_kernel(const float4 *posMass, float4 *accOut, int N) {
    extern __shared__ float4 shPos[];  // dynamic shared memory

    int tid  = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;

    bool isActive = (gtid < N);
    float4 myPos = isActive ? posMass[gtid] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 acc   = make_float3(0.0f, 0.0f, 0.0f);

    int numTiles = (N + blockDim.x - 1) / blockDim.x;

    for (int tile = 0; tile < numTiles; tile++) {
        int j = tile * blockDim.x + tid;
        if (j < N) {
            shPos[tid] = posMass[j];
        }
        // avoid the previous tile's data being used
        else {
            shPos[tid] = make_float4(0,0,0,0);
        }
        __syncthreads();

        int tileSize = min(blockDim.x, N - tile * blockDim.x);

        if (isActive) {
            for (int k = 0; k < tileSize; k++) {
                acc = bodyBodyInteraction(myPos, shPos[k], acc);
            }
        }
        __syncthreads();
    }

    if (isActive) {
        accOut[gtid] = make_float4(acc.x, acc.y, acc.z, 0.0f);
    }
}
// --------------------------------------------------------
// Function 3: Write output to file
// --------------------------------------------------------
void write_outputs(const char *output_file, int N, const float4 *acc) {
    FILE *fo = fopen(output_file, "w");
    if (!fo) {
        perror("fopen output file");
        exit(1);
    }

    for (int i = 0; i < N; i++) {
        fprintf(fo, "%.6f %.6f %.6f\n", acc[i].x, acc[i].y, acc[i].z);
    }

    fclose(fo);
}

// --------------------------------------------------------
// Main with timing
// --------------------------------------------------------
int main(int argc, char *argv[]) {

    if (argc != 4) {
        fprintf(stderr, "Usage: %s <mass_file> <coord_file> <output_file>\n", argv[0]);
        return 1;
    }

    const char *mass_file   = argv[1];
    const char *coord_file  = argv[2];
    const char *output_file = argv[3];

    float *mass = NULL;
    Vec3  *pos  = NULL;

    // Read inputs
    int N = read_inputs(mass_file, coord_file, &mass, &pos);
    printf("Loaded %d bodies.\n", N);

    // use float4 for better memory alignment
    float4 *h_posMass = (float4 *)malloc(sizeof(float4) * N);
    float4 *h_acc4    = (float4 *)malloc(sizeof(float4) * N);
    if (!h_posMass || !h_acc4) {
        fprintf(stderr, "Host float4 allocation failed\n");
        exit(1);
    }

    pack_pos_mass(N, pos, mass, h_posMass);

    float4 *d_posMass = NULL;
    float4 *d_acc     = NULL;
    CUDA_CHECK(cudaMalloc(&d_posMass, sizeof(float4) * N));
    CUDA_CHECK(cudaMalloc(&d_acc,     sizeof(float4) * N));

    struct timespec start, end;

    // --------------------------
    // GPU transfer + kernel timing
    // --------------------------
    clock_gettime(CLOCK_MONOTONIC, &start);

    CUDA_CHECK(cudaMemcpy(d_posMass, h_posMass, sizeof(float4) * N,
                          cudaMemcpyHostToDevice));

    float eps2_h = EPS2;
    CUDA_CHECK(cudaMemcpyToSymbol(d_EPS2, &eps2_h, sizeof(float)));

    // setting of kernel
    int blockSize = 128;  // could be 256 or 512 as well (I need to figure out the best one)
    int gridSize  = (N + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(float4);

    compute_forces_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_posMass, d_acc, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());  // wait for kernel to finish

    CUDA_CHECK(cudaMemcpy(h_acc4, d_acc, sizeof(float4) * N, cudaMemcpyDeviceToHost));

    clock_gettime(CLOCK_MONOTONIC, &end);
    // --------------------------

    double elapsed =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Computation time: %.6f seconds\n", elapsed);

    // Write results
    write_outputs(output_file, N, h_acc4);

    CUDA_CHECK(cudaFree(d_posMass));
    CUDA_CHECK(cudaFree(d_acc));
    free(h_posMass);
    free(h_acc4);
    free(mass);
    free(pos);

    return 0;
}
