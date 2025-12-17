#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>

__constant__ float d_EPS2;

typedef struct {
    float x, y, z;
} Vec3;

#define EPS2 1e-9f

static const int *g_cluster_ids_for_sort = NULL;

static void pack_pos_mass(int N, const Vec3 *pos, const float *mass, float4 *posMass) {
    for (int i = 0; i < N; i++) {
        posMass[i].x = pos[i].x;
        posMass[i].y = pos[i].y;
        posMass[i].z = pos[i].z;
        posMass[i].w = mass[i];
    }
}

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _err = (call);                                           \
        if (_err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(_err));                               \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

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

int read_inputs(const char *mass_file,
                const char *coord_file,
                float **mass_out,
                Vec3 **pos_out) {
    int N = count_lines(mass_file);
    int coord_lines = count_lines(coord_file);
    if (coord_lines != N) {
        fprintf(stderr, "Input size mismatch between mass and coord files\n");
        exit(1);
    }

    float *mass = (float *)malloc(sizeof(float) * N);
    Vec3 *pos = (Vec3 *)malloc(sizeof(Vec3) * N);
    if (!mass || !pos) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

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
// // Comparator for qsort to sort indices based on cluster IDs
// int cmp_indices(const void *a, const void *b) {
//     int ia = *(const int *)a;
//     int ib = *(const int *)b;
//     int ca = g_cluster_ids_for_sort[ia];
//     int cb = g_cluster_ids_for_sort[ib];
//     if (ca < cb) return -1;
//     if (ca > cb) return 1;
//     if (ia < ib) return -1;
//     if (ia > ib) return 1;
//     return 0;
// }

// Build fixed-size clusters by grouping bodies sequentially.
// This implementation assumes that bodies are generated in a spatially
// coherent order, so explicit sorting or reordering is unnecessary.
// Each cluster is mapped to a single CUDA block for shared-memory reuse.
void build_clusters_fixed(int N,
                          int bodies_per_cluster,
                          int **cluster_offsets_out,
                          int **cluster_sizes_out,
                          int *cluster_count_out) {
    if (N % bodies_per_cluster != 0) {
        fprintf(stderr, "Total bodies (%d) not divisible by cluster size (%d)\n",
                N, bodies_per_cluster);
        exit(1);
    }

    int cluster_count = N / bodies_per_cluster;
    int *offsets = (int *)malloc(sizeof(int) * cluster_count);
    int *sizes = (int *)malloc(sizeof(int) * cluster_count);
    if (!offsets || !sizes) {
        fprintf(stderr, "Allocation failed for cluster metadata\n");
        exit(1);
    }

    for (int c = 0; c < cluster_count; ++c) {
        offsets[c] = c * bodies_per_cluster;
        sizes[c] = bodies_per_cluster;
    }

    *cluster_offsets_out = offsets;
    *cluster_sizes_out = sizes;
    *cluster_count_out = cluster_count;
}

__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai) {
    float3 r;
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + d_EPS2;

    float invDist = rsqrtf(distSqr);
    float invDist3 = invDist * invDist * invDist;
    float s = bj.w * invDist3;

    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

// Each CUDA block processes one cluster of bodies
// Bodies within a cluster are first loaded into shared memory
// Each thread computes the acceleration for one body by performing all pairwise interactions within the cluster
// This design reduces global memory traffic and exploits data reuse.
__global__ void hierarchical_kernel(const float4 *posMass,
                                    float4 *accOut,
                                    const int *cluster_offsets,
                                    const int *cluster_sizes) {
    extern __shared__ float4 s_pos[];

    int cluster_id = blockIdx.x;
    int cluster_start = cluster_offsets[cluster_id];
    int cluster_size = cluster_sizes[cluster_id];

    int tid = threadIdx.x;
    // global to shared
    if (tid < cluster_size) {
        s_pos[tid] = posMass[cluster_start + tid];
    }
    __syncthreads();

    if (tid >= cluster_size) return;

    float4 myPos = s_pos[tid];
    float3 acc = make_float3(0.0f, 0.0f, 0.0f);

    for (int j = 0; j < cluster_size; ++j) {
        if (j == tid) continue;
        acc = bodyBodyInteraction(myPos, s_pos[j], acc);
    }

    accOut[cluster_start + tid] = make_float4(acc.x, acc.y, acc.z, 0.0f);
}

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

int main(int argc, char *argv[]) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr,
                "Usage: %s <mass_file> <coord_file> [bodies_per_cluster] <output_file>\n",
                argv[0]);
        return 1;
    }

    const char *mass_file = argv[1];
    const char *coord_file = argv[2];
    int bodies_per_cluster = 256;
    const char *output_file = NULL;

    // Here we could set bodies_per_cluster from argv[3] if provided
    if (argc == 5) {
        bodies_per_cluster = atoi(argv[3]);
        output_file = argv[4];
    } else {  // argc == 4, default cluster size
        output_file = argv[3];
    }

    if (bodies_per_cluster <= 0) {
        fprintf(stderr, "Invalid bodies_per_cluster (got %d)\n", bodies_per_cluster);
        return 1;
    }

    float *mass = NULL;
    Vec3 *pos = NULL;

    int N = read_inputs(mass_file, coord_file, &mass, &pos);
    printf("Loaded %d bodies.\n", N);

    int *cluster_offsets = NULL;
    int *cluster_sizes = NULL;
    int cluster_count = 0;
    build_clusters_fixed(N, bodies_per_cluster,
                         &cluster_offsets, &cluster_sizes, &cluster_count);
    printf("Configured %d clusters of %d bodies each.\n", cluster_count, bodies_per_cluster);
    float4 *h_posMass = (float4 *)malloc(sizeof(float4) * N);
    float4 *h_acc4 = (float4 *)malloc(sizeof(float4) * N);
    if (!h_posMass || !h_acc4) {
        fprintf(stderr, "Host float4 allocation failed\n");
        exit(1);
    }

    pack_pos_mass(N, pos, mass, h_posMass);

    // allcate the memory on device
    // Allocate global memory on the GPU.
    // d_posMass : packed position (x,y,z) and mass (w) for all bodies
    // d_acc     : output acceleration vector for each body
    // d_cluster_offsets/sizes : cluster metadata used for hierarchical blocking
    float4 *d_posMass = NULL; 
    float4 *d_acc = NULL; 
    int *d_cluster_offsets = NULL;  
    int *d_cluster_sizes = NULL; 
    CUDA_CHECK(cudaMalloc(&d_posMass, sizeof(float4) * N));
    CUDA_CHECK(cudaMalloc(&d_acc, sizeof(float4) * N));
    CUDA_CHECK(cudaMalloc(&d_cluster_offsets, sizeof(int) * cluster_count));
    CUDA_CHECK(cudaMalloc(&d_cluster_sizes, sizeof(int) * cluster_count));

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // Copy input data and cluster information to GPU
    CUDA_CHECK(cudaMemcpy(d_posMass, h_posMass, sizeof(float4) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cluster_offsets, cluster_offsets, sizeof(int) * cluster_count,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cluster_sizes, cluster_sizes, sizeof(int) * cluster_count,
                          cudaMemcpyHostToDevice));

    float eps2_h = EPS2;
    CUDA_CHECK(cudaMemcpyToSymbol(d_EPS2, &eps2_h, sizeof(float)));

    int blockSize = 0;
    for (int i = 0; i < cluster_count; ++i) {
        if (cluster_sizes[i] > blockSize) blockSize = cluster_sizes[i];
    }
    int gridSize = cluster_count;
    size_t sharedMemSize = blockSize * sizeof(float4);

    hierarchical_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_posMass, d_acc, d_cluster_offsets, d_cluster_sizes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_acc4, d_acc, sizeof(float4) * N, cudaMemcpyDeviceToHost));

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Hierarchical kernel time: %.6f seconds\n", elapsed);

    write_outputs(output_file, N, h_acc4);

    CUDA_CHECK(cudaFree(d_posMass));
    CUDA_CHECK(cudaFree(d_acc));
    CUDA_CHECK(cudaFree(d_cluster_offsets));
    CUDA_CHECK(cudaFree(d_cluster_sizes));
    free(h_posMass);
    free(h_acc4);
    free(cluster_offsets);
    free(cluster_sizes);
    free(mass);
    free(pos);

    return 0;
}
