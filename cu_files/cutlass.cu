// nbody_cutlass_shared.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

// CUTLASS
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

typedef struct { float x, y, z; } Vec3;

#define EPS2 1e-9f

#define CHECK_CUDA(call) do {                                 \
  cudaError_t e = (call);                                     \
  if (e != cudaSuccess) {                                     \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                 \
            __FILE__, __LINE__, cudaGetErrorString(e));       \
    std::exit(1);                                             \
  }                                                           \
} while(0)

// ------------------------------
// Count number of bodies (lines)
// ------------------------------
int count_lines(const char *filename) {
  FILE *fp = fopen(filename, "r");
  if (!fp) { perror("fopen"); std::exit(1); }
  int count = 0;
  char buffer[256];
  while (fgets(buffer, sizeof(buffer), fp)) count++;
  fclose(fp);
  return count;
}

// --------------------------------------------------------
// Read mass + position inputs
// --------------------------------------------------------
int read_inputs(const char *mass_file,
                const char *coord_file,
                float **mass_out,
                Vec3 **pos_out)
{
  int N = count_lines(mass_file);

  float *mass = (float*)malloc(sizeof(float) * N);
  Vec3  *pos  = (Vec3*) malloc(sizeof(Vec3)  * N);
  if (!mass || !pos) { fprintf(stderr, "malloc failed\n"); std::exit(1); }

  FILE *fm = fopen(mass_file, "r");
  if (!fm) { perror("fopen mass file"); std::exit(1); }
  for (int i = 0; i < N; i++) {
    if (fscanf(fm, "%f", &mass[i]) != 1) {
      fprintf(stderr, "Invalid mass at line %d\n", i + 1);
      std::exit(1);
    }
  }
  fclose(fm);

  FILE *fc = fopen(coord_file, "r");
  if (!fc) { perror("fopen coordinate file"); std::exit(1); }
  for (int i = 0; i < N; i++) {
    if (fscanf(fc, "%f %f %f", &pos[i].x, &pos[i].y, &pos[i].z) != 3) {
      fprintf(stderr, "Invalid coordinate at line %d\n", i + 1);
      std::exit(1);
    }
  }
  fclose(fc);

  *mass_out = mass;
  *pos_out  = pos;
  return N;
}

// --------------------------------------------------------
// Write output to file
// --------------------------------------------------------
void write_outputs(const char *output_file, int N, const Vec3 *acc) {
  FILE *fo = fopen(output_file, "w");
  if (!fo) { perror("fopen output file"); std::exit(1); }
  for (int i = 0; i < N; i++) {
    fprintf(fo, "%.6f %.6f %.6f\n", acc[i].x, acc[i].y, acc[i].z);
  }
  fclose(fo);
}

// ======================= GPU kernels =======================

// Pack AoS Vec3 -> row-major [N x 3]
__global__ void pack_pos_rowmajor3(const Vec3* __restrict__ pos,
                                   float* __restrict__ pos_rm,
                                   int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    pos_rm[i*3 + 0] = pos[i].x;
    pos_rm[i*3 + 1] = pos[i].y;
    pos_rm[i*3 + 2] = pos[i].z;
  }
}

// Build posT row-major: [3 x N], row stride = N
__global__ void make_posT_rowmajor(const float* __restrict__ pos_rm,
                                   float* __restrict__ posT_rm,
                                   int N) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < N) {
    posT_rm[0*N + j] = pos_rm[j*3 + 0];
    posT_rm[1*N + j] = pos_rm[j*3 + 1];
    posT_rm[2*N + j] = pos_rm[j*3 + 2];
  }
}

__global__ void compute_norm2(const float* __restrict__ pos_rm,
                              float* __restrict__ norm2,
                              int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float x = pos_rm[i*3 + 0];
    float y = pos_rm[i*3 + 1];
    float z = pos_rm[i*3 + 2];
    norm2[i] = x*x + y*y + z*z;
  }
}

__global__ void zero_acc(Vec3* __restrict__ acc, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) acc[i] = {0.f, 0.f, 0.f};
}

// Accumulate for one (i-tile, j-tile) with shared-memory caching of j data.
// dotTile has row-stride = ldc (fixed JB).
__global__ void accumulate_tile_sharedJ(
    int N, int i0, int j0,
    int curIB, int curJB,
    int ldc,
    const float* __restrict__ pos_rm,   // [N x 3], row-major
    const float* __restrict__ norm2,    // [N]
    const float* __restrict__ mass,     // [N]
    const float* __restrict__ dotTile,  // [curIB x curJB] logically, stride=ldc
    Vec3* __restrict__ acc)             // [N]
{
  extern __shared__ float4 shPj[]; // curJB entries: {x,y,z,mass}

  // Cooperative load of the j-tile into shared memory
  for (int tj = threadIdx.x; tj < curJB; tj += blockDim.x) {
    int j = j0 + tj;
    if (j < N) {
      float xj = pos_rm[j*3 + 0];
      float yj = pos_rm[j*3 + 1];
      float zj = pos_rm[j*3 + 2];
      float mj = mass[j];
      shPj[tj] = make_float4(xj, yj, zj, mj);
    } else {
      shPj[tj] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
  __syncthreads();

  // Each thread computes one i within this i-tile
  int ti = blockIdx.x * blockDim.x + threadIdx.x;
  if (ti >= curIB) return;

  int i = i0 + ti;
  if (i >= N) return;

  float xi = pos_rm[i*3 + 0];
  float yi = pos_rm[i*3 + 1];
  float zi = pos_rm[i*3 + 2];

  float ax = 0.f, ay = 0.f, az = 0.f;
  float ni = norm2[i];

  // Loop over j in this tile (using shared memory for x,y,z,mass)
  for (int tj = 0; tj < curJB; tj++) {
    int j = j0 + tj;
    if (j >= N) break;
    if (i == j) continue;

    // Dot product from CUTLASS result (stride = ldc)
    float dot = dotTile[ti * ldc + tj];

    // dist^2 = ||pi||^2 + ||pj||^2 - 2*pi·pj + EPS2
    float dist2 = ni + norm2[j] - 2.0f * dot + EPS2;

    // Match serial reference for strict correctness
    float invDist  = 1.0f / sqrtf(dist2);
    float invDist3 = invDist * invDist * invDist;

    float4 pj = shPj[tj];         // {xj,yj,zj,massj}
    float s = pj.w * invDist3;    // mass[j] * invDist^3 (G=1)

    float rx = pj.x - xi;
    float ry = pj.y - yi;
    float rz = pj.z - zi;

    ax += rx * s;
    ay += ry * s;
    az += rz * s;
  }

  // Accumulate into global (safe: i-tile rows are disjoint across blocks)
  acc[i].x += ax;
  acc[i].y += ay;
  acc[i].z += az;
}

// ======================= CUTLASS GEMM type =======================
// dotTile = A(curIB x 3) * B(3 x curJB), row-major.
// A: lda=3
// B: ldb=N (posT row-major [3 x N])
// C: ldc=JB (fixed stride for dotTile buffer)
using Layout = cutlass::layout::RowMajor;
using GemmDot = cutlass::gemm::device::Gemm<
    float, Layout,
    float, Layout,
    float, Layout,
    float
>;

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <mass_file> <coord_file> <output_file>\n", argv[0]);
    return 1;
  }

  const char* mass_file   = argv[1];
  const char* coord_file  = argv[2];
  const char* output_file = argv[3];

  float* h_mass = nullptr;
  Vec3*  h_pos  = nullptr;
  int N = read_inputs(mass_file, coord_file, &h_mass, &h_pos);
  printf("Loaded %d bodies.\n", N);

  std::vector<Vec3> h_acc(N);

  // Device allocations
  Vec3*  d_pos_struct = nullptr;
  Vec3*  d_acc        = nullptr;
  float* d_mass       = nullptr;

  float* d_pos_rm   = nullptr; // [N x 3]
  float* d_posT_rm  = nullptr; // [3 x N]
  float* d_norm2    = nullptr; // [N]

  CHECK_CUDA(cudaMalloc(&d_pos_struct, sizeof(Vec3)  * N));
  CHECK_CUDA(cudaMalloc(&d_acc,        sizeof(Vec3)  * N));
  CHECK_CUDA(cudaMalloc(&d_mass,       sizeof(float) * N));
  CHECK_CUDA(cudaMalloc(&d_pos_rm,     sizeof(float) * N * 3));
  CHECK_CUDA(cudaMalloc(&d_posT_rm,    sizeof(float) * 3 * N));
  CHECK_CUDA(cudaMalloc(&d_norm2,      sizeof(float) * N));

  CHECK_CUDA(cudaMemcpy(d_pos_struct, h_pos,  sizeof(Vec3)  * N, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_mass,       h_mass, sizeof(float) * N, cudaMemcpyHostToDevice));

  // Precompute packed pos, posT, norm2, and zero acc
  {
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    pack_pos_rowmajor3<<<blocks, threads>>>(d_pos_struct, d_pos_rm, N);
    make_posT_rowmajor<<<blocks, threads>>>(d_pos_rm, d_posT_rm, N);
    compute_norm2<<<blocks, threads>>>(d_pos_rm, d_norm2, N);
    zero_acc<<<blocks, threads>>>(d_acc, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  // Tile sizes (tuneable)
  constexpr int IB = 128;
  constexpr int JB = 128;

  // dotTile buffer with fixed stride JB
  float* d_dotTile = nullptr;
  CHECK_CUDA(cudaMalloc(&d_dotTile, sizeof(float) * IB * JB));

  GemmDot gemm;

  // GPU timing
  cudaEvent_t ev0, ev1;
  CHECK_CUDA(cudaEventCreate(&ev0));
  CHECK_CUDA(cudaEventCreate(&ev1));
  CHECK_CUDA(cudaEventRecord(ev0));

  // Main tiled computation:
  for (int i0 = 0; i0 < N; i0 += IB) {
    int curIB = std::min(IB, N - i0);

    for (int j0 = 0; j0 < N; j0 += JB) {
      int curJB = std::min(JB, N - j0);

      // A: [curIB x 3], row-major, lda=3
      const float* A = d_pos_rm + i0 * 3;

      // B: [3 x curJB] view into posT_rm [3 x N], row-major, ldb=N
      const float* B = d_posT_rm + j0;

      // C/D: dotTile [curIB x curJB], row-major, ldc=JB (fixed stride)
      float* C = d_dotTile;

      cutlass::gemm::GemmCoord problem_size(curIB, curJB, 3);

      typename GemmDot::Arguments args(
          problem_size,
          {A, 3},     // A, lda
          {B, N},     // B, ldb (row stride N)
          {C, JB},    // C, ldc (fixed JB)
          {C, JB},    // D, ldd
          {1.0f, 0.0f}
      );

      cutlass::Status status = gemm(args);
      if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS GEMM failed at tile i0=%d j0=%d\n", i0, j0);
        std::exit(1);
      }

      // Accumulate using shared memory for j-tile positions + masses
      int threads = 128;
      int blocks  = (curIB + threads - 1) / threads;
      size_t shmem_bytes = (size_t)curJB * sizeof(float4);

      accumulate_tile_sharedJ<<<blocks, threads, shmem_bytes>>>(
          N, i0, j0,
          curIB, curJB,
          JB,               // ldc = fixed stride used in CUTLASS
          d_pos_rm, d_norm2, d_mass,
          d_dotTile,
          d_acc
      );
      CHECK_CUDA(cudaGetLastError());
    }
  }

  CHECK_CUDA(cudaEventRecord(ev1));
  CHECK_CUDA(cudaEventSynchronize(ev1));
  float ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, ev0, ev1));
  printf("Computation time: %.6f seconds\n", ms / 1000.0f);

  // Copy back and write
  CHECK_CUDA(cudaMemcpy(h_acc.data(), d_acc, sizeof(Vec3) * N, cudaMemcpyDeviceToHost));
  write_outputs(output_file, N, h_acc.data());

  // Cleanup
  CHECK_CUDA(cudaFree(d_dotTile));
  CHECK_CUDA(cudaFree(d_norm2));
  CHECK_CUDA(cudaFree(d_posT_rm));
  CHECK_CUDA(cudaFree(d_pos_rm));
  CHECK_CUDA(cudaFree(d_mass));
  CHECK_CUDA(cudaFree(d_acc));
  CHECK_CUDA(cudaFree(d_pos_struct));
  CHECK_CUDA(cudaEventDestroy(ev0));
  CHECK_CUDA(cudaEventDestroy(ev1));

  free(h_mass);
  free(h_pos);

  return 0;
}
