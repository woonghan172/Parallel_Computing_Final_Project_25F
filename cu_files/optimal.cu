// Nicholas Part 2
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>


#define COARSENING_FACTOR 4
#define THREAD_PER_BLOCK 64

#define EPS2 1e-9f   // softening factor

#define CHECK_CUDA(call)                                                      \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__,                    \
             cudaGetErrorString(_e));                                         \
    }                                                                         \
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
        float **mass_out)
{
    int N = count_lines(mass_file);

    float *full_array = (float *)malloc(sizeof(float) * N * 4);

    if (!full_array) {
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
        if (fscanf(fm, "%f", &full_array[4*i + 3]) != 1) {
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
                   &full_array[4*i], &full_array[4*i + 1], &full_array[4*i + 2]) != 3) {
            fprintf(stderr, "Invalid coordinate at line %d\n", i + 1);
            exit(1);
        }
    }
    fclose(fc);

    *mass_out = full_array;

    return N;
}

// --------------------------------------------------------
// Function 2: GPU Gems 3 Interaction Code
// --------------------------------------------------------

   __device__ float4
bodyBodyInteraction(float4 bi, float4 bj, float4 ai)
{
  float3 r;
  // r_ij  [3 FLOPS]
  r.x = bj.x - bi.x;
  r.y = bj.y - bi.y;
  r.z = bj.z - bi.z;
  // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
  // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
  float invDist  = rsqrtf(distSqr);
  float invDist3 = invDist * invDist * invDist;
//    float distSixth = distSqr * distSqr * distSqr;
//   float invDistCube = 1.0f/sqrtf(distSixth);
  // s = m_j * invDistCube [1 FLOP]
   float s = bj.w * invDist3;
  // a_i =  a_i + s * r_ij [6 FLOPS]
  ai.x += r.x * s;
  ai.y += r.y * s;
  ai.z += r.z * s;
  return ai;
}

// --------------------------------------------------------
// Function 3: Within Tile Calculations
// --------------------------------------------------------

__device__ void
withinTileCalc(int num_items, int shared_size, float *my_coord_mass, float *shared_coord_mass, float *my_accel){
    for (int i = 0; i < num_items; i++){
        for (int ele=0; ele<shared_size; ele++){
            reinterpret_cast<float4 *>(&my_accel[i*4])[0] = bodyBodyInteraction(reinterpret_cast<float4 *>(&my_coord_mass[i*4])[0],
                reinterpret_cast<float4 *>(&shared_coord_mass[ele*4])[0], reinterpret_cast<float4 *>(&my_accel[i*4])[0]);
        }
    }
}

// --------------------------------------------------------
// Function 4: Kernel with Tiling and Coarsening
// --------------------------------------------------------

__global__ void
coarseningKernel(int N, int C_factor, int total_threads, float *d_coord_mass, float *d_accel){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int local_idx = threadIdx.x;
    float4 zero_vector = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    //Error Condition
    if (THREAD_PER_BLOCK%blockDim.x != 0){
        printf("Incorrect Tile Dimension\n");
        return;
    }

    //Assign Local Memory for Positions and Accelerations
    float localPositions[sizeof(float)*4*COARSENING_FACTOR]  = {0.f};
    float localAccels[sizeof(float)*4*COARSENING_FACTOR]  = {0.f};

    //Read the Positions from Global Memory
    for (int Citer = 0; Citer<C_factor; Citer++){
        if ((Citer*total_threads)+idx<N){
            reinterpret_cast<float4 *>(&localPositions[Citer*4])[0] = reinterpret_cast<float4 *>(&d_coord_mass[(total_threads*Citer+idx)*4])[0];
        } else {
            reinterpret_cast<float4 *>(&localPositions[Citer*4])[0] = zero_vector;
        }
    }
    __syncthreads();

    //Load tiles of Tile_size into Shared Memory
    __shared__ float shared_tile[sizeof(float)*4*THREAD_PER_BLOCK];

    for (int tile_iter = 0; tile_iter<((N-1)/THREAD_PER_BLOCK)+1; tile_iter++){
        if (tile_iter * THREAD_PER_BLOCK + local_idx < N){
            reinterpret_cast<float4 *>(&shared_tile[(local_idx)*4])[0] = 
            reinterpret_cast<float4 *>(&d_coord_mass[(tile_iter * THREAD_PER_BLOCK + local_idx)*4])[0];
        } else {
            reinterpret_cast<float4 *>(&shared_tile[(local_idx)*4])[0] = zero_vector;
        }
        __syncthreads();

        //Perform the Computations
        for (int i = 0; i < COARSENING_FACTOR; i++)
            for (int k = 0; k < THREAD_PER_BLOCK; k++)
                reinterpret_cast<float4 *>(&localAccels[i*4])[0] = bodyBodyInteraction(reinterpret_cast<float4 *>(&localPositions[i*4])[0],
                reinterpret_cast<float4 *>(&shared_tile[k*4])[0], reinterpret_cast<float4 *>(&localAccels[i*4])[0]);
        __syncthreads();


    }

    //Write Results to d_accel
    for (int Citer = 0; Citer<C_factor; Citer++){
        if ((Citer*total_threads)+idx<N){
            reinterpret_cast<float4 *>(&d_accel[(total_threads*Citer+idx)*4])[0] = reinterpret_cast<float4 *>(&localAccels[Citer*4])[0];
        }
    }
}

// --------------------------------------------------------
// Function 5: Write output to file
// --------------------------------------------------------
void write_outputs(const char *output_file, int N, float *acc) {
    FILE *fo = fopen(output_file, "w");
    if (!fo) {
        perror("fopen output file");
        exit(1);
    }

    for (int i = 0; i < N; i++) {
        fprintf(fo, "%.6f %.6f %.6f\n",  acc[4*i], acc[4*i + 1], acc[4*i + 2]);
    }

    fclose(fo);
}

int main(int argc, char *argv[]){

    const char *mass_file   = argv[1];
    const char *coord_file  = argv[2];
    const char *output_file = argv[3];

    //Host Input Matrix
    float *coord_mass = NULL;

    //Read inputs
    int N = read_inputs(mass_file, coord_file, &coord_mass);

    //Host Output Matrix
    float *accel = (float *)calloc(N * 4, sizeof(float));

    //Initialize and Write to Device Memory
    float *device_coord_mass = NULL, *device_accel = NULL;
    CHECK_CUDA(cudaMalloc((void **)&device_coord_mass, (size_t)N * 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&device_accel, (size_t)N * 4 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(device_coord_mass, coord_mass, (size_t)N * 4  * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_accel, accel, (size_t)N * 4  * sizeof(float), cudaMemcpyHostToDevice));

    //Run the Kernel
    int num_blocks = ((N-1)/(THREAD_PER_BLOCK*COARSENING_FACTOR))+1;
    dim3 grid(num_blocks);
    dim3 block(THREAD_PER_BLOCK);

    // --------------------------
    // Timing starts here
    // --------------------------
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    coarseningKernel<<<grid, block>>>(N, COARSENING_FACTOR, THREAD_PER_BLOCK*num_blocks, device_coord_mass, device_accel);
    CHECK_CUDA(cudaDeviceSynchronize());

    //Write Memory Back to Host
    CHECK_CUDA(cudaMemcpy(accel, device_accel, (size_t)N * 4  * sizeof(float), cudaMemcpyDeviceToHost));

    clock_gettime(CLOCK_MONOTONIC, &end);
    // --------------------------

    double elapsed =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Computation time: %.6f seconds\n", elapsed);

    //Write Results
    write_outputs(output_file, N, accel);

    //Cleanup
    free(accel);
    free(coord_mass);
    cudaFree(device_coord_mass);
    cudaFree(device_accel);

    return 0;
}