#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct {
    float x, y, z;
} Vec3;

#define EPS2 1e-9f   // softening factor

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
// Function 2: Serial O(N^2) acceleration computation
// --------------------------------------------------------
void compute_accelerations(int N, const Vec3 *pos, const float *mass, Vec3 *acc) {
    const float G = 1.0f;

    for (int i = 0; i < N; i++) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f;

        float xi = pos[i].x;
        float yi = pos[i].y;
        float zi = pos[i].z;

        for (int j = 0; j < N; j++) {
            if (i == j) continue;

            float rx = pos[j].x - xi;
            float ry = pos[j].y - yi;
            float rz = pos[j].z - zi;

            float distSqr = rx*rx + ry*ry + rz*rz + EPS2;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            float s = G * mass[j] * invDist3;

            ax += rx * s;
            ay += ry * s;
            az += rz * s;
        }

        acc[i].x = ax;
        acc[i].y = ay;
        acc[i].z = az;
    }
}

// --------------------------------------------------------
// Function 3: Write output to file
// --------------------------------------------------------
void write_outputs(const char *output_file, int N, const Vec3 *acc) {
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

    Vec3 *acc = (Vec3 *)malloc(sizeof(Vec3) * N);

    // --------------------------
    // Timing starts here
    // --------------------------
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    compute_accelerations(N, pos, mass, acc);

    clock_gettime(CLOCK_MONOTONIC, &end);
    // --------------------------

    double elapsed =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Computation time: %.6f seconds\n", elapsed);

    // Write results
    write_outputs(output_file, N, acc);

    free(mass);
    free(pos);
    free(acc);

    return 0;
}
