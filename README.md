# N-Body Simulation: Performance and Parallelization Study

This project implements and evaluates multiple CPU and GPU versions of an N-body simulation. We explore various parallelization and optimization strategies to analyze their correctness, performance, and trade-offs on modern GPU architectures.

**Group Members:**
- Po-Tsun Yu
- Alexander Kachan
- Allen Liao
- Nicholas Drew
- Will Han

---

## Implemented Strategies

| Strategy | Description | Main Files / Related Scripts |
| --- | --- | --- |
| 0. Serial | CPU baseline used for correctness checks | `cu_files/serial.cu` |
| 1. Shared Memory | GPU kernel using a tiled shared-memory approach inspired by GPU Gems | `cu_files/shared_memory.cu` |
| 2. Thread Coarsening | Each thread processes multiple bodies to reduce global memory traffic | `cu_files/thread_coarsening.cu` |
| 3. Hyperparameter Tuning | Searches block size and coarsening factor combinations | `cu_files/optimal.cu`, `tuning_build.sh` |
| 4. Hierarchical N-Body | Cluster-based blocking evaluated on spatially localized datasets (no force approximation) | `hierarchical_dataset.py`, `cu_files/hierarchical.cu` |
| 5. CUTLASS-based version | NVIDIA CUTLASS-based version using dot production to calculate L2 distances | `cutlass/`, `cu_files/cutlass.cu` |

## Environment & Setup

### Dependencies
- **CUDA Toolkit:** Version 12.0 or newer
- **Python:** Version 3.8 or newer
- **CUTLASS:** Latest version
- **NumPy:** Required ONLY if generating new test data    

About CUTLASS, go to the project root directory (where .sh and .py files are located) and just clone the git of CUTLASS:
```bash
git clone https://github.com/NVIDIA/cutlass.git
```
You can install Python dependencies using pip:
```bash
pip install numpy
```

### Environment
All experiments were conducted on the University of Minnesota CUDA server with the following specifications:
- **GPU:** NVIDIA Tesla T4 (15,360 MiB)
- **CUDA Toolkit:** 13.0
- **Driver Version:** 580.105.08
- **CPU:** Intel Xeon Gold 6148 @ 2.40 GHz (40 physical cores / 80 logical cores)
- **Platform:** Single-GPU compute node

---

## Build and Run

TL;DR: 
```bash
./build.sh
python3 run_test.py
```

More thorough instructions:

### 1. Build the Executables

A general-purpose build script compiles all main strategies (0, 1, 2, 4) into the `build/` directory.

```bash
./build.sh
```

For hyperparameter tuning (Strategy 3), use the `tuning_build.sh` script, which compiles specialized kernels into the `build_test/` directory.
```bash
./tuning_build.sh
```

### 2. Run Tests

We recommend running the full testing suite ([full suite](#run-automated-test-suites))

#### Run a Single Test Case
Use the `test_one.sh` script to run a specific strategy on a single test case.

The `strategy_id`s supported by this script are:
- **0:** Serial
- **1:** Shared Memory
- **2:** Thread Coarsening
- **3:** Hyperparameter Tuning
- **4:** Hierarchical N-Body
- **5:** CUTLASS-based version

**Syntax:**
```bash
./test_one.sh [strategy_id] [test_case_num] [output_file_name (optional)]
```

**Examples:**
```bash
./test_one.sh 1 0 # Run Strategy 1 (Shared Memory) on test case 0 and save to a default results file

./test_one.sh 2 5 my_output.txt # Run Strategy 2 (Thread Coarsening) on test case 5 and save to my_output.txt
```
> Default output is saved to `results/[strategy_name]_test[test_num].txt`.

#### Run Automated Test Suites
The `run_test.py` script provides an easy way to run comprehensive correctness and performance tests.

- **Run Correctness & Performance Tests:**
  ```bash
  python3 run_test.py
  ```

- **Run Only Correctness Tests:**
  Compares GPU outputs against the serial baseline for all strategies.
  ```bash
  python3 run_test.py 1
  ```

- **Run Only Performance Tests:**
  Measures and reports the speedup of parallel strategies.
  ```bash
  python3 run_test.py 2
  ```

### 3. Creating test case for Hierarchical Simulation (Optional)

As current impplemented hierarchical method requires customized dataset, we used this python script to generate one.
1.  **Generate the dataset**
    ```bash
    python3 hierarchical_dataset.py
    ```
    This creates `mass.txt`, `coord.txt`, and `cluster_id.txt` in project root directory.

2.  **Run the simulation:**
    You can use `test_one.sh` with strategy ID 4 to test the generated test case. Copy and paste the mass/coordinate information to tests for directory following name format of other files in there (testinxx_mass.txt and testinxx_coordinate.txt).
    ```bash
    ./test_one.sh 4 123
    ```
