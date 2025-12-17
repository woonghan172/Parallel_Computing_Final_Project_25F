import subprocess
import sys
import re

CORRECTNESS_SH = "correctness_test.sh"
PERFORMANCE_SH = "performance_test.sh"

LABEL_WIDTH = 25
NUM_WIDTH   = 10

strategy_name = {
    0: "serial",
    1: "shared_memory",
    2: "thread_coarsening",
    3: "optimal",
}

bodies_for_speedup = {
    0: "100,000",
    1: "500,000",
    2: "1,000,000",
}

if len(sys.argv)==1 or sys.argv[1]=="1":
    print("[ Start correctness test ]")

    # strategy
    for i in range(4): 
        print("  Testing " + strategy_name[i])
        # test case
        for j in range(6): # TO DO: Change the range when test cases are done.
            res = subprocess.run(
                ["bash", CORRECTNESS_SH, str(i), str(j)],
                capture_output=True,
                text=True
            )
            if res.returncode == 0:
                print("    test case {}: PASS".format(j))
            elif res.returncode == 1:
                print("    FAIL on test case {} (Numeric mismatch)".format(j))
            elif res.returncode == 2:
                print("    FAIL on test case {} (File/format error)".format(j))

if len(sys.argv)==1 or sys.argv[1]=="2":
    print("\n[ Start performance test ]")

    # test case
    for j in range(3): # TO DO: Change the range when test cases are done.
        print(f"Speed Up Test Case {j} ("+ bodies_for_speedup[j] +" bodies)")
        # strategy
        for i in range(1, 4): 
            res = subprocess.run(
                ["bash", PERFORMANCE_SH, str(i), str(j)],
                capture_output=True,
                text=True
            )
            #print(res.stdout)
            match = re.search(r"Computation time:\s*([0-9.]+)\s*seconds", res.stdout)
            if match:
                comp_time = float(match.group(1))
                print(f"    {strategy_name[i]:<{LABEL_WIDTH}}{comp_time:{NUM_WIDTH}.5f} seconds")
            else:
                print("    " + strategy_name[i] + ": Computation time not found in output")

print("[Done]")
