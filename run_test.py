import subprocess
import sys

CORRECTNESS_SH = "correctness_test.sh"
PERFORMANCE_SH = "performance_test.sh"

strategy_name = {
    0: "serial",
    1: "shared_memory",
    2: "thread_coarsening",
    3: "optimal",
}

if len(sys.argv)==1 or sys.argv[1]=="1":
    print("[ Start correctness test ]")

    # strategy
    for i in range(4): 
        print("  Testing " + strategy_name[i])
        # test case
        for j in range(7): # TO DO: Change the range when test cases are done.
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
    print("[ Start performance test ]")

print("[Done]")
