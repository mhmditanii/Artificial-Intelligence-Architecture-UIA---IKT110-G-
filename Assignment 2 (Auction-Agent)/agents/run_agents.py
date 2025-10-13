import os
import subprocess

agents = [
    "agent_tiny_bid.py",
    "mhmdmain.py",
    "rand_single.py",
    "rand_walk.py",
    "print.py",
    "agent_tiny_bid.py",
    "rand_walk.py",
    "test1.py",
    "test2.py",
    "./gametest/ignacio.py",
    "./gametest/fortuna_agent.py",
    "./gametest/helper.py",
    "./gametest/raphael.py",
    "./gametest/victor.py",
    "./gametest/magnus.py",
    "./gametest/lebron.py",
    "./gametest/",
    "./gametest/victor2.py",
    "./gametest/corni.py",
    "./gametest/maxi.py",
    "./mfgrim.py",
]

current_dir = os.path.dirname(os.path.abspath(__file__))

processes = []
for filename in agents:
    filepath = os.path.join(current_dir, filename)
    print(f"Starting {filename}...")
    p = subprocess.Popen(["python3", filepath])
    processes.append(p)

# Optionally wait for all to finish:
for p in processes:
    p.wait()
