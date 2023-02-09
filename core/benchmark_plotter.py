import matplotlib.pyplot as plt
from pathlib import Path

try:
    open(str(Path(__file__).parent.absolute())+f"/results/time/totaltime.png", "x").close()
except:
    pass

xlist = [2, 3, 4, 5]
ylist_model = [0.92, 1.18, 9.21, 636.49]

plt.plot(xlist, ylist_model, color='r', label='Our model')
plt.yscale("log")
plt.title(f"Total runtime")
plt.xlabel("number of qubits N")
plt.ylabel("runtime per iteration (s)")
plt.xticks(xlist)
leg = plt.legend(loc='lower right')
plt.savefig(str(Path(__file__).parent.absolute())+f"/results/time/totaltime.png")
plt.clf()