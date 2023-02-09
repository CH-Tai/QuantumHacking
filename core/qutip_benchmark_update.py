import numpy as np
import matplotlib.pyplot as plt
import datetime

from qutip import Qobj, identity, sigmax, sigmaz
from qutip.qip import hadamard_transform
import qutip.logging_utils as logging
logger = logging.get_logger()
#Set this to None or logging.WARN for 'quiet' execution
log_level = logging.INFO
#QuTiP control modules
import qutip.control.pulseoptim as cpo

import torch

example_name = 'Hadamard'
# Drift Hamiltonian
H_d = sigmaz()

def construct_hamiltonian(n):

    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    Y = torch.tensor([[0, complex(0, -1)], [complex(0, 1), 0]], dtype=torch.complex64)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
    I = torch.eye(2, dtype=torch.complex64)

    # For defining n-qubit drift & control Hamiltonian
    def hat(gate, index):
        hatgate = torch.tensor([1], dtype=torch.complex64)
        for i in range(n):
            if i != index:
                unitary = I
            if i == index:
                unitary = gate
            hatgate = torch.kron(hatgate, unitary)
        return hatgate

    Xs = torch.stack(list(hat(X, i) for i in range(n)))
    Ys = torch.stack(list(hat(Y, i) for i in range(n)))
    Zs = torch.stack(list(hat(Z, i) for i in range(n)))

    F = torch.zeros(n, n, dtype=torch.complex64)/2
    for i in range(n):
        for j in range(n):
            if j == i + 1:
                F[i][j] = 1

    Hd = torch.einsum("ij, ikl, jlm -> km", F, Zs, Zs)
    Hc = torch.stack([torch.einsum("ijk -> jk", Xs), torch.einsum("ijk -> jk", Ys)])
    return Hc, Hd


def benchmark(n, N):
        H_d = Qobj(construct_hamiltonian(n)[1].cpu().detach().numpy())
        # The (single) control Hamiltonian
        H_c = [Qobj(construct_hamiltonian(n)[0][0].cpu().detach().numpy()), Qobj(construct_hamiltonian(n)[0][1].cpu().detach().numpy())]
        # start point for the gate evolution
        U_0 = identity(2**n)
        # Target for the gate evolution Hadamard gate
        U_targ = hadamard_transform(n)
        # Number of time slots
        n_ts = N
        # Time allowed for the evolution
        evo_time = 0.05
        # Fidelity error target
        fid_err_targ = 1e-2
        # Maximum iterations for the optimisation algorithm
        max_iter = 10000
        # Maximum (elapsed) time allowed in seconds
        max_wall_time = 120
        # Minimum gradient (sum of gradients squared)
        # as this tends to 0 -> local minima has been found
        min_grad = 10*1/10000000000000*N
        # pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
        p_type = 'RND'
        #Set to None to suppress output files
        f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)
        result = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, n_ts, evo_time, 
                        fid_err_targ=fid_err_targ, min_grad=min_grad, 
                        max_iter=max_iter, max_wall_time=max_wall_time, 
                        out_file_ext=f_ext, init_pulse_type=p_type, 
                        log_level=log_level, gen_stats=True)
        result.stats.report()
        print("Final evolution\n{}\n".format(result.evo_full_final))
        print("********* Summary *****************")
        print("Final fidelity error {}".format(result.fid_err))
        print("Final gradient normal {}".format(result.grad_norm_final))
        print("Terminated due to {}".format(result.termination_reason))
        print("Number of iterations {}".format(result.num_iter))
        print("Completed in {} HH:MM:SS.US".format(
                datetime.timedelta(seconds=result.wall_time)))
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(2, 1, 1)
        ax1.set_title("Initial control amps")
        #ax1.set_xlabel("Time")
        ax1.set_ylabel("Control amplitude")
        ax1.step(result.time,
                np.hstack((result.initial_amps[:, 0], result.initial_amps[-1, 0])),
                where='post')

        ax2 = fig1.add_subplot(2, 1, 2)
        ax2.set_title("Optimised Control Sequences")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Control amplitude")
        ax2.step(result.time,
                np.hstack((result.final_amps[:, 0], result.final_amps[-1, 0])),
                where='post')
        plt.tight_layout()
        plt.show()

benchmark(2,1)
