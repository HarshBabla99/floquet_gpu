# Set threading limits
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
####################################################################################################
import argparse
import ast

import numpy as np
import scqubits as scq
import qutip as qt

import floquet as ft

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Floquet simulation")
    parser.add_argument("--idx", default=-1, type=int, help="index that is unraveled")
    parser.add_argument("--hilbert_dim", default=25, type=int)
    
    parser.add_argument(
        "--qubit_params",
        default="{'omega_p': 1.0, 'zeta': 0.28, 'ncut': 41, 'ng': 0.0}"
    )

    parser.add_argument("--drive_freq_ratios_file", default=None)
    parser.add_argument("--xi_sq_file", default=None)

    parser.add_argument("--fit_range_fraction", default=1.0, type=float, help="should be between (0, 1.0]")
    parser.add_argument("--floquet_sampling_time_fraction", default=0.0, type=float)
    parser.add_argument("--fit_cutoff", default=4, type=int)
    parser.add_argument("--overlap_cutoff", default=0.7, type=float)
    parser.add_argument("--nsteps", default=30000, type=int)
    parser.add_argument("--num_cpus", default=4, type=int)
    parser.add_argument("--save_floquet_modes", default=1, type=int)
    parser.add_argument("--save_directory", default="out/", type=str)
    parser_args = vars(parser.parse_args())


    qubit_params = ast.literal_eval(parser_args['qubit_params'])


    # Omega_d vals
    with open(parser_args['drive_freq_ratios_file'], 'r') as file:
        drive_freq_ratios_vals = file.read()
    formatted_output = drive_freq_ratios_vals.replace('\n', ', ')
    omega_d_linspace = np.array(
        ast.literal_eval(f"[{formatted_output}]")
    ) * qubit_params['omega_p']
    if parser_args['idx'] != -1:
        filename = f'{str(parser_args['idx']).zfill(5)}'
    else:
        filename = 'tmon_floquet'
        
    filepath = ft.generate_file_path("h5py", filename, parser_args["save_directory"])

    # Hamiltonian setup
    qubit_params = ast.literal_eval(parser_args['qubit_params'])
    scq_qubit_params = {
        'EC' : qubit_params['omega_p'] * qubit_params['zeta']/8,
        'EJ' : qubit_params['omega_p'] / qubit_params['zeta'],
        'ncut': qubit_params['ncut'],
        'ng' : qubit_params['ng'],
        'truncated_dim' : parser_args['hilbert_dim'],
    }
    tmon = scq.Transmon(**scq_qubit_params)

    hilbert_space = scq.HilbertSpace([tmon])
    hilbert_space.generate_lookup()
    evals = hilbert_space["evals"][0][0:parser_args['hilbert_dim']]
    evals = evals - evals[0]

    H0 = qt.Qobj(np.diag(evals))    
    H1 = hilbert_space.op_in_dressed_eigenbasis(tmon.n_operator)
    
    state_indices = [0, 1]

    # Xi-Sq & Amp vals
    with open(parser_args['xi_sq_file'], 'r') as file:
        xi_sq_values = file.read()
    formatted_output = xi_sq_values.replace('\n', ', ')
    xi_sq_linspace = np.array(ast.literal_eval(f"[{formatted_output}]"))
    xi_sq_to_amp = ft.XiSqToAmp(H0, H1, state_indices, omega_d_linspace)
    amp_linspace = xi_sq_to_amp.amplitudes_for_omega_d(xi_sq_linspace)

    # To-save
    init_data_to_save = tmon.get_initdata() | {"xi_sq_linspace" : xi_sq_linspace.tolist(), 'omega_d_linspace' : omega_d_linspace.tolist()}

    # Floquet analysis
    model = ft.Model(H0, H1, omega_d_linspace, amp_linspace)
    options = ft.Options(
        fit_range_fraction=parser_args["fit_range_fraction"],
        floquet_sampling_time_fraction=parser_args["floquet_sampling_time_fraction"],
        fit_cutoff=parser_args["fit_cutoff"],
        overlap_cutoff=parser_args["overlap_cutoff"],
        nsteps=parser_args["nsteps"],
        num_cpus=parser_args["num_cpus"],
        save_floquet_modes=parser_args["save_floquet_modes"],

    )
    floquet_instance = ft.FloquetAnalysis(
        model=model,
        state_indices=state_indices,
        options=options,
        init_data_to_save=init_data_to_save,
    )
    floquet_instance.run(filepath=filepath)
    