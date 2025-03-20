import torch
import numpy as np
import pickle as pkl
import pennylane as qml
import os
from torch import Tensor
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Batch, Session, SamplerV2 as Sampler
from qiskit.compiler import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import Layout
from modified_elivagar.circuits.arbitrary import get_circ_params
from modified_elivagar.circuits.create_circuit import create_qiskit_circ, create_gate_circ
from modified_elivagar.inference.quantumnat import quantize_and_normalize
from modified_elivagar.inference.inference_metrics import mse_batch_loss, mse_vec_batch_loss, batch_acc, vec_batch_acc, auc_binary, auc_oct, auc_multi_class, TQCeLoss
from modified_elivagar.utils.create_noise_models import noisy_dev_from_backend, get_noise_model
from modified_elivagar.utils.datasets import load_dataset
import mthree

def get_params_from_tq_model(model_dir, num_params):
    """
    Get the parameters stoerd in a torch quantum model.
    """
    model_data = torch.load(os.path.join(model_dir, 'model.pt'))
    model_params = np.zeros(num_params)
    
    for i in range(num_params):
        model_params[i] = model_data[f'var_gates.{i}.params'].cpu().numpy().item()
    
    return model_params

def run_qiskit_circ(circuit, dev, num_meas_qubits, num_shots=1024, transpile_circ=False, basis_gates=None,
                    coupling_map=None, mode='exp', opt_level=0, qubit_mapping=None):
    if qubit_mapping is None:
        qubit_mapping = [i for i in range(circuit.num_qubits)]
    
    if transpile_circ:
        circuit = transpile(circuit, basis_gates=basis_gates, coupling_map=coupling_map,
                           initial_layout=list(qubit_mapping), optimization_level=opt_level)

    new_circuit = transpile(circuit, basis_gates=basis_gates, coupling_map=coupling_map, initial_layout=list(qubit_mapping), optimization_level=opt_level)
    sampler = Sampler(mode=dev)
    job = sampler.run([new_circuit], shots = num_shots)
    outputs = job.result()[0].data.c.get_counts()
    
    if mode == 'exp':
        qubit_probs = np.zeros((num_meas_qubits, 2))

        for key in outputs.keys():
            for q in range(num_meas_qubits):
                qubit_probs[q, int(key[q])] += outputs[key]

        qubit_probs = qubit_probs[::-1, :] / num_shots
        ret_val = qubit_probs[:, 0] - qubit_probs[:, 1]
    elif mode == 'probs':
        ret_val = np.zeros(2 ** num_meas_qubits)
        
        for key in outputs.keys():
            key_bin = int(key[::-1], 2)
            
            ret_val[key_bin] = outputs[key]
            
        ret_val /= num_shots
    
    return ret_val
    
def run_qiskit_circ_inference(circuit, dev, num_meas_qubits, num_shots=1024, transpile_circ=False, basis_gates=None,
                    coupling_map=None, mode='exp', opt_level=0, qubit_mapping=None, mit=None):
    if qubit_mapping is None:
        qubit_mapping = [i for i in range(circuit.num_qubits)]
    
    if transpile_circ:
        circuit = transpile(circuit, basis_gates=basis_gates, coupling_map=coupling_map,
                           initial_layout=list(qubit_mapping), optimization_level=opt_level)

    new_circuit = transpile(circuit, basis_gates=basis_gates, coupling_map=coupling_map, initial_layout=list(qubit_mapping), optimization_level=opt_level)
    sampler = Sampler(mode=dev)
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.num_randomizations = "auto"
    job = sampler.run([new_circuit], shots = num_shots)
    outputs_inter = job.result()[0].data.c.get_counts()
    #m3 mitigation
    #for multi-class except 4-class, use the following lines to obtain the qubit_list
    # qubit_list = list(qubit_mapping)
    # qubit_list = [int(q) for q in qubit_list]
    
    #for binary and 4-class, use the following line to obtain the qubit_list
    qubit_list = mthree.utils.final_measurement_mapping(new_circuit)
    quasis = mit.apply_correction(outputs_inter, qubit_list)
    real_quasis = quasis.nearest_probability_distribution()
    outputs = {key: round(value * num_shots) for key, value in real_quasis.items()}
    
    if mode == 'exp':
        qubit_probs = np.zeros((num_meas_qubits, 2))

        for key in outputs.keys():
            for q in range(num_meas_qubits):
                qubit_probs[q, int(key[q])] += outputs[key]

        qubit_probs = qubit_probs[::-1, :] / num_shots
        ret_val = qubit_probs[:, 0] - qubit_probs[:, 1]
    elif mode == 'probs':
        ret_val = np.zeros(2 ** num_meas_qubits)
        
        for key in outputs.keys():
            key_bin = int(key[::-1], 2)
            
            ret_val[key_bin] = outputs[key]
            
        ret_val /= num_shots
    
    return ret_val


def tq_model_inference_on_noisy_sim_qiskit(circ_dir, device_name, num_runs, num_qubits, meas_qubits, noisy_dev, basis_gates,
                                           coupling_map, qubit_mapping, x_test, y_test, params=None, save=True,
                                           num_shots=1024, compute_noiseless=False, transpile_opt_level=0,
                                           file_suffix='', index_map=None, results_save_dir=None, pick_best=False,
                                           use_quantumnat=False, quantumnat_trained_dir=None):
    """
    Peform inference on test data x_test using a noisy simulator noisy_dev with a trained circuit stored in circ_dir.
    """
    num_meas_qubits = len(meas_qubits)
    circ_gates, gate_params, inputs_bounds, weights_bounds = get_circ_params(circ_dir)
    circ_creator = create_qiskit_circ(circ_gates, gate_params, inputs_bounds,
                                      weights_bounds, meas_qubits, num_qubits)
    
    if not os.path.exists(results_save_dir):
        os.mkdir(results_save_dir)
    
    device_results_save_dir = os.path.join(results_save_dir, device_name)

    if not os.path.exists(device_results_save_dir):
        os.mkdir(device_results_save_dir) 
    
    losses_list = []
    accs_list = []
    aucs_list = []
    
    pennylane_dev = qml.device('lightning.qubit', wires=num_qubits)
    curr_pennylane_circ = create_gate_circ(pennylane_dev, circ_gates, gate_params, inputs_bounds,
                                           weights_bounds, meas_qubits)
    
    if use_quantumnat:
        noiseless_accs = np.genfromtxt(os.path.join(circ_dir, quantumnat_trained_dir, 'accs.txt'))
    else:
        noiseless_accs = np.genfromtxt(os.path.join(circ_dir, 'accs.txt'))
        
    if len(noiseless_accs.shape) == 0:
        noiseless_accs = np.array([noiseless_accs])

    if index_map is not None and qubit_mapping is not None:
        qubit_mapping = [index_map[q] for q in qubit_mapping]

    if pick_best:
        ordering = np.argsort(-1 * noiseless_accs)
        chosen_run_indices = [(ordering[i], noiseless_accs[ordering[i]]) for i in range(num_runs)]
    else:
        chosen_run_indices = [(i, noiseless_accs[i]) for i in range(num_runs)]
    
    for run_index, run_noiseless_acc in chosen_run_indices:
        if use_quantumnat:
            curr_run_dir = os.path.join(circ_dir, quantumnat_trained_dir, f'run_{run_index + 1}')
        else:
            curr_run_dir = os.path.join(circ_dir, f'run_{run_index + 1}')
        
        if params is None:
            print('Fetching param values from', curr_run_dir)
            curr_params = get_params_from_tq_model(curr_run_dir, weights_bounds[-1])
        else:
            curr_params = params[run_index]

        val_exps = []
            
        circ_list = [circ_creator(sample, curr_params) for sample in x_test]
        qubit_list = list(qubit_mapping)
        qubit_list = [int(q) for q in qubit_list]
        with Session(backend=noisy_dev) as session:
            mit = mthree.M3Mitigation(session._backend)
            mit.cals_from_system(qubit_list, num_shots, runtime_mode=session)
            for i in range(len(x_test)):            
                val_exps.append(
                    run_qiskit_circ_inference(
                        circ_list[i], session, num_meas_qubits, num_shots,
                        False, basis_gates, coupling_map, 'exp', transpile_opt_level,
                        qubit_mapping, mit
                    )
                )
                
                if compute_noiseless:
                    pennylane_outputs = curr_pennylane_circ(x_test[i], curr_params)
    
                    print(f'Noiseless: {pennylane_outputs} | Noisy: {val_exps[-1]}')
            
        val_exps = np.array(val_exps)
        #save the results for future check if so desired
        with open(os.path.join(device_results_save_dir, 'val_exps_sim.pkl'), 'wb') as f:
            pkl.dump(val_exps, f)

        with open(os.path.join(device_results_save_dir, 'x_test_sim.pkl'), 'wb') as f:
            pkl.dump(x_test, f)

        with open(os.path.join(device_results_save_dir, 'y_test_sim.pkl'), 'wb') as f:
            pkl.dump(y_test, f)

        #load the trained model
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # model = torch.load(os.path.join(curr_run_dir, 'model.pt'))
        # batch_norm = torch.nn.BatchNorm1d(num_qubits)
        # batch_norm.eps = 1e-5
        # batch_norm.to(device)
        # batch_norm.weight = torch.nn.Parameter(model['normalizer.weight']).to(device)
        # batch_norm.bias = torch.nn.Parameter(model['normalizer.bias']).to(device)
        # batch_norm.running_mean = model['normalizer.running_mean'].to(device)
        # batch_norm.running_var = model['normalizer.running_var'].to(device)
        # batch_norm.eval()
        # val_exps = batch_norm(torch.tensor(val_exps, dtype=torch.float32, device=device))
        # val_exps = val_exps.cpu()

        if use_quantumnat:
            val_exps = quantize_and_normalize(val_exps, curr_run_dir)

        if len(meas_qubits) > 2:
            val_exps = val_exps.reshape((len(x_test), len(meas_qubits)))
            acc = torch.sum(torch.eq(torch.argmax(val_exps, dim=1, keepdim=True), torch.tensor(y_test).reshape(-1,1)))/len(y_test)
            print(acc)
            loss = TQCeLoss(len(meas_qubits), num_qubits)
            val_loss = loss(Tensor(val_exps),Tensor(y_test))
            print(val_loss)
            auc = auc_multi_class(val_exps, y_test, num_meas_qubits)
            print(auc)
        elif len(meas_qubits) == 2:
            val_exps = val_exps.reshape((len(x_test), len(meas_qubits)))
            val_loss = mse_vec_batch_loss(val_exps, y_test)
            acc = vec_batch_acc(val_exps, y_test)
            auc = auc_oct(val_exps, y_test, num_meas_qubits)
        else:
            val_exps = val_exps.reshape(len(x_test))
            val_loss = mse_batch_loss(val_exps, y_test)
            acc = batch_acc(val_exps, y_test)
            auc = auc_binary(val_exps, y_test)

        losses_list.append(val_loss)
        accs_list.append(acc)
        aucs_list.append(auc)
        
        print(f'Loss: {val_loss} | Acc: {acc} | Noiseless Acc: {run_noiseless_acc}')

    if save:
        losses_list = [loss.detach().numpy() if isinstance(loss, torch.Tensor) else loss for loss in losses_list]
        accs_list = [acc.detach().numpy() if isinstance(acc, torch.Tensor) else acc for acc in accs_list]
        aucs_list = [auc.detach().numpy() if isinstance(auc, torch.Tensor) else auc for auc in aucs_list]
        np.savetxt(os.path.join(device_results_save_dir, f'val_losses_inference_only{file_suffix}.txt'), losses_list)
        np.savetxt(os.path.join(device_results_save_dir, f'accs_inference_only{file_suffix}.txt'), accs_list)
        np.savetxt(os.path.join(device_results_save_dir, f'aucs_inference_only{file_suffix}.txt'), aucs_list)
        
    return losses_list, accs_list, aucs_list


def run_noisy_inference_for_tq_circuits_qiskit(circ_dir, circ_prefix, num_circs, num_runs, num_qubits, meas_qubits, device_name, noise_model, basis_gates,
                                               coupling_map, dataset, embed_type, num_data_reps, num_test_samples=None, human_design=False, compute_noiseless=False,
                                               use_qubit_mapping=False, num_shots=1024, qubit_mapping_file_name=None, transpile_opt_level=0, file_suffix='',
                                               index_map=None, results_save_dir=None, pick_best=False, use_quantumnat=False, quantumnat_trained_dir=None):
    """
    Run noisy inference for TQ circuits in the same folder - used to perform infrence for all ex. random, human designed, etc. circuits with one call.
    """
    x_train, y_train, x_test, y_test = load_dataset(dataset, embed_type, num_data_reps)
    if noise_model is None and 'ibm' in device_name:
        noise_model, basis_gates, coupling_map = get_noise_model(device_name)
        
    #noisy_dev = AerSimulator(noise_model=noise_model, device = "GPU")
    #noisy_dev = AerSimulator(device = "GPU")
    service = QiskitRuntimeService(channel='ibm_quantum',
                   instance='',
                   token=''
                )
    
    noisy_dev = service.backend(device_name)
    
    all_accs = []
    
    for i in range(num_circs):
        if num_test_samples:
            sel_inds = np.random.choice(len(x_test), num_test_samples, False)

            x_test = x_test[sel_inds]
            y_test = y_test[sel_inds]
        
        if human_design:
            curr_circ_dir = circ_dir
        else:
            curr_circ_dir = os.path.join(circ_dir, f'{circ_prefix}_{i + 1}')
            
        print('Saving results in', curr_circ_dir, results_save_dir)
        curr_results_save_dir = os.path.join(curr_circ_dir, results_save_dir)
            
        if use_qubit_mapping:
            qubit_mapping = np.genfromtxt(os.path.join(curr_circ_dir, qubit_mapping_file_name))
        else:
            qubit_mapping = None
        
        losses_list, accs_list, aucs_list = tq_model_inference_on_noisy_sim_qiskit(
            curr_circ_dir, device_name, num_runs, num_qubits, meas_qubits,
            noisy_dev, basis_gates, coupling_map, qubit_mapping,
            x_test, y_test, None, True, num_shots, compute_noiseless,
            transpile_opt_level, file_suffix, index_map, curr_results_save_dir,
            pick_best, use_quantumnat, quantumnat_trained_dir
        )
        
        all_accs.append(accs_list[0])
        
        print(i)
        
    print(np.mean(np.sort(all_accs)[5:]))