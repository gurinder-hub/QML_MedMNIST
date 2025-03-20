dataset_circuit_hyperparams = dict()
gatesets = dict()

dataset_names = ['pneumonia_2/pneumonia_emeds_49', 'pneumonia_2/pneumonia_emeds_64', 'breast_2/breast_emeds_49', 'breast_2/breast_emeds_64', 'oct_4/oct_emeds_49', 'oct_4/oct_emeds_64', 'retina_5/retina_emeds_49', 'retina_5/retina_emeds_64', 'derma_7/derma_emeds_49', 'derma_7/derma_emeds_64', 'blood_8/blood_emeds_49', 'blood_8/blood_emeds_64', 'path_9/path_emeds_49', 'path_9/path_emeds_64', 'organs_11/organs_emeds_49', 'organs_11/organs_emeds_64']

circuit_params = [60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60]
circuit_embeds = [49, 64, 49, 64, 49, 64, 49, 64, 49, 64, 49, 64, 49, 64, 49, 64]
circuit_qubits = [4, 4, 4, 4, 8, 8, 5, 5, 7, 7, 8, 8, 9, 9, 11, 11]
num_data_reps = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
num_meas_qubits = [1, 1, 1, 1, 2, 2, 5, 5, 7, 7, 8, 8, 9, 9, 11, 11]
num_embed_layers_angle_iqp = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
num_embed_layers_amp = [1 for i in range(16)]
num_var_layers = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
num_classes = [2, 2, 2, 2, 4, 4, 5, 5, 7, 7, 8, 8, 9, 9, 11, 11]

for i in range(len(dataset_names)):
    dataset_circuit_hyperparams[dataset_names[i]] = dict()
    dataset_circuit_hyperparams[dataset_names[i]]['num_params'] = circuit_params[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_embeds'] = circuit_embeds[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_qubits'] = circuit_qubits[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_data_reps'] = num_data_reps[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_meas_qubits'] = num_meas_qubits[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_var_layers'] = num_var_layers[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_embed_layers'] = dict()
    dataset_circuit_hyperparams[dataset_names[i]]['num_embed_layers']['angle'] = num_embed_layers_angle_iqp[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_embed_layers']['iqp'] = num_embed_layers_angle_iqp[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_embed_layers']['amp'] = num_embed_layers_amp[i]
    dataset_circuit_hyperparams[dataset_names[i]]['num_classes'] = num_classes[i]

gateset_names = ['rxyz_cz', 'rzx_rxx', 'ibm_basis', 'rigetti_aspen_m2_basis', 'oqc_lucy_basis']
gateset_gates = [[['rx', 'ry', 'rz'], ['cz']], [[], ['rzx', 'rxx']], [['rz', 'sx', 'x'], ['cx']], 
                [[], []], [[], []]]

gateset_param_nums = [[[1, 1, 1], [0]], [[], [1, 1]], [[1, 0, 0], [0]], 
                [[], []], [[], []]]

for i in range(len(gateset_names)):
    gatesets[gateset_names[i]] = (gateset_gates[i], gateset_param_nums[i])
