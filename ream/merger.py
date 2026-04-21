# Copyright (c) 2026. Samsung Electronics Co., Ltd.

# All rights reserved.

#

# This source code is licensed under the license found in the

# LICENSE file in the root directory of this source tree.

"""

Merger.

"""

import torch
import numpy as np
import os
import os.path
import torch.nn as nn
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from .hc import hcsmoe
from .ream import pseudo_group
from .utils import mem, normalize_rows, num_parameters
from .moe_utils import run_all_experts, moe_forward, get_moe_input, get_num_experts
from .weight_utils import ffn_weight_matrix, pca_reduce, apply_perm_to_ffn, experts_weight_matrix
from .saliency import reap, freq
from data.calibration_data import print_seq_stats

class Merger:
    def __init__(self,
                 model: nn.Module,
                 mtp_state_dict: dict = None,
                 merge_size: int = 96,
                 grouping: str = 'ream',
                 merging: str = 'logits+weights',
                 saliency: str = 'reap',
                 dataset: str = 'c4+math+code',
                 mix_ratio: str = '0.0,0.3,0.7',
                 tokenizer_name: str = 'qwen3',
                 batch_size: int = 8,
                 group_size: int = 16,
                 sequential: bool = True,
                 use_gate_output: bool = True,
                 gated_sim: bool = True,
                 calibration_data_size: int = 3072,
                 calibration_data_seq_len: int = 512,
                 seed: int = 42,
                 verbose: bool = True,
                 checkpoint_dir: str = None,
                 ):
        self.model = model
        self.mtp_state_dict = mtp_state_dict
        self.merge_size = merge_size
        self.first_moe_layer = getattr(model.config, 'first_k_dense_replace', 0)
        first_moe = model.model.layers[self.first_moe_layer].mlp
        if not hasattr(first_moe, 'top_k'):
            for idx in range(self.first_moe_layer, len(model.model.layers)):
                moe = model.model.layers[idx].mlp
                moe.top_k = moe.gate.top_k
                moe.num_experts = get_num_experts(moe)

        self.top_k = first_moe.top_k
        assert self.top_k == model.config.num_experts_per_tok, (self.top_k, model.config.num_experts_per_tok)
        self.n_experts = get_num_experts(first_moe)
        self.grouping = grouping
        self.merging = merging
        self.saliency = saliency
        assert self.saliency in ['freq', 'reap'], self.saliency
        self.sequential = sequential
        self.pca_dim = 64
        self.dataset = dataset
        self.mix_ratio = [float(x) for x in mix_ratio.split(',')]
        assert len(self.mix_ratio) >= len(dataset.split('+')), (self.mix_ratio, dataset)
        self.group_size = group_size
        self.batch_size = batch_size
        self.use_gate_output = use_gate_output
        self.gated_sim = gated_sim
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        def get_batch_file(dset_):
            return (f'data/{dset_}_b{calibration_data_size}_'
                    f'seq{calibration_data_seq_len}_{tokenizer_name}_seed{seed}.pt')

        if '+' in dataset:
            dsets = dataset.split('+')
            batch = {'input_ids': [], 'attention_mask': []}
            for dset, ratio in zip(dsets, self.mix_ratio):
                batch_file = get_batch_file(dset)
                if os.path.exists(batch_file):
                    print(f'loading batch from {batch_file}', flush=True)
                    batch_dset = torch.load(batch_file)
                    print_seq_stats(batch_dset)
                    n_samples = int(calibration_data_size * ratio)
                    if n_samples == 0:
                        print(f'skipping dataset {dset} with ratio {ratio}')
                        continue
                    np.random.seed(seed)
                    ind = np.random.permutation(batch_dset['input_ids'].shape[0])[:n_samples]
                    batch_dset['input_ids'] = batch_dset['input_ids'][ind]
                    batch_dset['attention_mask'] = batch_dset['attention_mask'][ind]
                    assert batch_dset['input_ids'].shape[1] >= calibration_data_seq_len, (
                        f'seq len={batch_dset["input_ids"].shape[1]},'
                        f'make sure the dataset is properly prepared using calibration_data.py')
                    batch['input_ids'].append(batch_dset['input_ids'])
                    batch['attention_mask'].append(batch_dset['attention_mask'])
                    print_seq_stats(batch_dset)
                else:
                    raise FileNotFoundError(f'File {batch_file} not found')
            batch['input_ids'] = torch.cat(batch['input_ids'])
            batch['attention_mask'] = torch.cat(batch['attention_mask'])
            print(f'dataset {dataset} loaded', flush=True)
        else:
            batch_file = get_batch_file(dataset)
            if os.path.exists(batch_file):
                print(f'loading batch from {batch_file}', flush=True)
                batch = torch.load(batch_file)
                print_seq_stats(batch)
            else:
                raise ValueError(f'file {batch_file} not found, '
                                 f'please run calibration_data.py in the data folder first')

        print('final batch',
              'input_ids', batch['input_ids'].shape,
              'attention_mask', batch['attention_mask'].shape,
              'attention mask sum', batch['attention_mask'].sum().item(), flush=True)
        self.batch = {k: v.to(self.device) for k, v in batch.items()}

        self.num_layers = len(self.model.model.layers)
        if self.mtp_state_dict is None:
            self.mtp_layer = None
        else:
            if 'qwen3_5' in self.model.__class__.__name__.lower():
                from .qwen3_mtp import build_mtp_layer_qwen3_5 as build_mtp_layer
            else:
                from .qwen3_mtp import build_mtp_layer
            self.num_layers += 1
            num_experts_orig = len(self.mtp_state_dict['mtp.layers.0.mlp.gate.weight'])
            self.mtp_layer = build_mtp_layer(self.mtp_state_dict, self.model, num_experts=num_experts_orig)
            self.mtp_layer.eval()
        for layer_ind in range(self.num_layers):
            moe_layer = (self.model.model.layers[layer_ind] if layer_ind < len(self.model.model.layers)
                         else self.mtp_layer.layer).mlp
            moe_layer._layer_ind = layer_ind

    @torch.no_grad()
    def _forward_pass(self, states, layer_ind, collect_outputs=True, upd_hid=False, inputs_embeds=None, verbose=False):
        expert_outs = {'gate': [], 'hid_act': [], 'final': 0, 'saliency': 0}
        handles = []
        hidden_states = []
        is_mtp = layer_ind >= len(self.model.model.layers)

        if collect_outputs:
            n_batches = max(1, self.batch['input_ids'].shape[0] // self.batch_size)

            def hook_fn(module, inputs, output):
                x = inputs[0]
                gate, final, act = run_all_experts(module,
                                                   x,
                                                   final_reduce=self.saliency != 'reap',
                                                   act_samples=max(1, 1024 * 32 // n_batches),
                                                   gated_sim=self.gated_sim)
                assert gate.dim() == 3, gate.shape
                assert final.dim() == 3, final.shape
                assert act.dim() == 3, act.shape

                if self.use_gate_output:
                    expert_outs['gate'].append(gate.data.cpu())
                expert_outs['hid_act'].append(act.data.cpu())

                if self.saliency == 'reap':
                    expert_outs['final'] += (final.data.mean(1, keepdim=True) / n_batches).cpu()
                    expert_outs['saliency'] += (reap(gate, final, top_k=self.top_k) / n_batches)
                else:
                    assert final.shape[1] == 1, final.shape
                    expert_outs['final'] += (final.data / n_batches).cpu()
                    expert_outs['saliency'] += (freq(gate, top_k=self.top_k) / n_batches)

            mlp = (self.mtp_layer.layer if is_mtp else self.model.model.layers[layer_ind]).mlp
            if verbose:
                print('adding hook to layer', layer_ind, mlp._layer_ind)
            handles.append(mlp.register_forward_hook(hook_fn))

        if verbose:
            print('moe forward input', 'mtp', is_mtp,
                  states['hidden_states'].shape, states['attention_mask'].shape)
        for i_ in tqdm(range(0, len(states['hidden_states']), self.batch_size),
                       desc=f"{'MTP' if is_mtp else 'MoE'} Running forward pass to "
                            f"{'collect expert activations' if collect_outputs else 'propagate hidden states'}..."):
            if not is_mtp:
                hid_states = moe_forward(self.model.model.layers[layer_ind],
                                         states,
                                         i=i_,
                                         chunk_size=self.batch_size,
                                         device=self.device).data
            else:
                if i_ == 0:
                    self.mtp_layer.to(self.device).to(states['hidden_states'].dtype)
                hid_states = self.mtp_layer(hidden_states=states['hidden_states'][i_:i_ + self.batch_size],
                                            position_ids=states['position_ids'],
                                            inputs_embeds=inputs_embeds[i_:i_ + self.batch_size],
                                            attention_mask=states['attention_mask'][i_:i_ + self.batch_size],
                                            shift=False)

            if i_ == 0 and verbose:
                n_exp = get_num_experts(self.model.model.layers[layer_ind].mlp) if hasattr(
                    self.model.model.layers[layer_ind].mlp, 'experts') and not is_mtp else \
                    (get_num_experts(self.mtp_layer.layer.mlp) if is_mtp else 'dense')
                print('moe forward, hid_states', hid_states.shape, hid_states.dtype, hid_states.device,
                      'experts', n_exp,
                      flush=True)

            if upd_hid:
                hidden_states.append(hid_states)

        if upd_hid:
            states['hidden_states'] = torch.cat(hidden_states).data
            if verbose:
                print('after moe forward, hidden_states', states['hidden_states'].shape, flush=True)

        for handle in handles:
            handle.remove()

        return expert_outs, states

    @torch.no_grad()
    def fit(self):
        total_experts = 0
        top_k = []
        for layer_ind, layer in enumerate(self.model.model.layers):
            if layer_ind < self.first_moe_layer:
                top_k.append(0)
                continue
            total_experts += get_num_experts(layer.mlp)
            top_k.append(layer.mlp.top_k)
        print(f'\nTotal experts before merging: {total_experts}', flush=True)
        print(
            f'Number of activated experts per token (top_k min-max): '
            f'{np.min([t for t in top_k if t > 0])}-{np.max(top_k)}\n',
            flush=True)

        print(f'running merging for {self.num_layers} layers', flush=True)
        states = get_moe_input(self.model, self.device, **self.batch)
        inputs_embeds = None if self.mtp_layer is None else states['hidden_states'].clone()

        # --- CHECKPOINTING: setup ---
        start_layer = 0
        checkpoint_file = None
        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(self.checkpoint_dir, 'ream_checkpoint.pt')
            if os.path.exists(checkpoint_file):
                print(f'Found checkpoint at {checkpoint_file}, resuming...', flush=True)
                ckpt = torch.load(checkpoint_file, map_location='cpu')
                self.model.load_state_dict(ckpt['model_state_dict'])
                states = {k: v for k, v in ckpt['states'].items()}
                start_layer = ckpt['completed_layer'] + 1
                print(f'Resuming from layer {start_layer}', flush=True)

        top_k = []
        total_experts = 0

        for layer_ind in range(start_layer, self.num_layers):
            verbose = self.verbose and layer_ind == 0

            if layer_ind < self.first_moe_layer:
                if self.sequential:
                    _, states = self._forward_pass(states,
                                                   layer_ind,
                                                   collect_outputs=False,
                                                   upd_hid=True)
                continue

            is_mtp = layer_ind >= len(self.model.model.layers)
            moe_layer = (self.mtp_layer.layer if is_mtp else self.model.model.layers[layer_ind]).mlp
            assert getattr(moe_layer.gate, 'bias', None) is None, ('no bias expected', moe_layer.gate)
            num_params = num_parameters(moe_layer)

            n_experts = get_num_experts(moe_layer)
            if self.merge_size < n_experts:
                expert_outs, states = self._forward_pass(states,
                                                         layer_ind,
                                                         collect_outputs=True,
                                                         upd_hid=not self.sequential,
                                                         inputs_embeds=inputs_embeds,
                                                         verbose=verbose)

                if self.use_gate_output:
                    gate_logits = torch.cat(expert_outs['gate']).view(-1, n_experts)
                else:
                    gate_logits = None

                expert_logits = expert_outs['final']
                expert_act = torch.cat(expert_outs['hid_act'], dim=1)
                saliency = expert_outs['saliency']

                if verbose:
                    print(f'layer {layer_ind}: expert_logits', expert_logits.shape, expert_logits.dtype,
                          'expert_act', expert_act.shape, expert_act.dtype,
                          'gate_logits', (gate_logits.shape, gate_logits.dtype) if self.use_gate_output else 'None',
                          f'{self.saliency} saliency scores', saliency.shape,
                          [str('%.8f' % f) for f in saliency], flush=True)

                zeros = saliency == 0
                n_zeros = zeros.sum().item()
                if n_zeros > 0:
                    assert n_zeros < len(saliency), saliency
                    saliency[zeros] = min(0.5, saliency[saliency > 0].min().item())

                print(f'\nrunning {self.grouping} grouping for {self.group_size} clusters on {self.device}')

                expert_weights = None
                if 'weight' in self.merging:
                    if isinstance(moe_layer.experts, nn.ModuleList):
                        expert_weights = torch.stack(
                            [ffn_weight_matrix(mlp) for mlp in moe_layer.experts])
                    else:
                        expert_weights = experts_weight_matrix(moe_layer.experts)
                    weights_sz = expert_weights.shape
                    expert_weights = pca_reduce(normalize_rows(expert_weights).flatten(0, 1),
                                                r=self.pca_dim,
                                                verbose=verbose)
                    expert_weights = expert_weights.reshape(n_experts, -1, self.pca_dim)
                    if verbose:
                        print(f'layer {layer_ind}: expert_weights', weights_sz,
                                'expert_weights low dim', expert_weights.shape, expert_weights.dtype,
                                flush=True)

                if self.grouping == 'hcsmoe':
                    cluster_lbl, centroid_inds = hcsmoe(expert_logits, self.merge_size)
                elif self.grouping == 'ream':
                    cluster_lbl, centroid_inds = pseudo_group(saliency,
                                                              expert_logits=expert_logits,
                                                              k=self.merge_size,
                                                              gate_logits=gate_logits,
                                                              group_size=self.group_size)
                else:
                    raise NotImplementedError(self.grouping)

                groups = []
                group_sizes_stat = {}
                for j in range(self.merge_size):
                    group = list(map(int, np.where(cluster_lbl == j)[0]))
                    if centroid_inds is not None and len(group) > 1:
                        group = list(sorted(group, key=lambda g: g not in centroid_inds))
                    groups.append(np.asarray(group))
                    s = len(group)
                    if s not in group_sizes_stat:
                        group_sizes_stat[s] = 0
                    group_sizes_stat[s] += 1

                print(f'grouping into {self.merge_size} groups is done', flush=True)
                if verbose:
                    for s in sorted(group_sizes_stat.keys()):
                        print(f'{group_sizes_stat[s]} groups with {s} experts', flush=True)

                experts_to_delete = []
                experts_to_keep = []
                experts = None
                if not isinstance(moe_layer.experts, nn.ModuleList):
                    from copy import deepcopy
                    experts = deepcopy(moe_layer.experts)

                for group_ind, group in enumerate(groups):
                    if verbose:
                        print(f'layer {layer_ind}: group_ind:', group_ind, 'size:', len(group),
                              'group:', group,
                              'saliency:', [float(saliency[ind]) for ind in group], flush=True)

                    if len(group) > 1:
                        merged_exp = self._merge(experts=moe_layer.experts,
                                                 group=group,
                                                 saliency=saliency,
                                                 expert_weights=expert_weights,
                                                 expert_act=expert_act
                                                 )
                        if experts is None:
                            moe_layer.experts[group[0]] = merged_exp
                        else:
                            experts.gate_up_proj.data[group[0]] = merged_exp.gate_up_proj.data
                            experts.down_proj.data[group[0]] = merged_exp.down_proj.data

                        experts_to_delete.extend(group[1:])
                    experts_to_keep.append(group[0])

                experts_to_delete.sort(reverse=True)
                experts_to_keep.sort()
                experts_to_keep = np.array(experts_to_keep)
                if experts is None:
                    for idx in experts_to_delete:
                        del moe_layer.experts[idx]
                else:
                    experts.gate_up_proj.data = experts.gate_up_proj.data[experts_to_keep]
                    experts.down_proj.data = experts.down_proj.data[experts_to_keep]
                    experts.num_experts = len(experts_to_keep)
                    moe_layer.experts = experts

                moe_layer.num_experts = get_num_experts(moe_layer)
                moe_layer.gate.weight.data = moe_layer.gate.weight.data[experts_to_keep]
                moe_layer.gate.out_features = get_num_experts(moe_layer)
                if hasattr(moe_layer.gate, 'n_routed_experts'):
                    moe_layer.gate.n_routed_experts = get_num_experts(moe_layer)
                if hasattr(moe_layer.gate, 'e_score_correction_bias'):
                    moe_layer.gate.e_score_correction_bias = moe_layer.gate.e_score_correction_bias[experts_to_keep]

                if is_mtp:
                    self.mtp_layer.layer.mlp = moe_layer
                    self.mtp_state_dict = self.mtp_layer.state_dict()

            elif self.merge_size > n_experts:
                raise ValueError(f'merge_size {self.merge_size} > n_experts {n_experts}')
            else:
                print(f'\nlayer {layer_ind}: k={self.merge_size}, '
                      f'so model is already compressed, skipping compression', flush=True)

            assert moe_layer.num_experts == self.merge_size, (moe_layer.num_experts, self.merge_size)

            total_experts += get_num_experts(moe_layer)
            top_k.append(moe_layer.top_k)

            if self.sequential:
                _, states = self._forward_pass(states,
                                               layer_ind,
                                               collect_outputs=False,
                                               upd_hid=True,
                                               inputs_embeds=inputs_embeds,
                                               verbose=verbose)

            if is_mtp:
                self.mtp_layer.to('cpu')
            else:
                self.model.model.layers[layer_ind].to('cpu')

            print('finished layer', layer_ind,
                  'num experts orig', n_experts,
                  'num experts merge', get_num_experts(self.mtp_layer.layer.mlp
                                           if is_mtp else self.model.model.layers[layer_ind].mlp),
                  'num params orig', num_params,
                  'num params merged', num_parameters(self.mtp_layer.layer.mlp
                                                      if is_mtp else self.model.model.layers[layer_ind].mlp),
                  'cpu mem=%.2f' % mem('cpu'),
                  'gpu mem=%.2f' % mem(0),
                  '\n\n',
                  flush=True)

            # --- CHECKPOINTING: save after each layer ---
            if checkpoint_file is not None:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'states': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in states.items()},
                    'completed_layer': layer_ind,
                }, checkpoint_file)
                print(f'Checkpoint saved at layer {layer_ind}', flush=True)

            torch.cuda.empty_cache()

        if self.verbose:
            print(self.model.model.layers[self.first_moe_layer].mlp, flush=True)
            print(self.model.model.layers[self.first_moe_layer + 1].mlp, flush=True)
            print(self.model.model.layers[-1].mlp, flush=True)

        moe_layer = self.model.model.layers[self.first_moe_layer].mlp
        self.model.config.num_experts = moe_layer.num_experts
        self.model.config.num_experts_per_tok = moe_layer.top_k
        if hasattr(self.model.config, 'n_routed_experts'):
            self.model.config.n_routed_experts = moe_layer.num_experts

        print(f'\nTotal experts after merging: {total_experts}', flush=True)
        if len(top_k) > 0:
            print(f'Number of activated experts per token (top_k min-max): '
                  f'{np.min(top_k)}-{np.max(top_k)}\n',
                  flush=True)

        return self.model

    @torch.no_grad()
    def _merge(self,
               experts,
               group,
               saliency=None,
               expert_weights: torch.Tensor = None,
               expert_act: torch.Tensor = None) -> nn.Module:

        assert len(group) > 0, group
        def get_expert(ind):
            if isinstance(experts, nn.ModuleList):
                return experts[ind]
            else:
                from copy import deepcopy
                exp = deepcopy(experts)
                exp.gate_up_proj.data = exp.gate_up_proj.data[ind]
                exp.down_proj.data = exp.down_proj.data[ind]
                return exp

        merged = get_expert(group[0])

        if len(group) == 1 or self.merging == 'none':
            return merged

        if self.merging == 'avg':
            for name, param in merged.named_parameters():
                stacked = torch.stack([get_expert(i).state_dict()[name] for i in group], dim=0)
                avg_param = stacked.mean(dim=0).to(param)
                param.copy_(avg_param)
        else:
            if isinstance(experts, nn.ModuleList):
                assert len(saliency) == len(experts), (len(saliency), len(experts))
            else:
                assert len(saliency) == experts.num_experts
            if isinstance(saliency, torch.Tensor):
                saliency = saliency.cpu().numpy()
            w = saliency[group]
            w_sum = w.sum()
            assert np.all(w >= 0) and w_sum > 0, (w, w_sum)
            w = w / w_sum

            use_logits = 'logits' in self.merging
            use_weights = 'weights' in self.merging

            if use_logits or use_weights:
                if use_logits:
                    expert_act = expert_act.permute(0, 2, 1).float()
                    act_centroid = expert_act[group[0]].to(self.device)
                    act_centroid = normalize_rows(act_centroid) if act_centroid.shape[-1] > 1 else act_centroid
                if use_weights:
                    expert_weights = normalize_rows(expert_weights)

                perm_lst = []
                for j, idx in enumerate(group[1:]):
                    cost = 0
                    if use_logits:
                        act_j = expert_act[idx].to(self.device)
                        act_j = normalize_rows(act_j) if act_j.shape[-1] > 1 else act_j
                        cost += torch.cdist(act_centroid, act_j).cpu().numpy()
                    if use_weights:
                        cost += torch.cdist(expert_weights[group[0]].to(self.device),
                                           expert_weights[idx].to(self.device)).cpu().numpy()
                    _, col_ind = linear_sum_assignment(cost)
                    perm_lst.append(col_ind)
            else:
                assert self.merging == 'avg_freq', f'{self.merging} is an unsupported merging method'

            for j, idx in enumerate(group[1:]):
                if self.merging == 'avg_freq':
                    aligned_b = get_expert(idx)
                else:
                    aligned_b = apply_perm_to_ffn(get_expert(idx), perm_lst[j])

                for name, param in merged.named_parameters():
                    if j == 0:
                        param.data = (param.data.float() * w[0]).to(param.data)
                    param.data = (param.data.float() +
                                  w[j + 1] * aligned_b.state_dict()[name].data.float()).to(param.data)

        return merged
