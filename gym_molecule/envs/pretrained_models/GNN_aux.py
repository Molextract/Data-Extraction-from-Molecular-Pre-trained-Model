import sys
import os

import torch
import torch.nn.functional as F
from rdkit import Chem
from torch import nn

from gym_molecule.envs.pretrained_models.context_pred.model import GNN_graphpred
from gym_molecule.envs.pretrained_models.context_pred.loader import mol_to_graph_data_obj_simple
from gym_molecule.envs.env_utils_graph import *
import argparse
from tqdm import tqdm
import pickle as pkl
from rdkit import Chem
from molecule_generation.utils_graph import *
from molecule_generation.molecule_graph import Molecule_Graph
import torch.optim as optim
import matplotlib.pyplot as plt


def calculate_extraction(records, scaffold_idx):
    scaffold_smi = SCAFFOLD_VOCAB[scaffold_idx]
    core2id = pkl.load(open(r'/data/renhong/mol-attack/data_preprocessing/zinc/core2mol_bank/core2id.pkl', 'rb'))
    mol_bank_id = core2id[Chem.MolToSmiles(Molecule_Graph(scaffold_smi).mol_core)]

    att = Chem.MolFromSmiles('*')
    H = Chem.MolFromSmiles('[H]')

    freq_list = []
    mol_bank_filtered = pkl.load(
        open(f'/data/renhong/mol-attack/data_preprocessing/zinc/core2mol_bank/{mol_bank_id}.pkl', 'rb'))
    for i in tqdm(range(len(records))):
        mg = records[i][0]
        mg1 = sanitize(Chem.ReplaceSubstructs(mg, att, H, replaceAll=True)[0])
        matched_mols = [mol for mol in mol_bank_filtered if mol.HasSubstructMatch(mg1)]
        freq = len(matched_mols)
        if freq > 0:
            freq_list.append(1)
        else:
            freq_list.append(0)
    return freq_list


class LearnTransform(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.dim = 300
        self.transform_merged = nn.Sequential(
            nn.Linear(emb_dim, self.dim),
        )
        self.transform_motif = nn.Sequential(
            nn.Linear(emb_dim, self.dim),
        )
        self.transform_scaffold = nn.Sequential(
            nn.Linear(emb_dim, self.dim),
        )
        self.transform_alpha = nn.Sequential(
            nn.Linear(3 * emb_dim, 1),
        )

    def forward(self, motif_representations, scaffold_representations, merged_representations, scaffold_sizes,
                motif_sizes):
        x = torch.cat([scaffold_representations, motif_representations, merged_representations], dim=-1)
        alpha = self.transform_alpha(x)

        motif_representations = self.transform_motif(motif_representations)
        scaffold_representations = self.transform_scaffold(scaffold_representations)
        merged_mol_representations = self.transform_merged(merged_representations)
        estimated_representations = (
                                            alpha * scaffold_representations * scaffold_sizes + (
                                            1 - alpha) * motif_representations * motif_sizes) / (
                                            scaffold_sizes + motif_sizes)
        score = (F.cosine_similarity(estimated_representations, merged_mol_representations) + 1) / 2

        return score

    def l1_regularizer(self, ):
        l1 = 0.0
        for param in self.transform.parameters():
            l1 += torch.sum(torch.abs(param))
        return l1

    def l2_regularizer(self, ):
        l2 = 0.0
        for param in self.transform.parameters():
            l2 += torch.norm(param, 2)
        return l2

    def loss(self, labels, motif_representations, scaffold_representations, merged_mol_representations, scaffold_sizes,
             motif_sizes):
        x = torch.cat([scaffold_representations, motif_representations, merged_mol_representations], dim=-1)
        alpha = self.transform_alpha(x)

        motif_representations = self.transform_motif(motif_representations)
        scaffold_representations = self.transform_scaffold(scaffold_representations)
        merged_mol_representations = self.transform_merged(merged_mol_representations)
        estimated_representations = (
                                            alpha * scaffold_representations * scaffold_sizes + (
                                            1 - alpha) * motif_representations * motif_sizes) / (
                                            scaffold_sizes + motif_sizes)
        score1 = (F.cosine_similarity(estimated_representations, merged_mol_representations) + 1) / 2
        weight = labels.sum() / labels.size(0)

        loss = torch.where(labels >= 1.0, -1 * score1, weight * score1).sum()
        loss_alpha = torch.where((0.0 < alpha) & (alpha <= 1.0), torch.tensor([0.0]), torch.abs(alpha)).sum()
        loss = loss + 50 * loss_alpha
        return loss

    def forward_alpha(self, motif_representations, scaffold_representations, merged_mol_representations, ):
        x = torch.cat([scaffold_representations, motif_representations, merged_mol_representations], dim=-1)
        alpha = self.transform_alpha(x)
        return alpha


class GNN_Aux(nn.Module):
    def __init__(self, params):
        super(GNN_Aux, self).__init__()
        self.device = params['device']
        self.model = GNN_graphpred(5, 300, 1, JK='last', drop_ratio=0.5, graph_pooling='mean').to(self.device)
        if params['model'] == 'contextpred':
            state = torch.load('./saved/contextpred.pth')
            self.model.gnn.load_state_dict(state)
        elif params['model'] == 'gcl':
            state = torch.load('./saved/graphcl_80.pth',
                               map_location={'cuda:0': 'cuda:' + str(self.device.index)})
            self.model.gnn.load_state_dict(state)
        elif params['model'] == 'masking':
            state = torch.load('./saved/masking.pth')
            self.model.gnn.load_state_dict(state)
        elif params['model'] == 'infomax':
            state = torch.load('./saved/infomax.pth')
            self.model.gnn.load_state_dict(state)
        elif params['model'] == 'mae':
            state = torch.load('./saved/pretrained.pth')
            self.model.gnn.load_state_dict(state)
        elif params['model'] == 'simgrace':
            state = torch.load('./saved/simgrace_100.pth')
            self.model.gnn.load_state_dict(state)
        elif params['model'] == 'molebert':
            state = torch.load('./saved/Mole-BERT.pth')
            self.model.gnn.load_state_dict(state)
        elif params['model'] == 'supercont':
            state = torch.load('./saved/supervised_contextpred.pth')
            self.model.gnn.load_state_dict(state)
        elif params['model'] == 'superedge':
            state = torch.load('./saved/supervised_edgepred.pth')
            self.model.gnn.load_state_dict(state)
        elif params['model'] == 'superinfo':
            state = torch.load('./saved/supervised_infomax.pth')
            self.model.gnn.load_state_dict(state)
        elif params['model'] == 'supermasking':
            state = torch.load('./saved/supervised_masking.pth')
            self.model.gnn.load_state_dict(state)
        elif params['model'] == 'supervised':
            state = torch.load('./saved/supervised.pth')
            self.model.gnn.load_state_dict(state)
        else:
            raise NotImplementedError

        self.model_type = params['model']
        self.reward_type = params['reward']
        self.model.eval()
        self.model_aux = LearnTransform(300).to(self.device)
        self.model_aux.eval()

        self.lr = 0.001
        self.decay = 5e-4
        self.params = params
        if params['model'] != None:
            self.pretrain()
        print("Done!")

    def pretrain(self):
        self.scaffold_idx = 0
        records = pkl.load(
            open(
                f'./records/records_{self.model_type}_ring/scaffold_{self.scaffold_idx}.pkl',
                'rb'))
        scaffold_representations, scaffold_sizes, motif_representations, motif_sizes, merged_representations, labels, labels_ground = self.get_representations(
            records)
        optimizer = optim.Adam(self.model_aux.parameters(), lr=self.lr, weight_decay=self.decay)

        self.model_aux.train()
        if self.device is not None:
            scaffold_representations = scaffold_representations.to(self.device)
            scaffold_sizes = scaffold_sizes.to(self.device)
            motif_representations = motif_representations.to(self.device)
            motif_sizes = motif_sizes.to(self.device)
            merged_representations = merged_representations.to(self.device)
            labels = labels.to(self.device)
            labels_ground = labels_ground.to(self.device)

        tmp = torch.tensor([0.0]).to(self.device)
        loss_list = []
        for e in tqdm(range(200)):
            optimizer.zero_grad()

            x = torch.cat([scaffold_representations, motif_representations, merged_representations], dim=-1)
            # alpha = torch.sigmoid(self.transform_alpha(x))
            alpha = self.model_aux.transform_alpha(x)

            motif_rep = self.model_aux.transform_motif(motif_representations)
            scaffold_rep = self.model_aux.transform_scaffold(scaffold_representations)
            merged_rep = self.model_aux.transform_merged(merged_representations)
            estimated_rep = (
                                    alpha * scaffold_rep * scaffold_sizes + (
                                    1 - alpha) * motif_rep * motif_sizes) / (
                                    scaffold_sizes + motif_sizes)
            # score = (F.cosine_similarity(estimated_rep, merged_rep) + 1) / 2
            score = F.cosine_similarity(estimated_rep, merged_rep)
            weight = labels.sum() / labels.size(0)

            loss = torch.where(labels >= 1.0, -1 * score, weight * score).sum()
            loss_alpha = torch.where((0.0 < alpha) & (alpha <= 1.0), tmp, torch.abs(alpha)).sum()
            loss = loss + 50 * loss_alpha
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
        # plt.plot([i for i in range(len(loss_list))], loss_list)
        # plt.show()

    def predict(self, smile_list, record_list):
        scores = []
        self.model.eval()
        for smi, record in zip(smile_list, record_list):
            motif_chains = [FRAG_VOCAB_MOL_WO_ATT[sc[1]] for _, sc in record]
            scaffold_chains = [Chem.DeleteSubstructs(sc, Chem.MolFromSmiles("*")) for sc, _ in record]

            motif_sizes = torch.tensor([m.GetNumAtoms() for m in motif_chains])
            scaffold_sizes = torch.tensor([m.GetNumAtoms() for m in scaffold_chains])

            motif_graphs = [mol_to_graph_data_obj_simple(m) for m in motif_chains]
            scaffold_graphs = [mol_to_graph_data_obj_simple(m) for m in scaffold_chains]
            G_hat = mol_to_graph_data_obj_simple(Chem.MolFromSmiles(smi))
            if len(motif_sizes) > 1:
                merged_graphs = scaffold_graphs[1:] + [G_hat]
            else:
                merged_graphs = [G_hat]

            if self.device is not None:
                motif_sizes = motif_sizes.to(self.device)
                scaffold_sizes = scaffold_sizes.to(self.device)
                motif_graphs = [g.to(self.device) for g in motif_graphs]
                scaffold_graphs = [g.to(self.device) for g in scaffold_graphs]
                merged_graphs = [g.to(self.device) for g in merged_graphs]

            motif_reps = [self.model.forward_graph_representation(g) for g in motif_graphs]
            scaffold_reps = [self.model.forward_graph_representation(g) for g in scaffold_graphs]
            merged_reps = [self.model.forward_graph_representation(g) for g in merged_graphs]

            alpha_list = []
            score_list = []
            for i in range(len(motif_reps)):
                x = torch.cat([scaffold_reps[i], motif_reps[i], merged_reps[i]], dim=-1)
                alpha = self.model_aux.transform_alpha(x)
                motif_representations = self.model_aux.transform_motif(motif_reps[i])
                scaffold_representations = self.model_aux.transform_scaffold(scaffold_reps[i])
                merged_representations = self.model_aux.transform_merged(merged_reps[i])
                estimated_representations = (
                                                    alpha * scaffold_representations * scaffold_sizes[i] + (
                                                    1 - alpha) * motif_representations * motif_sizes[i]) / (
                                                    scaffold_sizes[i] + motif_sizes[i])
                score = F.cosine_similarity(estimated_representations, merged_representations)
                alpha_list.append(alpha.detach().cpu().numpy())
                score_list.append(score.detach().cpu().numpy())
            label = calculate_extraction(record, self.params['scaffold_idx'])
            if self.reward_type == 'score':
                result = np.sum(np.array(score_list).reshape(-1) * np.array(label))
                scores.append(result)
            elif self.reward_type == 'ori':
                result = np.sum(np.array(score_list).reshape(-1))
                scores.append(result)
            elif self.reward_type == 'aux':
                result = 1
                for k in range(len(label)):
                    result = result * label[k]
                scores.append(result)
        print(scores)
        return 5.0 * np.array(scores)


    def get_representations(self, records):
        scaffold_representations = []
        side_chain_representations = []
        merged_mol_representations = []
        scaffold_sizes = []
        side_chain_sizes = []

        labels = []
        labels_ground = []

        invalid_count = 0
        for r in tqdm(records):
            if np.abs(r.other['representation_merged_mol']).max() > 1e4:
                invalid_count += 1
                continue
            scaffold_representations.append(r.other['representation_scaffold'])
            side_chain_representations.append(r.other['representation_side_chain'])
            merged_mol_representations.append(r.other['representation_merged_mol'])
            scaffold_sizes.append(r.scaffold_mg.atom_num)
            side_chain_sizes.append(r.side_chain_mg.atom_num)

            labels.append(1. if r.other['freq_aux'] > 0 else 0.)
            labels_ground.append(1. if r.other['freq'] > 0 else 0.)

        scaffold_representations = torch.tensor(scaffold_representations)
        scaffold_sizes = torch.tensor(scaffold_sizes)[:, np.newaxis]
        side_chain_representations = torch.tensor(side_chain_representations)
        side_chain_sizes = torch.tensor(side_chain_sizes)[:, np.newaxis]
        merged_mol_representations = torch.tensor(merged_mol_representations)
        labels = torch.tensor(labels)
        labels_ground = torch.tensor(labels_ground)

        return scaffold_representations, scaffold_sizes, side_chain_representations, side_chain_sizes, merged_mol_representations, labels, labels_ground
