import sys
import os

import torch
import torch.nn.functional as F
from rdkit import Chem
from torch import nn

from gym_molecule.envs.pretrained_models.context_pred.model import GNN_graphpred
from gym_molecule.envs.pretrained_models.context_pred.loader import mol_to_graph_data_obj_simple
from gym_molecule.envs.env_utils_graph import *


class GNN_Aux(nn.Module):
    def __init__(self, params):
        super(GNN_Aux, self).__init__()
        self.device = params['device']
        self.model = GNN_graphpred(5, 300, 1, JK='last', drop_ratio=0.5, graph_pooling='mean').to(self.device)
        if params['model'] == 'contextpred':
            state = torch.load('./saved/contextpred.pth')
            self.model.gnn.load_state_dict(state)
        elif params['model'] == 'gcl':
            state = torch.load('./saved/graphcl_80.pth')
            self.model.gnn.load_state_dict(state)
        elif params['model'] == 'masking':
            state = torch.load('./saved/masking.pth')
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

        self.model.gnn.load_state_dict(state)
        self.model.eval()
        self.model_aux = LearnTransform(300).to(self.device)
        self.model_aux.eval()

    def predict(self, smile_list, record_list):
        scores = []
        self.model.eval()
        for smi, record in zip(smile_list, record_list):
            G_hat = mol_to_graph_data_obj_simple(Chem.MolFromSmiles(smi))
            if self.device is not None:
                G_hat = G_hat.to(self.device)
            G_hat_rep_ori = self.model.forward_graph_representation(G_hat)
            G_hat_rep = self.model_aux.forward_representation(G_hat_rep_ori)

            scaffold = record[0][0]
            scaffold = Chem.DeleteSubstructs(scaffold, Chem.MolFromSmiles("*"))
            side_chains = [FRAG_VOCAB_MOL_WO_ATT[sc[1]] for _, sc in record]

            sizes = torch.tensor([scaffold.GetNumAtoms()] + [m.GetNumAtoms() for m in side_chains])
            if self.device is not None:
                sizes = sizes.to(self.device)

            graphs = [mol_to_graph_data_obj_simple(scaffold)] + [mol_to_graph_data_obj_simple(m) for m in side_chains]
            if self.device is not None:
                graphs = [g.to(self.device) for g in graphs]
            reps = [self.model.forward_graph_representation(g) for g in graphs]

            alphas = [self.model_aux.forward_alpha(reps[i], reps[0], G_hat_rep_ori, sizes[0:1], sizes[i:i + 1]) for i in
                      range(1, len(reps))]
            alphas = torch.cat(alphas)

            if self.device is not None:
                alphas = torch.vstack([torch.tensor([[alphas.mean()]], device=self.device), 1 - alphas])
            else:
                alphas = torch.vstack([torch.tensor([[alphas.mean()]]), 1 - alphas])

            weights = sizes[:, np.newaxis] * alphas

            reps = [self.model_aux.forward_representation(rep) for rep in reps]
            rep_agg = (weights * torch.cat(reps)).sum(0, keepdim=True) / weights.sum()

            score = F.cosine_similarity(rep_agg, G_hat_rep).item()
            scores.append(score)

        return 5.0 * (np.array(scores) + 1.0)


class AlphaNet(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.transform_side = nn.Sequential(
            nn.Linear(emb_dim, self.dim),
        )
        self.transform_scaffold = nn.Sequential(
            nn.Linear(emb_dim, self.dim),
        )
        self.transform_merged = nn.Sequential(
            nn.Linear(emb_dim, self.dim),
        )

    def forward(self, side_chain_representations, scaffold_representations, merged_mol_representations, scaffold_sizes,
                side_chain_sizes):
        side_chain_representations = self.transform_side(side_chain_representations)
        scaffold_representations = self.transform_scaffold(scaffold_representations)
        merged_mol_representations = self.transform_merged(merged_mol_representations)
        estimated_representations = (
                                            scaffold_representations * scaffold_sizes + side_chain_representations * side_chain_sizes) / (
                                            scaffold_sizes + side_chain_sizes)

        alpha = torch.einsum('ik,ik->i', [merged_mol_representations, estimated_representations])[:, np.newaxis]
        return alpha


class LearnTransform(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.alpha_net = AlphaNet(emb_dim)

    def forward(self, side_chain_representations, scaffold_representations, merged_mol_representations, scaffold_sizes,
                side_chain_sizes):
        alpha = self.alpha_net(side_chain_representations, scaffold_representations, merged_mol_representations,
                               scaffold_sizes, side_chain_sizes)

        side_chain_representations = self.alpha_net.transform_side(side_chain_representations)
        scaffold_representations = self.alpha_net.transform_scaffold(scaffold_representations)
        merged_mol_representations = self.alpha_net.transform_merged(merged_mol_representations)

        estimated_representations = (alpha * scaffold_representations * scaffold_sizes + (
                1 - alpha) * side_chain_representations * side_chain_sizes) / (
                                            alpha * scaffold_sizes + (1 - alpha) * side_chain_sizes)
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

    def alpha_loss(self, side_chain_representations, scaffold_representations, merged_mol_representations,
                   scaffold_sizes, side_chain_sizes):
        alpha = self.alpha_net(side_chain_representations, scaffold_representations, merged_mol_representations,
                               scaffold_sizes, side_chain_sizes)
        loss_alpha = torch.where((0.0 < alpha) & (alpha <= 1.0), torch.tensor([0.0]), torch.abs(alpha)).sum()

        return loss_alpha

    def forward_representation(self, rep):
        rep = self.transform(rep)
        return rep

    def forward_alpha(self, side_chain_representations, scaffold_representations, merged_mol_representations,
                      scaffold_sizes, side_chain_sizes):
        # x = torch.cat([scaffold_representations, side_chain_representations, merged_mol_representations], dim=-1)
        device = side_chain_representations.device
        alpha = self.alpha_net(side_chain_representations, scaffold_representations, merged_mol_representations,
                               scaffold_sizes, side_chain_sizes)
        alpha = torch.where(alpha > 1.0, torch.tensor([1.0]).to(device), alpha)
        alpha = torch.where(alpha < 0.0, torch.tensor([0.0]).to(device), alpha)
        return alpha
