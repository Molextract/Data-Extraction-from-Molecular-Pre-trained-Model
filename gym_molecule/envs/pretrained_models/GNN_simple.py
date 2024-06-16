import sys 
import os 

import torch 
import torch.nn.functional as F 
from rdkit import Chem 

from gym_molecule.envs.pretrained_models.context_pred.model import GNN_graphpred
from gym_molecule.envs.pretrained_models.context_pred.loader import mol_to_graph_data_obj_simple
from gym_molecule.envs.env_utils_graph import * 


class GNN_Vina(object):
    def __init__(self, params):
        self.device = params['device']
        if params['model']=='context_pred':
            self.model = GNN_graphpred(5, 300, 1, JK = 'last', drop_ratio = 0.5, graph_pooling = 'mean').to(self.device)
            state = torch.load('gym_molecule/envs/pretrained_models/context_pred/contextpred.pth')
            self.model.gnn.load_state_dict(state)
            self.model.eval()
        else:
            raise NotImplementedError
        
    def predict(self, smile_list, record_list):
        scores = []
        self.model.eval()
        for smi, record in zip(smile_list, record_list):
            G_hat = mol_to_graph_data_obj_simple(Chem.MolFromSmiles(smi))
            if self.device is not None:
                G_hat = G_hat.to(self.device)
            G_hat_rep = self.model.forward_graph_representation(G_hat)

            scaffold = record[0][0]
            scaffold = Chem.DeleteSubstructs(scaffold, Chem.MolFromSmiles("*"))
            side_chains = [FRAG_VOCAB_MOL_WO_ATT[sc[1]] for _, sc in record]

            sizes = torch.tensor([scaffold.GetNumAtoms()] +  [m.GetNumAtoms() for m in side_chains])
            if self.device is not None:
                sizes = sizes.to(self.device)
    
            graphs = [mol_to_graph_data_obj_simple(scaffold)] + [mol_to_graph_data_obj_simple(m) for m in side_chains]
            if self.device is not None:
                graphs = [g.to(self.device) for g in graphs]
            reps = [self.model.forward_graph_representation(g)  for g in graphs]

            rep_agg = (sizes[:,np.newaxis] * torch.cat(reps)).sum(0, keepdim=True) / sizes.sum()

            score = F.cosine_similarity(rep_agg, G_hat_rep).item()

            scores.append(score)
        
        return 10 * np.array(scores)




            

