import torch
import torch.nn as nn
import torch.nn.functional as F

class OODML(nn.Module):
    """
    Main OODML model that encapsulates all the components.
    """
    def __init__(self, input_dim=1024, n_classes=2, K=5, embed_dim=512):
        super(OODML, self).__init__()
        self.K = K
        self.embed_dim = embed_dim
        self.spff = SPFF(input_dim=input_dim, embed_dim=embed_dim, K=K)
        self.eirl = EIRL(embed_dim=embed_dim, n_classes=n_classes)
        self.ddm = DDM(embed_dim=embed_dim, n_classes=n_classes)

    def forward(self, all_feats, all_coords):
        """
        Args:
            all_feats (Tensor): All feature vectors for a single WSI bag. 
                                Shape: [num_instances, feature_dim]
            all_coords (Tensor): All coordinate vectors for a single WSI bag.
        """
        adaptive_memory_bank = []
        
        # FIX: Split along the instance dimension (dim=0) instead of the feature dimension.
        pseudo_bags = torch.split(all_feats, 512, dim=0)
        
        pseudo_bag_tokens = []
        pseudo_label_preds = []

        for v_t in pseudo_bags: # v_t now has shape [M, feature_dim]
            F_t, y_hat_t = self.spff(v_t, adaptive_memory_bank)
            pseudo_bag_tokens.append(F_t)
            pseudo_label_preds.append(y_hat_t)

            # Simplified AMB update logic
            if len(adaptive_memory_bank) < self.K:
                adaptive_memory_bank.append({'token': F_t.detach(), 'pred': y_hat_t.detach()})
            else:
                current_preds = torch.cat([item['pred'] for item in adaptive_memory_bank])
                min_pred_val, min_pred_idx = torch.min(current_preds, 0)
                if y_hat_t.item() > min_pred_val.item():
                    adaptive_memory_bank[min_pred_idx] = {'token': F_t.detach(), 'pred': y_hat_t.detach()}
        
        if not pseudo_bag_tokens:
            # Handle cases where a bag has no instances
            return {
                "Y_hat_ER": torch.zeros(1, 2).to(all_feats.device), "Y_hat_IR": torch.zeros(1, 2).to(all_feats.device),
                "Y_hat_DM": torch.zeros(1, 2).to(all_feats.device), "A_ER": torch.zeros(1, 0).to(all_feats.device),
                "A_IR": torch.zeros(1, 0, 0).to(all_feats.device), "pseudo_label_preds": torch.zeros(0).to(all_feats.device)
            }

        F_all = torch.cat(pseudo_bag_tokens, dim=1)
        h_ER, Y_hat_ER, A_ER, h_IR, Y_hat_IR, A_IR = self.eirl(F_all)
        Y_hat_DM = self.ddm(h_ER, Y_hat_ER, h_IR, Y_hat_IR)

        results_dict = {
            "Y_hat_ER": Y_hat_ER, "Y_hat_IR": Y_hat_IR, "Y_hat_DM": Y_hat_DM,
            "A_ER": A_ER, "A_IR": A_IR, "pseudo_label_preds": torch.cat(pseudo_label_preds, dim=0)
        }
        return results_dict

class SPFF(nn.Module):
    def __init__(self, input_dim, embed_dim, K):
        super(SPFF, self).__init__()
        self.transformer_module = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.fc_in = nn.Linear(input_dim, embed_dim)
        self.initial_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mca = MemoryBasedCrossAttention(embed_dim, K)
        self.mlp_pseudo_label = nn.Linear(embed_dim, 1)

    def forward(self, v_t, adaptive_memory_bank):
        # v_t arrives with shape [M, input_dim]
        # FIX: Add a batch dimension for processing
        v_t = v_t.unsqueeze(0) # Shape becomes [1, M, input_dim]
        
        v_t_embed = self.fc_in(v_t)
        
        transformer_input = torch.cat([self.initial_token, v_t_embed], dim=1)
        u_t = self.transformer_module(transformer_input)[:, 0, :].unsqueeze(1)

        F_t = self.mca(u_t, adaptive_memory_bank)
        y_hat_t = torch.sigmoid(self.mlp_pseudo_label(F_t.squeeze(1)))
        
        return F_t, y_hat_t

class MemoryBasedCrossAttention(nn.Module):
    def __init__(self, embed_dim, K):
        super(MemoryBasedCrossAttention, self).__init__()
        self.K = K
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, u_t, adaptive_memory_bank):
        if not adaptive_memory_bank:
            return u_t

        mem_tokens = torch.cat([item['token'] for item in adaptive_memory_bank], dim=1)
        q = self.q_proj(u_t)
        k = self.k_proj(mem_tokens)
        v = self.v_proj(mem_tokens)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.K**0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)
        
        return u_t + context

class EIRL(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super(EIRL, self).__init__()
        self.lam = LinearAttention(embed_dim)
        self.mlp_er = nn.Linear(embed_dim, n_classes)
        self.msa = nn.MultiheadAttention(embed_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.h_ir_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mlp_ir = nn.Linear(embed_dim, n_classes)

    def forward(self, F_all):
        h_ER, A_ER = self.lam(F_all)
        Y_hat_ER = self.mlp_er(h_ER)

        h_IR, A_IR_weights = self.msa(query=self.h_ir_token, key=F_all, value=F_all)
        h_IR = h_IR.squeeze(1)
        Y_hat_IR = self.mlp_ir(h_IR)
        
        return h_ER, Y_hat_ER, A_ER, h_IR, Y_hat_IR, A_IR_weights

class LinearAttention(nn.Module):
    def __init__(self, embed_dim):
        super(LinearAttention, self).__init__()
        self.attention_fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        attn_weights = self.attention_fc(x).squeeze(-1)
        attn_probs = F.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_probs.unsqueeze(1), x).squeeze(1)
        return context, attn_probs

class DDM(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super(DDM, self).__init__()
        self.linear = nn.Linear(embed_dim * 2, n_classes)

    def forward(self, h_ER, Y_hat_ER, h_IR, Y_hat_IR):
        H_ER = h_ER * Y_hat_ER.softmax(dim=-1)[:, 1].unsqueeze(-1)
        H_IR = h_IR * Y_hat_IR.softmax(dim=-1)[:, 1].unsqueeze(-1)
        W = self.linear(torch.cat([H_ER, H_IR], dim=-1)).softmax(dim=-1)
        W_ER, W_IR = W[:, 0].unsqueeze(-1), W[:, 1].unsqueeze(-1)
        Y_hat_DM = W_ER * Y_hat_ER + W_IR * Y_hat_IR
        return Y_hat_DM
