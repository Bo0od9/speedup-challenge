import torch
import torch.nn as nn
import torch.nn.functional as F


class TromptCell(nn.Module):
    def __init__(self, n_columns, n_prompts, d_model):
        super().__init__()
        # Embeddings (Figure 3.2)
        self.feature_emb_weight = nn.Parameter(torch.empty(n_columns, d_model))
        self.feature_emb_bias = nn.Parameter(torch.empty(n_columns, d_model))
        self.ln_emb = nn.LayerNorm(d_model)

        # Importance Getter (Figure 3.1)
        self.ln_col = nn.LayerNorm(d_model)
        self.ln_prompt = nn.LayerNorm(d_model)
        self.dense_imp = nn.Linear(2 * d_model, d_model)

        self.emb_column = nn.Parameter(torch.empty(n_columns, d_model))
        self.emb_prompt = nn.Parameter(torch.empty(n_prompts, d_model))

        # Modified expansion block (Figure 3.3)
        # Without non-linearities! This is important to make significant speed-ups possible.
        self.dense_expand = nn.Linear(1, n_prompts)

        self.reset_parameters()

    def reset_parameters(self):
        d_rsqrt = self.feature_emb_weight.shape[1] ** -0.5
        nn.init.uniform_(self.feature_emb_weight, -d_rsqrt, d_rsqrt)
        nn.init.uniform_(self.feature_emb_bias, -d_rsqrt, d_rsqrt)
        nn.init.normal_(self.emb_column, std=0.01)
        nn.init.normal_(self.emb_prompt, std=0.01)

    def forward(self, x: torch.Tensor, prev_cell_out: torch.Tensor) -> torch.Tensor:
        # x_emb = x.unsqueeze(-1) * self.feature_emb_weight + self.feature_emb_bias.unsqueeze(0) # [B,N,1]*[N,D] + [1,N,D] есть броадкаст операции и при умножении получаем много elementwise ядер + + тензор может стать non-contiguous из-за чего могут быть copy
        # Эмбеддинг через einsum вместо покомпонентных broadcast-операций, оптимизированное cuda ядро
        x_emb = torch.einsum('bn,nd->bnd', x, self.feature_emb_weight) + self.feature_emb_bias
        x_emb = F.relu(x_emb)
        x_emb = self.ln_emb(x_emb)

        # x_prompt = self.emb_prompt.unsqueeze(0).repeat(x_emb.shape[0], 1, 1) # repeat создае реальную копию, заменим на expand который создаст нужный view
        B = x.shape[0]
        x_prompt = self.emb_prompt.unsqueeze(0).expand(B, -1, -1)
        x_prompt = self.dense_imp(torch.cat([self.ln_prompt(x_prompt), prev_cell_out], dim=-1)) + x_prompt
        # x_column = self.ln_col(self.emb_column.unsqueeze(0).repeat(x_emb.shape[0], 1, 1)) # сначала растягивается, а потом считается ln_col, можно наоборот
        x_column = self.ln_col(self.emb_column).unsqueeze(0).expand(B, -1, -1)
        mask = torch.softmax(x_prompt @ x_column.transpose(1,2), dim=-1)

        # x_emb = x_emb.unsqueeze(1) + self.dense_expand(x_emb.unsqueeze(-1)).permute(0, 3, 1, 2) # self.dense_expand выдате большой тензор, который потом permute -> возможно non-contiguous -> много copy
        # x_out = (mask.unsqueeze(-1) * x_emb).sum(dim=2) # для большого тензора поэлементные умножения (броадкастит mask до x_emb) и суммирование
        # sum(mask * x_emb) == bmm(mask, x_emb)
        # КОД НИЖЕ ПРЕДЛОЖЕН ChatGPT. Промпт типа: "Предложи как оптимизировать этот код, чтобы избавиться от работы с большими тензорами..."
        V = torch.bmm(mask, x_emb)
        w = self.dense_expand.weight.squeeze(-1)
        b = self.dense_expand.bias
        x_out = V * (1.0 + w.view(1, -1, 1)) + b.view(1, -1, 1)
        return x_out


class TromptDownstream(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dense0 = nn.Linear(d_model, 1)
        self.dense1 = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dense_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pw = torch.softmax(self.dense0(x).squeeze(-1), dim=-1)
        xnew = (pw.unsqueeze(-1) * x).sum(dim=-2)
        return self.dense_out(self.ln(F.relu(self.dense1(xnew))))


class Trompt(nn.Module):
    def __init__(self, n_columns, n_prompts, d_model, n_cycles):
        super().__init__()
        self.tcells = nn.ModuleList([TromptCell(n_columns, n_prompts, d_model) for _ in range(n_cycles)])
        self.tdown = TromptDownstream(d_model)
        self.prompt = nn.Parameter(torch.empty(n_prompts, d_model))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.prompt, std=0.01)

    def forward(self, x):
        # x_prompt = self.prompt.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x_prompt = self.prompt.unsqueeze(0).expand(x.shape[0], -1, -1)
        outputs = []
        for cell in self.tcells:
            outputs.append(self.tdown(cell(x, x_prompt)))
        return torch.stack(outputs, dim=1).squeeze(-1)