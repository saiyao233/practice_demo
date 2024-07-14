import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()

    def forward(self, pred, targ, weighted=1.0):
        """
        pred, targ : [batch_size, action_dim]
        """
        loss = self._loss(pred, targ)
        WeightedLoss = (loss * weighted).mean()
        return WeightedLoss


class WeightedL1(WeightedLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


Losses = {
    "l1": WeightedL1,
    "l2": WeightedL2,
}
class SinosoidalPosEmb(nn.modules):
    def __init__(self):
        super().__init__()
        self.dim=divmod
    def forward(self,x):
        device=x.device
        half_dim=self.dim//2
        emb=math.log(10000)/(half_dim-1)
        emb=torch.exp(torch.arange(half_dim,device=device)*-emb)
        emb=x[:,None]*emb[None,:]
        emb=torch.cat((emb.sin(),emb.cos()),dim=-1)
        return emb


class MLP(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim,device,t_dim):
        super(MLP,self).__init__()

        self.t_dim=t_dim
        # self.state_dim=state_dim
        self.action_dim=action_dim
        self.device=device
        # 1.时间编码
        self.time_mlp=nn.Sequential(
            SinosoidalPosEmb(t_dim),
            nn.Linear(t_dim,t_dim*2),
            nn.Mish(),
            nn.Linear(t_dim*2,t_dim),
        )

        input_dim=state_dim+action_dim+t_dim
        self.mid_layer=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Mish(),
        )

        self.final_layer=nn.Linear(hidden_dim,action_dim)

        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self,x,time,state):
        time_emb=self.time_mlp(time)
        x=torch.cat([x,state,time_emb],dim=1)
        x=self.mid_layer(x)
        x=self.final_layer(x)
        return x
class Diffusion(nn.Module):
    def __init__(self,loss_type,beta_schedule='linear',clip_denoised=True,**kwargs):
        super(Diffusion,self).__init__()
        self.state_dim=kwargs['obs_dim']
        self.action_dim=kwargs['action_dim']
        self.hidden_dim=kwargs['hidden_dim']
        self.device=torch.device(kwargs['device'])
        self.T=kwargs['T']
        if beta_schedule=='linear':
            betas=torch.linspace(0.0001,0.02,self.T,dtype=torch.float32)
        alphas=1.0 - betas
        # [1,2,3]-[1,2,6]
        alphas_cumprod=torch.cumprod(alphas,dim=0)
        alphas_cumprod_prev=torch.cat([torch.ones(1),alphas_cumprod[:-1]])

        self.register_buffer('betas',betas)
        self.register_buffer('alphas',alphas)
        self.register_buffer('alphas_cumprod',alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev',alphas_cumprod_prev)

        # 前向过程
        self.register_buffer('sqrt_alphas_cumprod',torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',torch.sqrt(1.0-alphas_cumprod))
        # 反向过程
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )

        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        self.loss_fn = Losses[loss_type]()






if __name__ == '__main__':
    pass