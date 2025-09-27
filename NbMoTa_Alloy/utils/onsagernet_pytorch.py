from IPython import embed
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init

def F_act(x):
    return F.relu(x)**2 - F.relu(x-0.5)**2

def makePDM(matA):
    """ Make Positive Definite Matrix from a given matrix
    matA has a size (batch_size x N x N) """
    AL = torch.tril(matA, 0)
    AU = torch.triu(matA, 1)
    Aant = AU - torch.transpose(AU, 1, 2)
    Asym = torch.bmm(AL, torch.transpose(AL, 1, 2))
    return Asym,  Aant

class OnsagerNet(nn.Module):
    """ A neural network to for the rhs function of an ODE,
    used to fitting data """
    def __init__(self, input_dim, n_nodes=[32, 32], forcing=True, ResNet=True,
                 pot_beta=0.1,
                 ons_min_d=0.1,
                 init_gain=0.1,
                 f_act=F_act,
                 f_linear=False,
                 embedding_dim=8,
                 ):
        super().__init__()
        
        n_nodes = np.array([input_dim + embedding_dim] + n_nodes)

        if n_nodes is None:   # used for subclasses
            return
        self.nL = n_nodes.size # 3
        self.embedding_dim = embedding_dim
        self.nVar = input_dim # 3
        self.nNodes = np.zeros(self.nL+1, dtype=np.int32) # 4
        self.nNodes[:self.nL] = n_nodes  # [3, 32, 32, 9]
        self.nNodes[self.nL] = self.nVar**2 
        self.nPot = self.nVar # 3
        self.forcing = forcing # True
        self.pot_beta = pot_beta # 0.1 
        self.ons_min_d = ons_min_d # 0.1
        self.F_act = f_act
        self.f_linear = f_linear
        if ResNet:
            self.ResNet = 1.0
            assert np.sum(n_nodes[1:]-n_nodes[1]) == 0, \
                f'ResNet structure is not implemented for {n_nodes}'
        else:
            self.ResNet = 0.0

        self.baselayer = nn.ModuleList([nn.Linear(self.nNodes[i], self.nNodes[i+1])
                                        for i in range(self.nL-1)]) # [3->32->32]
        self.MatLayer = nn.Linear(self.nNodes[self.nL-1], self.nVar**2) # [32->9]
        self.PotLayer = nn.Linear(self.nNodes[self.nL-1], self.nPot) # [32->3]
        self.PotLinear = nn.Linear(self.nVar, self.nPot) # [3->3]

        # Initialization 
        # init baselayer
        bias_eps = 0.5
        for i in range(self.nL-1):
            init.xavier_uniform_(self.baselayer[i].weight, gain=init_gain)
            init.uniform_(self.baselayer[i].bias, 0, bias_eps*init_gain)

        # init MatLayer
        init.xavier_uniform_(self.MatLayer.weight, gain=init_gain)
        w = torch.empty(self.nVar, self.nVar, requires_grad=True)
        nn.init.orthogonal_(w, gain=1.0)
        self.MatLayer.bias.data = w.view(-1, self.nVar**2)

        # init PotLayer and PotLinear
        init.orthogonal_(self.PotLayer.weight, gain=init_gain)
        init.uniform_(self.PotLayer.bias, 0, init_gain)
        init.orthogonal_(self.PotLinear.weight, gain=init_gain)
        init.uniform_(self.PotLinear.bias, 0, init_gain)

        # init lforce
        if self.forcing:
            if self.f_linear:
                self.lforce = nn.Linear(self.nVar, self.nVar) # [3->3]
            else:
                self.lforce = nn.Linear(self.nNodes[self.nL-1], self.nVar)
            init.orthogonal_(self.lforce.weight, init_gain)
            init.uniform_(self.lforce.bias, 0.0, bias_eps*init_gain)
    
    def forward(self, inputs, T):
        shape = inputs.shape
        inputs = inputs.reshape(-1, self.nVar) # [B, 3]
        with torch.enable_grad():
            inputs.requires_grad_(True)
            inputs.retain_grad()
            if self.embedding_dim > 0:
                output = self.F_act(self.baselayer[0](torch.cat([inputs, T], dim=1)))
            else:
                output = self.F_act(self.baselayer[0](inputs))
            
            for i in range(1, self.nL-1):
                output = (self.F_act(self.baselayer[i](output))
                          + self.ResNet*output)

            PotLinear = self.PotLinear(inputs)
            Pot = self.PotLayer(output) + PotLinear
            V = torch.sum(Pot**2) + self.pot_beta * torch.sum(inputs**2)

            g, = torch.autograd.grad(V, inputs, create_graph=True)
            g = - g.view(-1, self.nVar, 1)

        matA = self.MatLayer(output)
        matA = matA.view(-1, self.nVar, self.nVar)
        AM, AW = makePDM(matA)
        MW = AW+AM
        
        if self.forcing:
            if self.f_linear:
                lforce = self.lforce(inputs)
            else:
                lforce = self.lforce(output)
        

        output = torch.matmul(MW, g) + self.ons_min_d * g

        if self.forcing:
            output = output + lforce.view(-1, self.nVar, 1)    

        output = output.view(*shape)
        return output

    # def energy(self, inputs, T):
        
    #     shape = inputs.shape
    #     inputs = inputs.reshape(-1, self.nVar) # [B, 3]
    #     with torch.enable_grad():
    #         inputs.requires_grad_(True)
    #         inputs.retain_grad()
    #         output = self.F_act(self.baselayer[0](torch.cat([inputs, T], dim=1)))
            
    #         for i in range(1, self.nL-1):
    #             output = (self.F_act(self.baselayer[i](output))
    #                       + self.ResNet*output)

    #         PotLinear = self.PotLinear(inputs)
    #         V2 = self.PotLayer(output)
    #         Pot = PotLinear + V2

    #         V = torch.sum(Pot**2, -1) + self.pot_beta * torch.sum(inputs**2, -1)

    #     return V, PotLinear, V2, torch.sum(inputs**2, -1)


class DiffusitivityNet(nn.Module):
    """Generate diffusitivity term for SDE
    """
    def __init__(self, n_dim, layer_sizes=[32], mode='arbitrary',embedding_dim=8):
        super(DiffusitivityNet, self).__init__()
        self.n_dim = n_dim
        self.layer_sizes = layer_sizes
        self.mode = mode
        self.embedding_dim = embedding_dim

        layers = []


        if mode == 'arbitrary':
            self.output_layer = nn.Sequential(
                # nn.Linear(embedding_dim , 64),
                nn.Linear(embedding_dim + n_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, n_dim * n_dim)
            )
        elif mode in ['diagonal']:
            self.output_layer = nn.Sequential(
                nn.Linear(embedding_dim + n_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, n_dim)
            )

        elif mode == 'constant_diagonal':
            self.kernel = nn.Parameter(torch.ones(n_dim) * 0.1)  # Parameters are automatically considered for optimization
        
        elif mode == 'constant':
            self.kernel = nn.Parameter(torch.ones(n_dim * n_dim) * 0.1)

        else:
            raise ValueError(f"Mode {mode} is unknown")
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, x, T):
    
        if self.mode == 'arbitrary':
            
            # output = self.output_layer(T)

            # FIXME: Now the diffusion term depend on x and T
            if self.embedding_dim > 0:
                output = self.output_layer(torch.cat([x, T], dim=1))
            else:
                output = self.output_layer(x)

            output = output.view(-1, self.n_dim, self.n_dim)

        elif self.mode in ['diagonal']:
         
            if self.embedding_dim > 0:
                output = self.output_layer(torch.cat([x, T], dim=1))
            else:
                output = self.output_layer(x)
 
            output = torch.diag_embed(output)
            output = output.view(-1, self.n_dim, self.n_dim)

        elif self.mode == 'constant_diagonal':

            output = self.kernel.unsqueeze(0).repeat(x.shape[0], 1)
            output = torch.diag_embed(output)
            output = output.view(-1, self.n_dim, self.n_dim)

        elif self.mode == 'constant':

            output = self.kernel.unsqueeze(0).repeat(x.shape[0], 1)
            output = output.view(-1, self.n_dim, self.n_dim) 

        return output

class SDE_Net(nn.Module):
    def __init__(self, delta_t, mode='constant_diagonal', n_dim=3, embedding_dim=8, epsilon=1e-4):
        super().__init__()
        self.delta_t = nn.Parameter(torch.tensor(delta_t), requires_grad=False)
        self.n_dim = n_dim
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.epsilon = epsilon
        # self.T_embedding = nn.Embedding(num_embeddings=20, embedding_dim=embedding_dim)

        if embedding_dim > 0:
            self.T_embedding = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                # nn.Linear(32, 32),
                # nn.ReLU(),
                nn.Linear(32, embedding_dim)
            )

        self.drift_net = nn.Sequential(
            nn.Linear(n_dim + embedding_dim, 64),
            # nn.Linear(n_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_dim)
        )
        self.sigma_net = DiffusitivityNet(n_dim=n_dim, mode=mode, embedding_dim=embedding_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def drift(self, x, T):
        # x: [B, 3]
        if self.embedding_dim > 0:
            drift = self.drift_net(torch.cat([x, T], dim=1))  # [B, 3]
        else:
            drift = self.drift_net(x)  # [B, 3]
        return drift
    
    def custom_loss(self, x0, T0, x1, T1, dt=None, coeff=1):
        # x0: [B, n_dim]
        # x1: [B, n_dim]
        if dt == None:
            dt = self.delta_t

        if self.embedding_dim > 0:
            T0 = (T0 - 100) / (3000 - 100)
            T0 = self.T_embedding(T0.unsqueeze(-1))

        drift = self.drift(x0, T0) # [B, 3]
        sigma = self.sigma_net(x0, T0) # [B, 3, 3]
        
        cov_matrix = torch.bmm(sigma, sigma.transpose(1, 2)) * dt.view(-1, 1, 1)
        Sigma = cov_matrix * coeff + torch.eye(self.n_dim, device=x0.device) * self.epsilon
        Sigma_inv = torch.linalg.inv(Sigma)
        
        X = x1 - x0 - drift * dt
        a1 = torch.einsum('ij,ijk,ik->i', X, Sigma_inv, X)
        a1 = a1.view(-1, 1)

        a2 = torch.linalg.slogdet(Sigma)[1]  # Returns a tuple (sign, logdet); we need logdet
        a2 = a2.view(-1, 1)

        return torch.mean(a1 + a2)
    

    def predict(self, x, T, steps, dt=None):
        if dt == None:
            dt = self.delta_t
        
        # x: [B, latent_dim]
        # T: [B, 1]

        if self.embedding_dim > 0:
            T_normalized = (T - 100) / (3000 - 100)
            T = self.T_embedding(T_normalized.unsqueeze(-1))
    

        predict_tra = [x]

        for _ in range(steps-1):            

            x0 = predict_tra[-1]
            drift = self.drift(x0, T) # [B, 3]
            sigma = self.sigma_net(x0, T) # [B, 3, 3]   
            delta_W = torch.normal(mean=0., std=torch.sqrt(dt), size=(x.shape[0],self.n_dim), device=x.device)

            x1 = x0 + drift * dt + torch.einsum('ijk,ik->ij', sigma, delta_W)
            predict_tra.append(x1)

        predict_tra = torch.stack(predict_tra, 2).transpose(2, 1)
        return predict_tra
        
    def forward(self, z0, noise):
        # z0: [B, D]
        # noise: [B, D]
        drift = self.drift(z0) # [B, D]
        sigma = self.sigma_net(z0) # [B, D, D]

        z1 = z0 + drift * self.delta_t + torch.einsum('ijk,ik->ij', sigma, noise) * torch.sqrt(self.delta_t)
        return z1
        


class S_OnsagerNet(SDE_Net):
    def __init__(self, delta_t,
                 n_nodes=[32, 32], 
                 forcing=True, 
                 ResNet=True,
                 pot_beta=0.5,
                 ons_min_d=0.1,
                 init_gain=0.1,
                 f_act=F_act,
                 f_linear=False,
                 mode='constant_diagonal',
                 epsilon = 1e-3,
                 n_dim=3,
                 embedding_dim=8):
        super().__init__(delta_t=delta_t, mode=mode, n_dim=n_dim, embedding_dim=embedding_dim, epsilon=epsilon)
        self.onsagernet= OnsagerNet(input_dim=n_dim, 
                         n_nodes=n_nodes, 
                         forcing=forcing, 
                         ResNet=ResNet,
                         pot_beta=pot_beta,
                         ons_min_d=ons_min_d,
                         init_gain=init_gain,
                         f_act=f_act,
                         f_linear=f_linear,
                         embedding_dim=embedding_dim)
    
    def drift(self, x, T):
        # x: [B, 3]

        drift = self.onsagernet(x, T)  # [B, 3]
        return drift   

    def energy(self, x, T):
        # x: [B, 3]
        T = (T - 100) / (3000 - 100)
        T = self.T_embedding(T.unsqueeze(-1))

        energy = self.onsagernet.energy(x, T)
        return energy
    


