from helper import *
from model.message_passing import MessagePassing
import numpy as np

#NOTE:Autoencoder for Ali's paper
class AE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
			torch.nn.Linear(512,512),
			torch.nn.Tanh()
		)
    def forward(self,x):
        encoded = self.encoder(x)
        return encoded


class GrapHD(MessagePassing):
	def __init__(self, in_channels, out_channels, num_rels, act=lambda x:x, params=None):
		super(self.__class__, self).__init__()

		self.p 			= params
		self.in_channels	= in_channels
		self.out_channels	= out_channels
		self.num_rels 		= num_rels
		self.act 		= act
		self.device		= None

		self.w_loop		= get_param((in_channels, out_channels))
		# self.w_in		= get_param((in_channels, out_channels))
		# self.w_out		= get_param((in_channels, out_channels))
		# self.w_rel 		= get_param((in_channels, out_channels))
		self.w_in = self.w_loop
		self.w_out = self.w_loop
		self.w_rel = self.w_loop
		self.loop_rel 		= get_param((1, in_channels));

		self.drop		= torch.nn.Dropout(self.p.dropout)
		self.bn			= torch.nn.BatchNorm1d(out_channels)

		#self.AE = AE()
  
		if self.p.bias:
			self.register_parameter('bias', Parameter(torch.zeros(out_channels)))
	
	def forward(self, x, edge_index, edge_type, rel_embed): 
		if self.device is None:
			self.device = edge_index.device

        #NOTE: freeze base HDVs (w_loop, w_in, w_rel)
		
		self.w_loop.requires_grad = False
		self.w_in.requires_grad = False
		self.w_rel.requires_grad = False
		self.w_out.requires_grad = False
		

		rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
		num_edges = edge_index.size(1) // 2
		num_ent   = x.size(0)

		self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
		self.in_type,  self.out_type  = edge_type[:num_edges], 	 edge_type [num_edges:]

		self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
		self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)

		#TODO: why using norm ?
		self.in_norm     = self.compute_norm(self.in_index,  num_ent)
		self.out_norm    = self.compute_norm(self.out_index, num_ent)
		
		#NOTE: For Ali's paper, add autoencoder before encoding into HDVs
		# x = self.AE(x)
  
		in_res		= self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,   rel_embed=rel_embed, edge_norm=self.in_norm, 	mode='in')
		loop_res	= self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, 		mode='loop')
		out_res		= self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,  rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out')
		out		= self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)

		#NOTE: For Ali's paper, add autoencoder before encoding into HDVs
		#out = self.AE(out)

		if self.p.bias: out = out + self.bias
		out = self.bn(out)

		# return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]		# Ignoring the self loop inserted
		return out, torch.matmul(rel_embed, self.w_rel)[:-1]

	#NOTE: binding entity embedding with relation embedding
	def rel_transform(self, ent_embed, rel_embed, weight):
		#NOTE: Apply mask based on Ali's CLOG
		# clog_mask = np.random.random_integers(500,size=50)
  
		if  self.p.opn == 'corr':
			#NOTE: HDC encode 
			ent_embed_HD = torch.matmul(ent_embed,weight)
			ent_embed_HD = self.act(ent_embed_HD)
			rel_embed_HD = torch.matmul(rel_embed,self.w_rel)
			rel_embed_HD = self.act(rel_embed_HD)
			trans_embed  = ccorr(ent_embed_HD, rel_embed_HD)
			# trans_embed = ccorr(ent_embed, rel_embed)
		elif self.p.opn == 'sub':
			#NOTE: HDC encode
			ent_embed_HD = torch.matmul(ent_embed,weight)
			ent_embed_HD = self.act(ent_embed_HD)
			rel_embed_HD = torch.matmul(rel_embed,self.w_rel)
			rel_embed_HD = self.act(rel_embed_HD)
			#NOTE: For Ali's project adding permutattion
			# ent_embed_HD = torch.roll(ent_embed_HD,1)
			# ent_embed_HD[clog_mask] = 1
			trans_embed  = ent_embed_HD - rel_embed_HD
			#NOTE: For Ali's paper, add autoencoder before encoding into HDVs
			# trans_embed = self.AE(trans_embed)
			# trans_embed  = ent_embed - rel_embed
		elif self.p.opn == 'mult':
			#NOTE: HDC encode
			ent_embed_HD = torch.matmul(ent_embed,weight)
			ent_embed_HD = self.act(ent_embed_HD)
			rel_embed_HD = torch.matmul(rel_embed,self.w_rel)
			rel_embed_HD = self.act(rel_embed_HD)
			#NOTE: For Ali's project adding permutattion
			# ent_embed_HD = torch.roll(ent_embed_HD,1)
			# ent_embed_HD[clog_mask] = 1
			trans_embed  = ent_embed_HD * rel_embed_HD
			# trans_embed  = ent_embed * rel_embed
		else: 
			raise NotImplementedError

		return trans_embed

	def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
		weight 	= getattr(self, 'w_{}'.format(mode))
		rel_emb = torch.index_select(rel_embed, 0, edge_type)
		xj_rel  = self.rel_transform(x_j, rel_emb,weight)
		out	= xj_rel
		# out	= torch.mm(xj_rel, weight)	

		return out if edge_norm is None else out * edge_norm.view(-1, 1)

	def update(self, aggr_out):
		return aggr_out

	def compute_norm(self, edge_index, num_ent):
		row, col	= edge_index
		edge_weight 	= torch.ones_like(row).float()
		deg		= scatter_add(edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
		deg_inv		= deg.pow(-0.5)							# D^{-0.5}
		deg_inv[deg_inv	== float('inf')] = 0
		norm		= deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

		return norm

	def __repr__(self):
		return '{}({}, {}, num_rels={})'.format(
			self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)