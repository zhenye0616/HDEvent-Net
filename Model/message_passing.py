import inspect, torch
from torch_scatter import scatter
import numpy as np

def scatter_(name, src, index, dim_size=None):
	"""Aggregates all values from the :attr:`src` tensor at the indices
	specified in the :attr:`index` tensor along the first dimension.
	If multiple indices reference the same location, their contributions
	are aggregated according to :attr:`name` (either :obj:`"add"`,
	:obj:`"mean"` or :obj:`"max"`).

	Args:
		name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
			:obj:`"max"`).
		src (Tensor): The source tensor.
		index (LongTensor): The indices of elements to scatter.
		dim_size (int, optional): Automatically create output tensor with size
			:attr:`dim_size` in the first dimension. If set to :attr:`None`, a
			minimal sized output tensor is returned. (default: :obj:`None`)

	:rtype: :class:`Tensor`
	"""
	if name == 'add': name = 'sum'
	assert name in ['sum', 'mean', 'max']
	out = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce='mean')
	# out = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce='mul') # NOTE: change reduce to mul for Ali's model
	return out[0] if isinstance(out, tuple) else out


class MessagePassing(torch.nn.Module):
    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()
        self.message_args = inspect.getargspec(self.message)[0][1:]	# In the defined message function: get the list of arguments as list of string| For eg. in rgcn this will be ['x_j', 'edge_type', 'edge_norm'] (arguments of message function)
        self.update_args  = inspect.getargspec(self.update)[0][2:]	# Same for update function starting from 3rd argument | first=self, second=out
        
    def propagate(self, aggr, edge_index, **kwargs):
        assert aggr in ['add', 'mean', 'max']
        kwargs['edge_index'] = edge_index
        size = None
        message_args = []
        #NOTE: self.message_args is args' name
        # clog_mask = np.random.random_integers(512,size=32)
        #clog_mask = np.concatenate([clog_mask,clog_mask+1,clog_mask+2,clog_mask+3,clog_mask+4])
        """
        for i in range(4):
            clog_mask = np.concatenate([clog_mask,clog_mask+1])
        """
            
        for arg in self.message_args:
            if arg[-2:] == '_i':					
                tmp  = kwargs[arg[:-2]]				 
                size = tmp.size(0)
                message_args.append(tmp[edge_index[0]])
            elif arg[-2:] == '_j':
                tmp  = kwargs[arg[:-2]]				
                size = tmp.size(0)
                message_args.append(tmp[edge_index[1]])
            # elif arg == 'clog_mask':
            #     message_args.append(clog_mask)
            else:
                message_args.append(kwargs[arg])		
        update_args = [kwargs[arg] for arg in self.update_args]		
        out = self.message(*message_args)
        out = scatter_(aggr, out, edge_index[0], dim_size=size)		
        out = self.update(out, *update_args)
        return out
    
    def message(self, x_j):  # pragma: no cover
        return x_j
    
    def update(self, aggr_out):
        return aggr_out
