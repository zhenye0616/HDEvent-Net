import sys
sys.path.append('./')
from Utils.KG_helper import *
from Utils.data_loader import *
from Model.models import *
from copy import deepcopy
import tqdm as tqdm
import logging
import random
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple


class Runner(object):

	def _inverse_rel(self, rel_idx: int) -> int:
		return rel_idx + self.p.num_rel if rel_idx < self.p.num_rel else rel_idx - self.p.num_rel

	def load_data(self):
		"""
		Reading in raw triples and converts it into a standard format. 

		Parameters
		----------
		self.p.dataset:         Takes in the name of the dataset (FB15k-237)
		
		Returns
		-------
		self.ent2id:            Entity to unique identifier mapping
		self.id2rel:            Inverse mapping of self.ent2id
		self.rel2id:            Relation to unique identifier mapping
		self.num_ent:           Number of entities in the Knowledge graph
		self.num_rel:           Number of relations in the Knowledge graph
		self.embed_dim:         Embedding dimension used
		self.data['train']:     Stores the triples corresponding to training dataset
		self.data['val']:     Stores the triples corresponding to validation dataset
		self.data['test']:      Stores the triples corresponding to test dataset
		self.data_iter:		The dataloader for different data splits

		"""

		ent_set: OrderedSet = OrderedSet()
		base_rel_set: OrderedSet = OrderedSet()
		raw_data = ddict(list)

		for split in ['train', 'test', 'val']:
			with open(f'./Data/{self.p.dataset}/{split}.txt', 'r') as handle:
				for line in handle:
					sub, rel, obj = map(str.lower, line.strip().split('\t'))
					ent_set.add(sub)
					ent_set.add(obj)
					base_rel = rel[:-8] if rel.endswith('_reverse') else rel
					base_rel_set.add(base_rel)
					raw_data[split].append((sub, rel, obj))

		self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}

		self.base_relations = list(base_rel_set)
		self.rel2id = {}
		for rel in self.base_relations:
			self.rel2id[rel] = len(self.rel2id)
		for rel in self.base_relations:
			self.rel2id[f'{rel}_reverse'] = len(self.rel2id)

		self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
		self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

		self.p.num_ent		= len(self.ent2id)
		self.p.num_rel		= len(self.rel2id) // 2
		self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

		self.segment_to_video: Dict[int, int] = {}
		self.segment_to_class: Dict[int, int] = {}
		self.video_to_segments: Dict[int, List[int]] = ddict(list)
		self.video_to_class: Dict[int, int] = {}
		self.class_to_videos: Dict[int, List[int]] = ddict(list)
		self.segment_attrs: Dict[int, Set[int]] = ddict(set)
		self.video_attrs: Dict[int, Set[int]] = ddict(set)
		self.class_attrs: Dict[int, Set[int]] = ddict(set)
		self.attribute_to_segments: Dict[int, Set[int]] = ddict(set)
		self.segment_order: Dict[int, int] = {}

		self.data = ddict(list)
		sr2o = ddict(set)

		for split in ['train', 'test', 'val']:
			for sub, rel, obj in raw_data[split]:
				sub_id, rel_id, obj_id = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
				self.data[split].append((sub_id, rel_id, obj_id))

				if split == 'train':
					sr2o[(sub_id, rel_id)].add(obj_id)
					inv_rel = self._inverse_rel(rel_id)
					sr2o[(obj_id, inv_rel)].add(sub_id)

					rel_name = self.id2rel[rel_id]
					is_reverse = rel_name.endswith('_reverse')
					base_rel = rel_name[:-8] if is_reverse else rel_name
					if base_rel == 'part_of':
						if not is_reverse:
							self.segment_to_video[sub_id] = obj_id
							self.video_to_segments[obj_id].append(sub_id)
						else:
							self.segment_to_video[obj_id] = sub_id
							self.video_to_segments[sub_id].append(obj_id)
					elif base_rel == 'has_attribute':
						if not is_reverse:
							self.segment_attrs[sub_id].add(obj_id)
							self.attribute_to_segments[obj_id].add(sub_id)
						else:
							self.segment_attrs[obj_id].add(sub_id)
							self.attribute_to_segments[sub_id].add(obj_id)
					elif base_rel == 'class_of':
						if not is_reverse:
							self.video_to_class[sub_id] = obj_id
							self.class_to_videos[obj_id].append(sub_id)
						else:
							self.video_to_class[obj_id] = sub_id
							self.class_to_videos[sub_id].append(obj_id)

			self.logger.debug('[load_data] %s triples=%d', split, len(self.data[split]))

		self.data = dict(self.data)

		self.sr2o = {k: list(v) for k, v in sr2o.items()}
		for split in ['test', 'val']:
			for sub, rel, obj in self.data[split]:
				sr2o[(sub, rel)].add(obj)
				sr2o[(obj, self._inverse_rel(rel))].add(sub)

		self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
		self.triples  = ddict(list)

		for (sub, rel), obj in self.sr2o.items():
			self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

		for split in ['test', 'val']:
			for sub, rel, obj in self.data[split]:
				rel_inv = self._inverse_rel(rel)
				self.triples[f'{split}_tail'].append({'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
				self.triples[f'{split}_head'].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

		self.triples = dict(self.triples)
		# finalize auxiliary structures for negative sampling
		for video, segments in self.video_to_segments.items():
			unique_segments = list(dict.fromkeys(segments))
			self.video_to_segments[video] = unique_segments
			attr_acc = self.video_attrs[video]
			for seg in unique_segments:
				attr_acc.update(self.segment_attrs.get(seg, set()))
		for cls, videos in self.class_to_videos.items():
			unique_videos = list(dict.fromkeys(videos))
			self.class_to_videos[cls] = unique_videos
			attr_acc = self.class_attrs[cls]
			for vid in unique_videos:
				attr_acc.update(self.video_attrs.get(vid, set()))
		for seg, vid in self.segment_to_video.items():
			cls = self.video_to_class.get(vid)
			if cls is not None:
				self.segment_to_class[seg] = cls
		for seg in self.segment_to_video.keys():
			name = self.id2ent.get(seg, "")
			token = name.split(':')[-1]
			if '-' in token:
				token = token.split('-')[0]
			digits = ''.join(ch for ch in token if ch.isdigit())
			if digits:
				self.segment_order[seg] = int(digits)

		self.logger.debug('[load_data] train pairs=%d val_pairs=%d test_pairs=%d',
			len(self.triples.get('train', [])),
			len(self.triples.get('val_tail', [])),
			len(self.triples.get('test_tail', []))
		)

		self._init_type_masks()

		def get_data_loader(dataset_class, split, batch_size, shuffle=True):
			return  DataLoader(
					dataset_class(self.triples[split], self.p),
					batch_size      = batch_size,
					shuffle         = shuffle,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = dataset_class.collate_fn
				)

		self.data_iter = {
			'train':    	get_data_loader(TrainDataset, 'train', 	    self.p.batch_size),
			'val_head':   get_data_loader(TestDataset,  'val_head', self.p.batch_size),
			'val_tail':   get_data_loader(TestDataset,  'val_tail', self.p.batch_size),
			'test_head':   	get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
			'test_tail':   	get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
		}

		self.edge_index, self.edge_type = self.construct_adj()
		self.logger.debug(
			'[load_data] num_ent=%d num_rel=%d total_relations=%d',
			self.p.num_ent,
			self.p.num_rel,
			len(self.rel2id)
		)

	def construct_adj(self):
		"""
		Constructor of the runner class

		Parameters
		----------
		
		Returns
		-------
		Constructs the adjacency matrix for GCN
		
		"""
		edge_index, edge_type = [], []

		for sub, rel, obj in self.data['train']:
			edge_index.append((sub, obj))
			edge_type.append(rel)
			edge_index.append((obj, sub))
			edge_type.append(self._inverse_rel(rel))

		edge_index	= torch.LongTensor(edge_index).to(self.device).t()
		edge_type	= torch.LongTensor(edge_type). to(self.device)
		return edge_index, edge_type

	def __init__(self, params):
		"""
		Constructor of the runner class

		Parameters
		----------
		params:         List of hyper-parameters of the model
		
		Returns
		-------
		Creates computational graph and optimizer
		
		"""
		self.p			= params
		self.p.debug		= getattr(self.p, 'debug', False)
		self.logger		= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)
		if self.p.debug:
			self.logger.setLevel(logging.DEBUG)

		self.logger.info(vars(self.p))
		pprint(vars(self.p))

		
		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')
		
		# self.device = torch.device('cpu')

		self.load_data()
		self.model        = self.add_model(self.p.model, self.p.score_func)
		self.optimizer    = self.add_optimizer(self.model.parameters())
		self.neg_samples = getattr(self.p, 'neg_samples', 32)
		self.neg_local_ratio = getattr(self.p, 'neg_local_ratio', 0.8)
		self.neg_inbatch_k = getattr(self.p, 'neg_inbatch_k', 4)
		self.neg_cache_size = getattr(self.p, 'neg_cache_size', 32)
		self.neg_cache_use = getattr(self.p, 'neg_cache_use', 8)
		self.neg_time_window = getattr(self.p, 'neg_time_window', 3)
		self.neg_cache: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
		self._grad_names: Optional[List[str]] = None


	def add_model(self, model, score_func):
		"""
		Creates the computational graph

		Parameters
		----------
		model_name:     Contains the model name to be created
		
		Returns
		-------
		Creates the computational graph for model and initializes it
		
		"""
		model_name = '{}_{}'.format(model, score_func)

		if	model_name.lower()	== 'compgcn_transe':
			model = CompGCN_TransE(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower()	== 'compgcn_distmult':
			model = CompGCN_DistMult(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower()	== 'compgcn_conve':
			model = CompGCN_ConvE(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower() == 'graphd_distmult':
			model = GrapHD_DistMult(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower() == 'graphd_transe':
			model = GrapHD_TransE(self.edge_index, self.edge_type, params=self.p)
		else: raise NotImplementedError

		model.to(self.device)
		return model

	def add_optimizer(self, parameters):
		"""
		Creates an optimizer for training the parameters

		Parameters
		----------
		parameters:         The parameters of the model
		
		Returns
		-------
		Returns an optimizer for learning the parameters of the model
		
		"""
		return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)

	def read_batch(self, batch, split):
		"""
		Function to read a batch of data and move the tensors in batch to CPU/GPU

		Parameters
		----------
		batch: 		the batch to process
		split: (string) If split == 'train', 'val' or 'test' split

		
		Returns
		-------
		Head, Relation, Tails, labels
		"""
		if split == 'train':
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label
		else:
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label

	def save_model(self, save_path):
		"""
		Function to save a model. It saves the model parameters, best validation scores,
		best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

		Parameters
		----------
		save_path: path where the model is saved
		
		Returns
		-------
		"""
		state = {
			'state_dict'	: self.model.state_dict(),
			'best_val'	: self.best_val,
			'best_epoch'	: self.best_epoch,
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.p)
		}
		torch.save(state, save_path)

	def load_model(self, load_path):
		"""
		Function to load a saved model

		Parameters
		----------
		load_path: path to the saved model
		
		Returns
		-------
		"""
		state			= torch.load(load_path)
		state_dict		= state['state_dict']
		self.best_val		= state['best_val']
		self.best_val_mrr	= self.best_val['mrr'] 

		self.model.load_state_dict(state_dict)
		self.optimizer.load_state_dict(state['optimizer'])

	def evaluate(self, split, epoch):
		"""
		Function to evaluate the model on validation or test set

		Parameters
		----------
		split: (string) If split == 'val' then evaluate on the validation set, else the test set
		epoch: (int) Current epoch count
		
		Returns
		-------
		resutls:			The evaluation results containing the following:
			results['mr']:         	Average of ranks_left and ranks_right
			results['mrr']:         Mean Reciprocal Rank
			results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

		"""
		left_results  = self.predict(split=split, mode='tail_batch')
		right_results = self.predict(split=split, mode='head_batch')
		results       = get_combined_results(left_results, right_results)
		
		self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))
		self.logger.info('=========================')
		self.logger.info('[Epoch {} {}]: @10: {:.5}, @3: {:.5}, @1:{:.5}'.format(epoch, split, results['hits@10'], results['hits@3'], results['hits@1']))
		self.logger.info('=========================')
		return results

	def _init_type_masks(self):
		type_buckets = {
			'segment': [],
			'video': [],
			'class': [],
			'attribute': [],
		}
		for ent, idx in self.ent2id.items():
			if ent.startswith('seg:'):
				type_buckets['segment'].append(idx)
			elif ent.startswith('video:'):
				type_buckets['video'].append(idx)
			elif ent.startswith('class:'):
				type_buckets['class'].append(idx)
			elif ent.startswith('attribute:'):
				type_buckets['attribute'].append(idx)

		self._type_index_tensors = {}
		for key, vals in type_buckets.items():
			if vals:
				self._type_index_tensors[key] = torch.tensor(vals, dtype=torch.long, device=self.device)
			else:
				self._type_index_tensors[key] = torch.empty(0, dtype=torch.long, device=self.device)

		total_rels = len(self.rel2id)
		mask_tensor = torch.ones(total_rels, self.p.num_ent, dtype=torch.float32, device=self.device)

		for rel_name, rel_id in self.rel2id.items():
			if rel_name.endswith('_reverse'):
				base = rel_name[:-8]
				is_reverse = True
			else:
				base = rel_name
				is_reverse = False

			allowed_indices = None
			if base == 'class_of':
				allowed_indices = self._type_index_tensors['video'] if is_reverse else self._type_index_tensors['class']
			elif base == 'part_of':
				allowed_indices = self._type_index_tensors['segment'] if is_reverse else self._type_index_tensors['video']
			elif base == 'has_attribute':
				allowed_indices = self._type_index_tensors['segment'] if is_reverse else self._type_index_tensors['attribute']

			if allowed_indices is not None and allowed_indices.numel() > 0:
				mask_tensor[rel_id].zero_()
				mask_tensor[rel_id].scatter_(0, allowed_indices, 1.0)

		self.tail_type_mask = mask_tensor
		self.type_id_lists = {k: tensor.tolist() for k, tensor in self._type_index_tensors.items()}
		all_classes = self.type_id_lists.get('class', [])
		anomaly_ids = [cid for cid in all_classes if 'normal' not in self.id2ent.get(cid, '')]
		normal_ids = [cid for cid in all_classes if 'normal' in self.id2ent.get(cid, '')]
		self.class_groups: Dict[int, List[int]] = {}
		for cid in all_classes:
			if cid in anomaly_ids and len(anomaly_ids) > 1:
				group = [x for x in anomaly_ids if x != cid]
			elif cid in normal_ids and len(normal_ids) > 1:
				group = [x for x in normal_ids if x != cid]
			else:
				group = [x for x in all_classes if x != cid]
			self.class_groups[cid] = group
		self.anomaly_classes = anomaly_ids
		self.normal_classes = normal_ids
		self.all_class_ids = all_classes

	def _get_tail_mask(self, rel_ids: torch.Tensor) -> torch.Tensor:
		return self.tail_type_mask.index_select(0, rel_ids)

	def _build_train_mask(self, sub: torch.Tensor, rel: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
		batch_size = rel.size(0)
		mask = torch.zeros_like(label)
		all_pos_indices: List[List[int]] = []
		for i in range(batch_size):
			pos = (label[i] > 0).nonzero(as_tuple=False).flatten().tolist()
			if not pos:
				pos = []
			all_pos_indices.append(pos)
		for i in range(batch_size):
			head_id = sub[i].item()
			rel_id = rel[i].item()
			pos_list = all_pos_indices[i]
			selected = set(pos_list)
			negatives = self._sample_negatives(head_id, rel_id, pos_list, rel, all_pos_indices, i)
			selected.update(negatives)
			if not selected:
				continue
			mask[i, list(selected)] = 1.0
		return mask

	def _sample_negatives(
		self,
		head_id: int,
		rel_id: int,
		pos_list: List[int],
		rel_tensor: torch.Tensor,
		all_pos_indices: List[List[int]],
		index: int,
	) -> List[int]:
		if self.neg_samples <= 0:
			return []
		pos_set = set(pos_list)
		negatives: List[int] = []
		exclude = set(pos_list)

		rel_name = self.id2rel[rel_id]
		is_reverse = rel_name.endswith('_reverse')
		base_rel = rel_name[:-8] if is_reverse else rel_name

		local_candidates = self._collect_local_candidates(base_rel, is_reverse, head_id, pos_set)
		local_target = int(self.neg_samples * self.neg_local_ratio)
		local_selected = self._sample_from_pool(local_candidates, local_target, exclude)
		negatives.extend(local_selected)
		exclude.update(local_selected)

		# cached negatives
		cache_key = (head_id, rel_id)
		if self.neg_cache_use > 0:
			cache = self.neg_cache.get(cache_key, [])
			for tail_id, _ in cache:
				if tail_id in exclude:
					continue
				negatives.append(tail_id)
				exclude.add(tail_id)
				if len(negatives) >= self.neg_samples:
					break

		# in-batch negatives
		if self.neg_inbatch_k > 0:
			inbatch_candidates: List[int] = []
			rel_val = rel_tensor[index].item()
			for j in range(rel_tensor.size(0)):
				if j == index or rel_tensor[j].item() != rel_val:
					continue
				for cand in all_pos_indices[j]:
					if cand not in exclude:
						inbatch_candidates.append(cand)
			random.shuffle(inbatch_candidates)
			for cand in inbatch_candidates[: self.neg_inbatch_k]:
				if cand in exclude:
					continue
				negatives.append(cand)
				exclude.add(cand)
				if len(negatives) >= self.neg_samples:
					break

		target_type = 'attribute'
		if base_rel == 'class_of':
			target_type = 'video' if is_reverse else 'class'
		elif base_rel == 'part_of':
			target_type = 'segment' if is_reverse else 'video'
		elif base_rel == 'has_attribute':
			target_type = 'segment' if is_reverse else 'attribute'

		global_needed = max(self.neg_samples - len(negatives), 0)
		if global_needed > 0:
			global_sample = self._sample_global_candidates(target_type, exclude, global_needed)
			negatives.extend(global_sample)

		if len(negatives) > self.neg_samples:
			negatives = negatives[: self.neg_samples]
		return negatives

	def _collect_local_candidates(
		self,
		base_rel: str,
		is_reverse: bool,
		head_id: int,
		pos_set: Set[int],
	) -> Set[int]:
		candidates: Set[int] = set()
		if base_rel == 'has_attribute':
			if not is_reverse:
				segment = head_id
				video = self.segment_to_video.get(segment)
				if video is not None:
					segments = self.video_to_segments.get(video, [])
					order = self.segment_order.get(segment)
					for seg in segments:
						if self.segment_attrs.get(seg):
							candidates.update(self.segment_attrs[seg])
						if order is not None:
							seg_order = self.segment_order.get(seg)
							if seg_order is not None and abs(seg_order - order) <= self.neg_time_window:
								candidates.update(self.segment_attrs.get(seg, set()))
					cls = self.video_to_class.get(video)
					if cls is not None:
						candidates.update(self.class_attrs.get(cls, set()))
				cls = self.segment_to_class.get(segment)
				if cls is not None:
					candidates.update(self.class_attrs.get(cls, set()))
			else:
				attribute = head_id
				segments = self.attribute_to_segments.get(attribute, set())
				for seg in segments:
					video = self.segment_to_video.get(seg)
					if video is not None:
						candidates.update(self.video_to_segments.get(video, []))
						cls = self.video_to_class.get(video)
						if cls is not None:
							for vid in self.class_to_videos.get(cls, []):
								candidates.update(self.video_to_segments.get(vid, []))
					order = self.segment_order.get(seg)
					if order is not None:
						video_segments = self.video_to_segments.get(self.segment_to_video.get(seg, -1), [])
						for other in video_segments:
							seg_order = self.segment_order.get(other)
							if seg_order is not None and abs(seg_order - order) <= self.neg_time_window:
								candidates.add(other)
		elif base_rel == 'class_of':
			if not is_reverse:
				segment = head_id
				cls = self.segment_to_class.get(segment)
				if cls is not None:
					group = self.class_groups.get(cls, [cid for cid in self.all_class_ids if cid != cls])
					candidates.update(group)
			else:
				cls = head_id
				candidates.update(self.class_to_videos.get(cls, []))
		elif base_rel == 'part_of':
			if not is_reverse:
				segment = head_id
				video = self.segment_to_video.get(segment)
				if video is not None:
					cls = self.video_to_class.get(video)
					if cls is not None:
						candidates.update(self.class_to_videos.get(cls, []))
					if video in candidates:
						candidates.discard(video)
			else:
				video = head_id
				candidates.update(self.video_to_segments.get(video, []))
		candidates.difference_update(pos_set)
		return candidates

	def _sample_from_pool(self, candidates: Set[int], count: int, exclude: Set[int]) -> List[int]:
		if count <= 0 or not candidates:
			return []
		available = list(candidates - exclude)
		if not available:
			return []
		if len(available) <= count:
			return available
		return random.sample(available, count)

	def _sample_global_candidates(self, type_key: str, exclude: Set[int], count: int) -> List[int]:
		if count <= 0:
			return []
		population = self.type_id_lists.get(type_key, [])
		if not population:
			return []
		result: List[int] = []
		attempts = 0
		max_attempts = count * 10 + 1
		while len(result) < count and attempts < max_attempts:
			cand = random.choice(population)
			attempts += 1
			if cand in exclude:
				continue
			result.append(cand)
			exclude.add(cand)
		return result

	def _update_neg_cache(
		self,
		sub: torch.Tensor,
		rel: torch.Tensor,
		label: torch.Tensor,
		scores: torch.Tensor,
		mask: torch.Tensor,
	) -> None:
		for i in range(sub.size(0)):
			head_id = sub[i].item()
			rel_id = rel[i].item()
			key = (head_id, rel_id)
			pos_idx = (label[i] > 0).nonzero(as_tuple=False).flatten().tolist()
			if not pos_idx:
				continue
			candidate_idx = ((mask[i] > 0) & (label[i] == 0)).nonzero(as_tuple=False).flatten().tolist()
			if not candidate_idx:
				continue
			score_vals = scores[i, candidate_idx]
			pairs = list(zip(candidate_idx, score_vals.tolist()))
			cache = self.neg_cache.get(key, [])
			cache.extend(pairs)
			cache.sort(key=lambda x: x[1], reverse=True)
			new_cache: List[Tuple[int, float]] = []
			seen: Set[int] = set()
			for tail_id, sc in cache:
				if tail_id in pos_idx or tail_id in seen:
					continue
				new_cache.append((tail_id, sc))
				seen.add(tail_id)
				if len(new_cache) >= self.neg_cache_size:
					break
			self.neg_cache[key] = new_cache

	def predict(self, split='val', mode='tail_batch'):
		"""
		Function to run model evaluation for a given mode

		Parameters
		----------
		split: (string) 	If split == 'val' then evaluate on the validation set, else the test set
		mode: (string):		Can be 'head_batch' or 'tail_batch'
		
		Returns
		-------
		resutls:			The evaluation results containing the following:
			results['mr']:         	Average of ranks_left and ranks_right
			results['mrr']:         Mean Reciprocal Rank
			results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

		"""
		self.model.eval()

		with torch.no_grad():
			results = {}
			train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

			for step, batch in enumerate(train_iter):
				sub, rel, obj, label	= self.read_batch(batch, split)
				pred		= self.model.forward(sub, rel)
				mask		= self._get_tail_mask(rel)
				label		= label * mask
				pred		= pred.masked_fill(mask == 0, -1e6)
				b_range			= torch.arange(pred.size()[0], device=self.device)
				target_pred		= pred[b_range, obj]
				pred 			= torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)
				pred[b_range, obj] 	= target_pred
				ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

				ranks 			= ranks.float()
				results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
				results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
				results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
				for k in range(10):
					results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

				if step % 100 == 0:
					self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

		return results


	def run_epoch(self, epoch, val_mrr = 0):
		"""
		Function to run one epoch of training

		Parameters
		----------
		epoch: current epoch count
		
		Returns
		-------
		loss: The loss value after the completion of one epoch
		"""
		self.model.train()
		losses = []
		train_iter = iter(self.data_iter['train'])
		grad_accum = {}

		for step, batch in enumerate(train_iter):
			#NOTE: Hanning measure training time
			torch.cuda.synchronize()
			start_epoch = time.time()

			self.optimizer.zero_grad()
			sub, rel, obj, label = self.read_batch(batch, 'train')
			scores	= self.model.forward(sub, rel)
			mask    = self._build_train_mask(sub, rel, label)
			masked_label = label * mask
			loss	= self.model.loss(scores, label, mask)
			loss.backward()

			if self.logger.isEnabledFor(logging.DEBUG):
				if self._grad_names is None:
					names: List[str] = []
					for name, param in self.model.named_parameters():
						if param.grad is None:
							continue
						names.append(name)
						if len(names) == 6:
							break
					self._grad_names = names
				if grad_accum is None or not grad_accum:
					grad_accum = {name: [0.0, 0] for name in (self._grad_names or [])}
				for name, param in self.model.named_parameters():
					if self._grad_names and name not in self._grad_names:
						continue
					if param.grad is None:
						continue
					stats = grad_accum.setdefault(name, [0.0, 0])
					stats[0] += param.grad.norm().item()
					stats[1] += 1

			self.optimizer.step()

			if self.logger.isEnabledFor(logging.DEBUG):
				mean_pos = masked_label.sum(dim=1).mean().item()
				self.logger.debug(
					'[train] epoch=%d step=%d loss=%.6f mean_pos=%.3f sub[0]=%d rel[0]=%d',
					epoch,
					step,
					loss.item(),
					mean_pos,
					sub[0].item(),
					rel[0].item()
				)

			torch.cuda.synchronize()
			end_epoch = time.time()
			elapsed = end_epoch - start_epoch
			# print("===============================")
			# print("Epoch training time is: {}".format(elapsed))
			# print("===============================")

			losses.append(loss.item())
			self._update_neg_cache(sub, rel, label, scores, mask)

			if step % 100 == 0:
				self.logger.info('[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses), self.best_val_mrr, self.p.name))

		loss = np.mean(losses)
		if self.logger.isEnabledFor(logging.DEBUG) and grad_accum:
			summary = []
			for name, (total, count) in grad_accum.items():
				if count == 0:
					continue
				summary.append(f"{name}:{(total/count):.6e}")
				if len(summary) >= 6:
					break
			self.logger.debug("[grad] epoch=%d %s", epoch, ", ".join(summary))
		self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
		return loss


	def fit(self):
		"""
		Function to run training and evaluation of model

		Parameters
		----------
		
		Returns
		-------
		"""
		self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
		save_dir = os.path.join('./checkpoints', self.p.name)
		os.makedirs(save_dir, exist_ok=True)
		save_path = os.path.join(save_dir, 'model.pt')

		if self.p.restore:
			self.load_model(save_path)
			self.logger.info('Successfully Loaded previous model')

		kill_cnt = 0
		for epoch in range(self.p.max_epochs):
			train_loss  = self.run_epoch(epoch, val_mrr)
			val_results = self.evaluate('val', epoch)

			if val_results['mrr'] > self.best_val_mrr:
				self.best_val	   = val_results
				self.best_val_mrr  = val_results['mrr']
				self.best_epoch	   = epoch
				self.save_model(save_path)
				kill_cnt = 0
			else:
				kill_cnt += 1
				if kill_cnt % 10 == 0 and self.p.gamma > 5:
					self.p.gamma -= 5 
					self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
				if kill_cnt > 25: 
					self.logger.info("Early Stopping!!")
					#TODO: Cancel eraly stop for WN18RR
					if self.p.dataset == 'FB15k-237':
						break

			#self.logger.info('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(epoch, train_loss, self.best_val_mrr))

		self.logger.info('Loading best model, Evaluating on Test data')
		self.load_model(save_path)
		#NOTE: Try to flip the model
		# print("The bit flip begin")
		# for noise_prob in np.arange(0.01, 0.02, 0.001):
		# 	print("=====================================")
		# 	print('The noise_prob is {}'.format(noise_prob))
		# 	for name, param in tqdm(self.model.named_parameters()):
		# 		if name == 'init_embed.weight' or name == 'init_rel.weight':
		# 			print("{} jump".format(name))
		# 			continue
		# 		print("Flip {}".format(name))
		# 		random_bit_flip_by_prob(param, noise_prob)

		test_results = self.evaluate('test', epoch)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('-name',		default='testrun',					help='Set run name for saving/restoring models')
	parser.add_argument('-data',		dest='dataset',         default='FB15k-237',            help='Dataset to use, default: FB15k-237')
	parser.add_argument('-model',		dest='model',		default='compgcn',		help='Model Name')
	parser.add_argument('-score_func',	dest='score_func',	default='conve',		help='Score Function for Link prediction')
	parser.add_argument('-opn',             dest='opn',             default='corr',                 help='Composition Operation to be used in CompGCN')

	parser.add_argument('-batch',           dest='batch_size',      default=128,    type=int,       help='Batch size')
	parser.add_argument('-gamma',		type=float,             default=5.0,			help='Margin')
	parser.add_argument('-gpu',		type=str,               default='0',			help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
	parser.add_argument('-epoch',		dest='max_epochs', 	type=int,       default=500,  	help='Number of epochs')
	parser.add_argument('-l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
	parser.add_argument('-lr',		type=float,             default=0.001,			help='Starting Learning Rate')
	parser.add_argument('-lbl_smooth',      dest='lbl_smooth',	type=float,     default=0.0,	help='Label Smoothing')
	parser.add_argument('-num_workers',	type=int,               default=10,                     help='Number of processes to construct batches')
	parser.add_argument('-seed',            dest='seed',            default=41504,  type=int,     	help='Seed for randomization')

	parser.add_argument('-restore',         dest='restore',         action='store_true',            help='Restore from the previously saved model')
	parser.add_argument('-bias',            dest='bias',            action='store_true',            help='Whether to use bias in the model')

	parser.add_argument('-num_bases',	dest='num_bases', 	default=-1,   	type=int, 	help='Number of basis relation vectors to use')
	parser.add_argument('-init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
	parser.add_argument('-gcn_dim',	  	dest='gcn_dim', 	default=200,   	type=int, 	help='Number of hidden units in GCN')
	parser.add_argument('-embed_dim',	dest='embed_dim', 	default=None,   type=int, 	help='Embedding dimension to give as input to score function')
	parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=1,   	type=int, 	help='Number of GCN Layers to use')
	parser.add_argument('-gcn_drop',	dest='dropout', 	default=0.1,  	type=float,	help='Dropout to use in GCN Layer')
parser.add_argument('-hid_drop',  	dest='hid_drop', 	default=0.3,  	type=float,	help='Dropout after GCN')

# ConvE specific hyperparameters
parser.add_argument('-hid_drop2',  	dest='hid_drop2', 	default=0.3,  	type=float,	help='ConvE: Hidden dropout')
parser.add_argument('-feat_drop', 	dest='feat_drop', 	default=0.3,  	type=float,	help='ConvE: Feature Dropout')
parser.add_argument('-k_w',	  	dest='k_w', 		default=10,   	type=int, 	help='ConvE: k_w')
parser.add_argument('-k_h',	  	dest='k_h', 		default=20,   	type=int, 	help='ConvE: k_h')
parser.add_argument('-num_filt',  	dest='num_filt', 	default=200,   	type=int, 	help='ConvE: Number of filters in convolution')
parser.add_argument('-ker_sz',    	dest='ker_sz', 		default=7,   	type=int, 	help='ConvE: Kernel size to use')
parser.add_argument('--neg-samples', dest='neg_samples', default=32, type=int, help='Number of negative tails to sample per positive.')
parser.add_argument('--neg-local-ratio', dest='neg_local_ratio', default=0.8, type=float, help='Fraction of negatives drawn from local buckets (rest global).')
parser.add_argument('--neg-inbatch-k', dest='neg_inbatch_k', default=4, type=int, help='Number of in-batch hard negatives to include per positive.')
parser.add_argument('--neg-cache-size', dest='neg_cache_size', default=32, type=int, help='FIFO cache size for hard negatives per (head, relation).')
parser.add_argument('--neg-cache-use', dest='neg_cache_use', default=8, type=int, help='Number of cached negatives to reuse each step.')
parser.add_argument('--neg-time-window', dest='neg_time_window', default=3, type=int, help='Temporal window (in segments) for local negative sampling.')

parser.add_argument('-logdir',          dest='log_dir',         default='./log/',               help='Log directory')
parser.add_argument('-config',          dest='config_dir',      default='./config/',            help='Config directory')
args = parser.parse_args()

if not args.restore: args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

set_gpu(args.gpu)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

model = Runner(args)
model.fit()


"""python KG.py \
  -name ucf_graphhd_transe \
  -data UCF_Crime \
  -model graphd \
  -score_func transe \
  -epoch 100 \
  -batch 128 \
  -lr 1e-3 \
  -gpu 1"""
