from openselfsup.datasets.pipelines.transform_utils import decompose_collated_batch
import torch
import torch.nn as nn
import torch.nn.functional as F
from openselfsup.utils import print_log
from . import builder
from .registry import MODELS
import numpy as np

@MODELS.register_module
class ConCL(nn.Module):
    def __init__(self, backbone, neck=None, pretrained=None, 
                 queue_len=65536, feat_dim=128, momentum=0.999, head=None,
                 num_concepts=4, cluster_indice=3,
                 concept_weight=1.0,
                 warmup=True,
                 **kwargs):
        super(ConCL, self).__init__()

        self.encoder_q = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.encoder_k = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.backbone = self.encoder_q[0]
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

        self.queue_len = queue_len
        self.momentum = momentum

        # create the queue
        self.register_buffer("queue", torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        ######  conncl-related ####
        self.num_concepts = num_concepts
        self.cluster_indice = cluster_indice - 5
        self.concept_weight = concept_weight
        self.local_start = False if warmup else True
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.register_buffer("concept_queue", torch.randn(feat_dim, queue_len))
        self.concept_queue = nn.functional.normalize(self.concept_queue, dim=0)
        self.register_buffer("concept_queue_ptr", torch.zeros(1, dtype=torch.long))
        #################

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.encoder_q[0].init_weights(pretrained=pretrained)
        self.encoder_q[1].init_weights(init_linear='kaiming')
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_concept(self, keys):
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather_v2(keys)
        
        batch_size = keys.shape[0]
        ptr = int(self.concept_queue_ptr)
        # assert self.queue_len % batch_size == 0  # for simplicity
        if (ptr + batch_size) < self.queue_len:
            # replace the keys at ptr (dequeue and enqueue)
            self.concept_queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
            ptr = ptr + batch_size  # move pointer
        else:
            self.concept_queue[:, ptr:] = \
                keys[:(self.queue_len - ptr)].transpose(0, 1)
            self.concept_queue[:, : (batch_size + ptr - self.queue_len)]\
                 = keys[(self.queue_len - ptr):].transpose(0,1)
            ptr = batch_size + ptr - self.queue_len

        self.concept_queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """Batch shuffle, for making use of BatchNorm.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Undo batch shuffle.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward_train(self, img, transf, **kwargs):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        im_q = img[:, 0, ...].contiguous()
        im_k = img[:, 1, ...].contiguous()
        im_r = img[:, 2, ...].contiguous()

        pre_gap_q = self.encoder_q[0](im_q)[-1]
        q = self.encoder_q[1]([self.avgpool(pre_gap_q)])[0]
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            im_k, idx_unshuffle_k = self._batch_shuffle_ddp(im_k)
            pre_gap_k = self.encoder_k[0](im_k)[-1]
            k = self.encoder_k[1]([self.avgpool(pre_gap_k)])[0]
            k = F.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle_k)

            # get the feature map of reference view
            if self.local_start:
                r = self.encoder_k[0](im_r)[self.cluster_indice]
        
        ############ Instance Part ##############
        instance_l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        instance_l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        instance_loss = self.head(instance_l_pos, instance_l_neg)['loss']
        self._dequeue_and_enqueue(k)

        ############ Concept Part ###############
        if self.local_start:
            with torch.no_grad():
                # Step 1. Clustering over reference image
                # (B, H, W, C) <-- (B, C, H, W)
                r = r.permute(0, 2, 3, 1)
                B,H,W,C = r.size()
                assign_r, _ = batch_cosine_KMeans(r.view(B, H*W, C), self.num_concepts)
                # (B, H, W)
                assign_r = assign_r.view(B, H, W).float()
                transf_q, transf_k = decompose_collated_batch(transf)
                assign_map_k = self.match_views_avg_pool(assign_r, transf_k)
                assign_map_q = self.match_views_avg_pool(assign_r, transf_q)

            q_protos, k_protos = self.get_paired_prototypes(
                                    pre_gap_q, pre_gap_k.detach(), 
                                    assign_map_q, assign_map_k)

            q_protos = self.encoder_q[1]([q_protos])[0]
            q_protos = F.normalize(q_protos, dim=1)

            with torch.no_grad():
                k_protos = self.encoder_k[1]([k_protos])[0]
                k_protos = F.normalize(k_protos, dim=1)

                # Padding needed.
                # have to pad the enqueued prototypes to ensure
                # the functionality of concat_all_gather
                batch_size = img.size(0)
                if self.num_concepts > 8:
                    num_sampled_protos = 4 * batch_size
                else:
                    num_sampled_protos = self.num_concepts * batch_size
                    
                if k_protos.size(0) >= num_sampled_protos:
                    sampled_k_protos = k_protos[sort_rand_gpu(k_protos.size(0), num_sampled_protos)]
                else:
                    sampled_k_protos = torch.cat(
                            [k_protos, k_protos[sort_rand_gpu(k_protos.size(0),
                                                                num_sampled_protos - k_protos.size(0))]], dim=0)
                
            concept_l_pos = torch.einsum('nc,nc->n', [q_protos, k_protos]).unsqueeze(-1)
            concept_l_neg = torch.einsum('nc,ck->nk', [q_protos, self.concept_queue.clone().detach()])
            concept_loss = self.head(concept_l_pos, concept_l_neg)['loss']
            self._dequeue_and_enqueue_concept(sampled_k_protos)
            total_loss = self.concept_weight * concept_loss + (1-self.concept_weight) * instance_loss

        else:
            concept_loss = torch.zeros(1).cuda()
            total_loss = instance_loss

        losses = dict(
            loss = total_loss,
            instance_l = instance_loss,
            concept_l = concept_loss)
        return losses

    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))

    def get_paired_prototypes(self, q, k, q_maps, k_maps):
        # q, k: (B, C, H, W), e.g. (B, 256, 14, 14)
        # q_maps, k_maps: (B, H, W), e.g. (B, 14, 14)
        unique_q = torch.unique(q_maps.view(q_maps.size(0), -1), dim=-1)
        unique_k = torch.unique(k_maps.view(k_maps.size(0), -1), dim=-1)

        q_shared_protos, k_shared_protos = [], []
        for ix, (q_map, k_map) in enumerate(zip(q_maps, k_maps)):
            for cluster_id in range(self.num_concepts):
                if (cluster_id in unique_q[ix]) and (cluster_id in unique_k[ix]):
                    q_mask = q_map == cluster_id
                    q_feat = q[ix] * q_mask.unsqueeze(0)
                    q_area = torch.sum(q_mask)
                    q_proto = torch.sum(q_feat, dim=(-1,-2)) / q_area

                    k_mask = k_map == cluster_id
                    k_feat = k[ix] * k_mask.unsqueeze(0)
                    k_area = torch.sum(k_mask)
                    k_proto = torch.sum(k_feat, dim=(-1,-2)) / k_area

                    q_shared_protos.append(q_proto.unsqueeze(0))
                    k_shared_protos.append(k_proto.unsqueeze(0))

        
        q_shared_protos = torch.cat(q_shared_protos, dim=0)
        k_shared_protos = torch.cat(k_shared_protos, dim=0)
        return q_shared_protos, k_shared_protos

    def match_views_avg_pool(self, reference_map, transformation, target_size=(7,7)):
        # reference_map: (B,H,W)
        # transformations: (B, 5)
        B, H, W = reference_map.shape
        one_hot_map = torch.zeros((B,self.num_concepts, H, W)).cuda().scatter_(1, reference_map.long().unsqueeze(1), 1)
        top_start = H * transformation[:, 0]
        left_start = W * transformation[:, 1]
        bottom_end = H * transformation[:, 2]
        right_end = W * transformation[:, 3]
        flip = transformation[:, 4] > 0
        assignment_map = torch.zeros((B,*target_size)).cuda()
        for ix in range(B):
            assignment_map[ix] = F.adaptive_avg_pool2d(
                    one_hot_map[ix, :, int(torch.round(top_start[ix])):int(torch.round(bottom_end[ix])),
                    int(torch.round(left_start[ix])):int(torch.round(right_end[ix]))].unsqueeze(0),
                    output_size=target_size).argmax(dim=1).squeeze()
            if flip[ix]:
                assignment_map[ix] = assignment_map[ix].flip(-1) 

        return assignment_map.contiguous()

@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.zeros_like(tensor)
        for _ in range(torch.distributed.get_world_size())]

    torch.distributed.all_gather(tensors_gather, tensor, )#async_rp=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

# utils
@torch.no_grad()
def concat_all_gather_v2(tensor):
    """Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    shape_tensor = torch.IntTensor(np.array(tensor.shape)).cuda()
    all_shape = [torch.IntTensor(np.array(tensor.shape)).cuda() for i in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(all_shape, shape_tensor)
    longest = max([ shape[0] for shape in all_shape ])
   
    origin_shape = tensor.shape
    if tensor.shape[0] != longest :
        pad = torch.zeros((longest - tensor.shape[0],*list(tensor.shape[1:])), device=tensor.device)
        tensor = torch.cat([tensor, pad])
        # print('Generating features with all zeros, which might be unwanted')
    tensors_gather = [
        torch.zeros_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    
    torch.distributed.all_gather(tensors_gather, tensor, )#async_rp=False)

    if origin_shape[0] != longest :
        rank = torch.distributed.get_rank()
        tensors_gather[rank] = tensors_gather[rank].resize_(origin_shape)
    
    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def batch_cosine_KMeans(X: torch.Tensor,num_clusters=2,max_iter=10):
    """
    X: (torch.tensor) matrix
    return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    # transfer to device
    # X: (B, N, C), 
    X = X.clone().detach()
    X = F.normalize(X, dim=2)

    centroids = []
    for i in range(X.size(0)):
        centroids.append(X[i, torch.randint(X.size(1), (num_clusters,)), :].unsqueeze(0))
    # B, K, C
    centroids = torch.cat(centroids, dim=0)
    # (B,C,K)
    centroids = centroids.permute(0, 2, 1)
    for _ in range(max_iter):
        centroids = F.normalize(centroids, dim=1)
        dis = 1 - torch.bmm(X, centroids)# (B,N,K) <-- (B,N,C) x (B,C,K)
        assignments = torch.argmin(dis, dim=2)# (B,N)
        for cluster_id in range(num_clusters):
            selected = (assignments == cluster_id).float().unsqueeze(dim=1) # (B,1,N)
            anomaly_selected = (1 - selected.mean(dim=2)).repeat([1,X.size(-1)]) #(B,C)  all zeros, anomaly_selected will be one.
            #(B,C)<--- (B,1,C) <--- (B,1,N) x (B,N,C)
            selected_mean = torch.bmm(selected, X).squeeze(dim=1)/(selected.sum(dim=2).repeat([1,X.size(-1)]))
            selected_mean = torch.where((anomaly_selected==1), X[:, torch.randint(0, X.size(1),(1,))].squeeze(dim=1), selected_mean)
            # print(torch.norm(selected_mean, dim=1))
            centroids[:, :, cluster_id] = selected_mean 
    # (B, K, C) <--- (B, C, K)
    centroids = centroids.permute(0,2,1)
    #      (B,N),          (B, K, C)
    return assignments, centroids

@torch.no_grad()
def sort_rand_gpu(pop_size, num_samples):
    """Generate a random torch.Tensor (GPU) and sort it to generate indices."""
    if pop_size < num_samples:
        print("Hi, you have inadequate population size, but I still give you my answer.")
        print(f"return number: {pop_size}")
    return torch.argsort(torch.rand(pop_size, device='cuda'))[:num_samples]