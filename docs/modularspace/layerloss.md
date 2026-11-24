# Process

The goal of this code is use the space to represent neuron affinity, dfunctional modulairty, withing a neural network, layer by layer ( there is an addition step to do it at model level) for now we discuss only layer level.

This whokle process is complitely decoupled from the neural network, and it's weights.

The process starts with a model, a mapper finds the interesting layers ( mlp for now, but i'll add support for all types of layers ) for each neuron ( computation unit: CU ) allocates a n-dimesion vector, a sort of embedding and adds a forward hook, for each layer.

now, by looking at the individual layer this is what happens:
1- activation are captured and detached right away ( and pushed into a queue to avoid hindering the forward )
2- CU are "ranked" ( mlp just requires raw activation, cnn, rnn, transformers will require different methods )
3- CU are split in active/inactive
4- a contrastive step is taken between active and inactive


some considerations:

1- torch.quantile is heavy, perhaps top_k is more efficent, but at this point why not using just raw activations? maybe just with a batchnorm layer before processing them. using the real 0 threhold between active and inactive.

2- loss_centers is there only to 

3- weights are not the leyer weights, perhaps the name is confusing, it's just the "magnitude" of the activation. has been added in the attempt to break simmery.

4- this is a brand new method, some of the math is arbitry, put there from intuition rather then formal choices (ex: torch.log, quantile )

5- what i ideally want to reach is a dynamic equilibium from the contant pull and push of positions. a map of common and specialized group of neurons ( computatyion grouop: CG )

6- later the space is clustered, clusters a fed back from the previuous step, via clusters_labels from (t-1).

7- the space tends to collapse for over optimization, destroing internal organization withing groups. 

8- maybe currently the code doesn't implement it correctly, but contrastive steps, ideally should be done as much as possible "per datapoint" not per batch, as if it was sequentially updated after each datapoint. However is deadly inefficednt, so batch processing, but, normalizations, etc should consider this.

9- stats vector, ideally i would like to use it as normalization factor for a bettwer equilibrium, mening high freq activated neurons "push more" then low freq. using something like the inverse, to normalize the push and obtain and evenly space map/atlas/space.

10- euclidin vs cosine space ? at the beginning i used euclidian, with adamw to avoid explosions, however weight decay often collapse structures at the edge.

11- i'm aiming for a lightweight implementation becuase this method is intended to run in parallel during training or inference. ( there are additional modules for contniual learning, improving emergent modularity, etc )

# desiored improvments

1- keep only the contrastive step, replace all other workaround for space sollpace and simmereis with sigreg
2- clusters_labels will drive a cluster_loss. The way i want to use them is the following: when i identify active/inactive is normal that they might belong to diffenrent cluster, but also, withing that cluster some could be active and some inactive. what i want to do is to use those 2 informations combined to "split" the cluster using the optimizer, so for a given cluster "push away" active from inactives. A part of my mind suggest me that, the same method applied to all position, could be applied identical to individual clusters which contains both active and inactive ( pure clusters don't need it )


# from spacemath.py


```py
    @classmethod
    def activations_to_scores(cls, xd: Tensor, layer_type: str, activation_fn: nn.Module = None, x_quantile=0.9, cnn_locality_weight=0.5) -> Tensor:
        q = None
        act = activation_fn
                
        if isinstance(act, nn.Softmax): 
            q = torch.max(xd, dim=1).values.unsqueeze(1).detach()
            scores = torch.ones_like(xd) * -1.0
            scores[xd>=q] = 1.0
            return scores
        
        if act is not None:
            q = torch.quantile(act(xd), x_quantile, interpolation='higher', dim=1).unsqueeze(1).detach()
        else:
            q = torch.quantile(xd, x_quantile, interpolation='higher', dim=1).unsqueeze(1).detach()

        scores = cls.normalize_by_threshold(xd, q)

        return scores
```



```py
    @classmethod
    def normalize_by_threshold(cls, scores:Tensor, q:Tensor) -> Tensor:
        idxs_on, idxs_off = cls.split_active_group(scores, threshold=q)
        scores_on = scores * idxs_on # keep X >= q untouched
        scores_off = ((scores - q) - cls.eps) * idxs_off # brings X < q below zero
        scores_max = torch.abs(torch.amax(scores_on,dim=1)).unsqueeze(1) + cls.eps
        scores_min = torch.abs(torch.amin(scores_off,dim=1)).unsqueeze(1) + cls.eps
        scores = (scores_on / scores_max ) + (scores_off / scores_min )
        return scores

    @classmethod
    def split_active_group(cls, x:Tensor, threshold=0.0) -> tuple[Tensor, Tensor]:
        idxs_on  = (x >= threshold)
        idxs_off = (~idxs_on)
        return idxs_on, idxs_off     
```
        
## activations_to_scores
torch.quantile


        
# from layerspace.py


```py
    def process_activations(self, layer_id: str, activations: torch.Tensor, clusters_labels: torch.Tensor|None )-> tuple[tuple, Tensor, Tensor | None]:
        layer_info = self.get_space(layer_id)[0]
        scores = self._process_activations(self.config.modular_space_active_q, activations, layer_info["layer_type"], layer_info["activation"])
        scores = scores.detach().clone()  # Detach and clone to avoid in-place modifications
        return self.process_scores(layer_id, scores, clusters_labels)

    @staticmethod_compile
    #@staticmethod
    def _process_activations(active_q: float, activations: torch.Tensor, layer_type: str, activation_fn) -> Tensor:
        # [1 |0| -1]
        scores = SpaceMath.activations_to_scores( 
            activations, layer_type, activation_fn, 
            active_q, cnn_locality_weight=0.5)
        
        return scores
    
    
    def process_scores(self, layer_id: str, scores: torch.Tensor, clusters_labels: torch.Tensor|None):
        _, positions, optimizer, stats = self.get_space(layer_id)
        if not self.stats_enabled: stats = None
        losses = self._process_scores(self.config, scores, positions, stats, clusters_labels)
        loss= losses[0]  # total loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    

        loss_names = ["loss", "loss_on", "loss_off", "loss_centers"] 
        losses = {name : loss.detach().cpu().numpy() for name, loss in zip( loss_names, losses)}
        return losses, positions.detach().cpu().numpy(), None if stats is None else stats.detach().cpu().numpy() 

    @staticmethod_compile
    #@staticmethod
    def _process_scores(config: PipelineConfig, scores: torch.Tensor, positions: torch.Tensor, stats: torch.Tensor, clusters_labels: torch.Tensor|None)-> tuple[torch.Tensor, tuple[ torch.Tensor, torch.Tensor, torch.Tensor]]:
        bs = scores.shape[0]
        
        # [True, False]
        idxs: tuple[torch.Tensor, torch.Tensor] = SpaceMath.split_active_group(scores)
        idxs_on, idxs_off = idxs

        # [0,1]
        scores_on = (scores * idxs_on).unsqueeze(2)
        scores_off = -(scores * idxs_off).unsqueeze(2)

        if stats is not None:
            stats[:, 0] += idxs_on.sum(dim=0).detach()                    # (N,)  
            stats[:, 1] += idxs_off.sum(dim=0).detach()                   # (N,)  
            stats[:, 2] += scores_on.sum(dim=0).squeeze(-1).detach()      # (N,1) -> (N,)
            stats[:, 3] += scores_off.sum(dim=0).squeeze(-1).detach()     # (N,1) -> (N,)

        
        # compute centers by weighted average of positions 
        batch_positions = positions.expand( bs, -1, -1 )
        
        # performing (batch_positions * weight_on/off).sum(dim=1) but more efficently
        scores_stack = torch.cat([scores_on, scores_off], dim=2)                    # (bs, N, 2)
        weights = scores_stack / scores_stack.sum(dim=1, keepdim=True).clamp(min=SpaceMath.eps)  # (bs, N, 2)
        centers = torch.bmm(weights.transpose(1, 2), batch_positions)               # (bs, 2, D)
        center_on, center_off = centers[:, 0], centers[:, 1]
       
        # LOSS: contastive CUs. On VS Off 
        batch_centers = center_on.unsqueeze(1).expand(-1, batch_positions.shape[1], -1)
        center_dist: torch.Tensor = SpaceMath.distance(batch_positions, batch_centers, dim=2, distance_type=config.modular_space_distance, pairwise=False)
        center_dist = center_dist.unsqueeze(2)  # (bs, N, 2)
        dists = torch.bmm(weights.transpose(1, 2), center_dist)
        
        dist_on, dist_off = dists[:, 0, 0], dists[:, 1, 0]


        loss_on = dist_on.mean() # mean for batch / sample 
        loss_off = dist_off.mean() # mean for batch / sample


        loss_on  = torch.log( 1 + loss_on ) * float(config.modular_space_loss_on)
        loss_off = torch.log( 1 + loss_off ) * float(config.modular_space_loss_off)
        loss = loss_on - loss_off

        # LOSS: center distance: break simmerity between on and off centers ( ex: random init uniform + random sampling )
        dist_centers: torch.Tensor = SpaceMath.distance(center_on, center_off, dim=1, distance_type=config.modular_space_distance) # on the same datapoint
        loss_centers = dist_centers.mean()
        loss -= loss_centers * float(config.modular_space_loss_centers)

  

        return loss, loss_on, loss_off, loss_centers

```


# from sigreg.py



```py
def sigreg(pos: torch.Tensor, sketch_dim: int = 64, method: str = "weak") -> torch.Tensor:
    if method == "weak":
        return sigreg_weak(pos, sketch_dim)
    elif method == "strong":
        return sigreg_strong(pos, sketch_dim)
    else:
        raise ValueError(f"Unknown method: {method}")

def sigreg_weak(pos: torch.Tensor, sketch_dim: int = 64) -> torch.Tensor:
    # https://arxiv.org/abs/2603.05924v1
    N, D = pos.shape
    if D > sketch_dim:
        S = torch.randn(sketch_dim, D, device=pos.device, dtype=pos.dtype) / (D ** 0.5)
        pos = pos @ S.T
        D = sketch_dim
    pos = pos - pos.mean(0, keepdim=True)
    cov = pos.T @ pos / (N - 1 + 1e-6)
    eye = torch.eye(D, device=pos.device, dtype=pos.dtype)
    return torch.sqrt((cov - eye).pow(2).sum() + 1e-12)   # Frobenius, eps contro il NaN

def sigreg_strong(pos: torch.Tensor, sketch_dim: int = 64) -> torch.Tensor:
    # https://arxiv.org/abs/2511.08544
    N, D = pos.shape
    A = torch.randn(D, sketch_dim, device=pos.device, dtype=pos.dtype)
    A = A / (A.norm(dim=0, keepdim=True) + 1e-6)
    t = torch.linspace(-5, 5, 17, device=pos.device, dtype=pos.dtype)
    target = torch.exp(-0.5 * t ** 2)
    xt = (pos @ A).unsqueeze(2) * t                      # (N, S, T)
    re, im = xt.cos().mean(0), xt.sin().mean(0)          # evita i complessi (più sicuro con compile)
    err = ((re - target) ** 2 + im ** 2) * target
    return (torch.trapz(err, t, dim=1) * N).mean()
```

## Overall description



