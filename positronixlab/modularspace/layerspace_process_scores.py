"""
Drop-in replacement for LayerSpace.process_scores / _process_scores.

UNTESTED: written without running it against your stack (no torch in my sandbox).
Run the smoke test at the bottom on a CPU first.

New TrialConfig fields (replace modular_space_loss_on/_off/_centers):
    modular_space_tau            float  softmax temperature (cosine: try 0.05-0.2)
    modular_space_loss_global    float  weight of the whole-layer contrast
    modular_space_loss_cluster   float  weight of the per-cluster contrast
    modular_space_loss_sigreg    float  weight of SIGReg (0 = off)
    modular_space_sigreg_method  str    "weak" | "strong"
    modular_space_sigreg_dim     int    sketch dim (64)
    modular_space_freq_alpha     float  0 = off; ~0.5 = inverse-frequency weighting of the centre
ASSUMPTION: config.modular_space_distance is a string starting with "cos" for cosine.
"""
import math
import torch
import torch.nn.functional as F

# from .sigreg import sigreg
# from ... import staticmethod_compile, SpaceMath   # same imports as before


def _is_cosine(config) -> bool:
    return str(config.modular_space_distance).lower().startswith("cos")


def _contrast(P: torch.Tensor, a_on: torch.Tensor, a_off: torch.Tensor,
              M: torch.Tensor, tau: float, cosine: bool, eps: float = 1e-8):
    """
    One function for both levels.
      P     (N, D)   positions (unit-norm if cosine)
      a_on  (bs, N)  soft "fires" weights  >= 0   (optionally frequency-weighted)
      a_off (bs, N)  soft "silent" weights >= 0   (only used for gating)
      M     (K, N)   {0,1} scope masks. Global: K=1, all ones. Clusters: one row each.

    Per datapoint and per scope k: centre = weighted mean of the on units inside k;
    cross-entropy of "which unit in k is close to the centre" against the on-weights.
    Scopes with no on-mass or no off-mass (pure clusters) are gated out: zero loss, zero grad.
    """
    Mb = M.unsqueeze(0)                                          # (1, K, N)
    Wk = a_on.unsqueeze(1) * Mb                                  # (bs, K, N)
    mass_on = Wk.sum(-1)                                         # (bs, K)
    mass_off = (a_off.unsqueeze(1) * Mb).sum(-1)                 # (bs, K)
    gate = ((mass_on > eps) & (mass_off > eps)).to(P.dtype)      # (bs, K)

    q = Wk / mass_on.unsqueeze(-1).clamp_min(eps)                # (bs, K, N), rows sum to 1 (or 0)
    C = q @ P                                                    # (bs, K, D) centres

    if cosine:
        C = F.normalize(C, dim=-1)
        logits = (C @ P.T) / tau                                 # cosine / tau
    else:
        d2 = C.pow(2).sum(-1, keepdim=True) - 2.0 * (C @ P.T) + P.pow(2).sum(-1)
        logits = -d2.clamp_min(0.0) / tau

    neg = -torch.finfo(logits.dtype).max / 2                     # finite "-inf": no 0*inf NaNs
    logits = logits.masked_fill(Mb == 0, neg)
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    ce = -(q * logp).sum(-1)                                     # (bs, K)

    loss = (ce * gate).sum() / gate.sum().clamp_min(1.0)
    return loss, gate.mean().detach()


class LayerSpaceMixin:   # keep your real class; these are its methods

    # ------------------------------------------------------------------ eager side
    def process_scores(self, layer_id: str, scores: torch.Tensor, clusters_labels: torch.Tensor | None):
        _, positions, optimizer, stats = self.get_space(layer_id)
        if not self.stats_enabled:
            stats = None
        cfg = self.config

        # side effects stay OUTSIDE the compiled region
        freq = None
        if stats is not None:
            self._update_stats(stats, scores)
            if float(cfg.modular_space_freq_alpha) > 0:
                freq = (stats[:, 0] / (stats[:, 0] + stats[:, 1]).clamp_min(1.0)).detach()

        membership = self._membership(clusters_labels, positions.device, positions.dtype)

        losses = self._process_scores(cfg, scores, positions, freq, membership)
        optimizer.zero_grad(set_to_none=True)
        losses[0].backward()
        optimizer.step()

        names = ["loss", "loss_global", "loss_cluster", "loss_sigreg", "frac_mixed_global", "frac_mixed_cluster"]
        out = {n: l.detach().cpu().numpy() for n, l in zip(names, losses)}
        return out, positions.detach().cpu().numpy(), None if stats is None else stats.detach().cpu().numpy()

    @staticmethod
    @torch.no_grad()
    def _update_stats(stats: torch.Tensor, scores: torch.Tensor) -> None:
        on = scores >= 0                                          # same split as split_active_group(threshold=0)
        stats[:, 0] += on.sum(0)
        stats[:, 1] += (~on).sum(0)
        stats[:, 2] += scores.clamp_min(0).sum(0)
        stats[:, 3] += (-scores).clamp_min(0).sum(0)

    @staticmethod
    def _membership(labels, device, dtype):
        """(K, N) one-hot masks from t-1 labels; None if there is nothing to split."""
        if labels is None:
            return None
        lab = torch.as_tensor(labels, device=device).long()
        K = int(lab.max().item()) + 1                             # one sync; labels are normally CPU anyway
        if K < 2:
            return None
        M = F.one_hot(lab.clamp_min(0), K).T.to(dtype)            # (K, N)
        return M * (lab >= 0).to(dtype).unsqueeze(0)              # label -1 (noise) -> in no cluster

    # ------------------------------------------------------------------ compiled side
    @staticmethod
    @torch.compile
    def _process_scores(config, scores: torch.Tensor, positions: torch.Tensor,
                        freq: torch.Tensor | None, membership: torch.Tensor | None):
        bs, N = scores.shape
        D = positions.shape[1]
        cosine = _is_cosine(config)
        tau = float(config.modular_space_tau)

        P = F.normalize(positions, dim=-1) if cosine else positions

        # scores in [-1, 1] (normalize_by_threshold): >0 fires, <0 silent
        a_on = scores.clamp_min(0.0)
        a_off = (-scores).clamp_min(0.0)

        alpha = float(config.modular_space_freq_alpha)
        if freq is not None and alpha > 0:
            g = freq.clamp_min(1e-3).pow(-alpha)
            a_on = a_on * (g / g.mean())                          # IDF-like: hubs don't dominate the centre

        zero = scores.new_zeros(())
        w_g = float(config.modular_space_loss_global)
        w_c = float(config.modular_space_loss_cluster)
        w_s = float(config.modular_space_loss_sigreg)

        # whole-layer contrast: scope = every unit
        loss_global, mixed_g = zero, zero
        if w_g > 0:
            M_all = torch.ones(1, N, device=P.device, dtype=P.dtype)
            loss_global, mixed_g = _contrast(P, a_on, a_off, M_all, tau, cosine)

        # per-cluster contrast: splits fires/silent *inside* a cluster
        loss_cluster, mixed_c = zero, zero
        if w_c > 0 and membership is not None:
            loss_cluster, mixed_c = _contrast(P, a_on, a_off, membership, tau, cosine)

        # SIGReg: isotropy / scale anchor. On the sphere rows have norm 1 -> cov = I/D, so rescale to I.
        loss_sigreg = zero
        if w_s > 0:
            X = P * math.sqrt(D) if cosine else P
            loss_sigreg = sigreg(X, int(config.modular_space_sigreg_dim), str(config.modular_space_sigreg_method))

        loss = w_g * loss_global + w_c * loss_cluster + w_s * loss_sigreg
        return loss, loss_global, loss_cluster, loss_sigreg, mixed_g, mixed_c


# ---------------------------------------------------------------------- smoke test
if __name__ == "__main__":
    torch.manual_seed(0)
    N, D, bs = 120, 8, 64
    groups = torch.arange(N) // 10                                # 12 modules of 10 units
    # each datapoint turns on 3 random modules
    on_mod = torch.stack([torch.randperm(12)[:3] for _ in range(bs)])
    fires = (groups[None, :, None] == on_mod[:, None, :]).any(-1).float()
    scores = fires * torch.rand(bs, N) - (1 - fires) * torch.rand(bs, N)

    pos = torch.nn.Parameter(torch.randn(N, D))
    opt = torch.optim.Adam([pos], lr=1e-2)
    P0 = F.normalize(pos, dim=-1)
    M = F.one_hot(groups, 12).T.float()
    for step in range(300):
        P = F.normalize(pos, dim=-1)
        l, _ = _contrast(P, scores.clamp_min(0), (-scores).clamp_min(0), torch.ones(1, N), 0.1, True)
        lc, _ = _contrast(P, scores.clamp_min(0), (-scores).clamp_min(0), M, 0.1, True)
        loss = l + lc
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(step, float(l), float(lc))
    assert torch.isfinite(loss)
    P = F.normalize(pos, dim=-1)
    sim = P @ P.T
    same = (groups[:, None] == groups[None, :]).float()
    print("mean cos within module:", float((sim * same).sum() / same.sum()),
          "| between:", float((sim * (1 - same)).sum() / (1 - same).sum()))
