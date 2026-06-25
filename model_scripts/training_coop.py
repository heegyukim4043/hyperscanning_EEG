# training_coop.py
"""
TrainerCoop
===========
MTAD_GAT_DGL_Coop 전용 Trainer.

Loss = sqrt(MSE_forecast)
     + sqrt(MSE_recon)
     + lambda_inter * L_inter      ← ① inter-brain contrast
     + lambda_coop  * BCE_coop     ← ② cooperation classification

① L_inter:  협력 구간(label=1)에서 A_feat inter > A_feat intra 가 되도록 유도
            L_inter = mean_over_label1( ReLU(intra_mean - inter_mean) )

② BCE_coop: CoopHead(A_feat[inter])의 이진 분류 loss
            pos_weight 으로 클래스 불균형 보정
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class TrainerCoop:
    def __init__(
        self,
        model,
        optimizer,
        window_size: int,
        n_features: int,
        n_ch_per: int = 19,
        n_epochs: int = 150,
        batch_size: int = 64,
        init_lr: float = 1e-4,
        lambda_inter: float = 0.1,
        lambda_coop: float = 1.0,
        pos_weight: float = 1.0,       # n_label0 / n_label1 for class balancing
        use_cuda: bool = False,
        dload: str = "",
        log_dir: str = "output/",
        print_every: int = 1,
        log_tensorboard: bool = True,
        args_summary: str = "",
        # ── early stopping ────────────────────────────────────────────────
        patience: int = 30,            # 0 = disabled
        min_delta: float = 1e-3,
        # ── mixed precision ───────────────────────────────────────────────
        use_amp: bool = False,
        lambda_align: float = 0.0,
        lambda_delta: float = 0.0,
        lambda_coop_schedule = None,
        lambda_align_schedule = None,
        lambda_delta_schedule = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.window_size = window_size
        self.n_features = n_features
        self.n_ch_per = n_ch_per
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.lambda_inter = lambda_inter
        self.lambda_coop = lambda_coop
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.dload = dload
        self.log_dir = log_dir
        self.print_every = print_every
        self.log_tensorboard = log_tensorboard

        self.recon_criterion = nn.MSELoss()
        self.forecast_criterion = nn.MSELoss()
        pw = torch.tensor([pos_weight], dtype=torch.float32)
        self.coop_criterion = nn.BCEWithLogitsLoss(pos_weight=pw.to(self.device))

        self.patience  = patience
        self.min_delta = min_delta
        self.use_amp   = use_amp and (self.device == "cuda")
        self.lambda_align = float(lambda_align)
        self.lambda_delta = float(lambda_delta)
        self.lambda_coop_schedule = lambda_coop_schedule
        self.lambda_align_schedule = lambda_align_schedule
        self.lambda_delta_schedule = lambda_delta_schedule
        self.cur_lambda_coop = float(lambda_coop)
        self.cur_lambda_align = float(lambda_align)
        self.cur_lambda_delta = float(lambda_delta)

        self.losses = {
            "train_total":    [], "train_forecast": [], "train_recon": [],
            "train_inter":    [], "train_coop":     [],
            "train_align":    [], "train_delta":    [],
            "val_total":      [], "val_forecast":   [], "val_recon":   [],
            "val_inter":      [], "val_coop":        [],
            "val_align":      [], "val_delta":       [],
        }
        self.epoch_times = []
        self._best_val = float("inf")
        self._patience_counter = 0
        self.stopped_epoch = None

        if self.device == "cuda":
            self.model.cuda()

        if self.log_tensorboard:
            self.writer = SummaryWriter(log_dir)
            self.writer.add_text("args_summary", args_summary)

    # ------------------------------------------------------------------
    def _unpack_loss_tuple(self, loss_tuple):
        if len(loss_tuple) == 5:
            loss, fl, rl, il, cl = loss_tuple
            zero = torch.zeros((), device=self.device)
            return loss, fl, rl, il, cl, zero, zero
        if len(loss_tuple) == 7:
            return loss_tuple
        raise ValueError(f"Unexpected loss tuple length: {len(loss_tuple)}")

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_lambda(epoch_idx: int, base_value: float, schedule):
        """
        epoch_idx: 0-based
        schedule: None or dict {"boundaries":[int,...], "values":[float,...]}
        values length must be len(boundaries)+1.
        """
        if not schedule:
            return float(base_value)
        boundaries = [int(x) for x in schedule.get("boundaries", [])]
        values = [float(x) for x in schedule.get("values", [])]
        if len(values) != len(boundaries) + 1:
            return float(base_value)
        ep1 = epoch_idx + 1
        for i, b in enumerate(boundaries):
            if ep1 <= b:
                return values[i]
        return values[-1]

    # ------------------------------------------------------------------
    def _update_epoch_lambdas(self, epoch_idx: int):
        self.cur_lambda_coop = self._resolve_lambda(
            epoch_idx, self.lambda_coop, self.lambda_coop_schedule
        )
        self.cur_lambda_align = self._resolve_lambda(
            epoch_idx, self.lambda_align, self.lambda_align_schedule
        )
        self.cur_lambda_delta = self._resolve_lambda(
            epoch_idx, self.lambda_delta, self.lambda_delta_schedule
        )

    # ------------------------------------------------------------------
    def _inter_intra_means(self, A_feat: torch.Tensor):
        c = self.n_ch_per
        inter_blocks = [
            A_feat[:, :c, c:2 * c],
            A_feat[:, :c, 2 * c:3 * c],
            A_feat[:, c:2 * c, :c],
            A_feat[:, c:2 * c, 2 * c:3 * c],
            A_feat[:, 2 * c:3 * c, :c],
            A_feat[:, 2 * c:3 * c, c:2 * c],
        ]
        intra_blocks = [
            A_feat[:, :c, :c],
            A_feat[:, c:2 * c, c:2 * c],
            A_feat[:, 2 * c:3 * c, 2 * c:3 * c],
        ]
        inter = torch.stack([b.mean(dim=[1, 2]) for b in inter_blocks], dim=1).mean(dim=1)
        intra = torch.stack([b.mean(dim=[1, 2]) for b in intra_blocks], dim=1).mean(dim=1)
        return inter, intra

    # ------------------------------------------------------------------
    def _compute_losses(self, xw, y_next, y_coop):
        """
        xw:     [B, W, F]   input window
        y_next: [B, F]      next-step signal  (forecast target)
        y_coop: [B]         float 0/1         (cooperation label)

        Returns: (total, forecast, recon, inter, coop, align, delta)
        """
        preds, recons, coop_logits, A_feat = self.model(xw)

        # shape alignment
        if preds.ndim == 3:
            preds = preds.squeeze(1)        # [B,1,F] → [B,F]
        if y_next.ndim == 3:
            y_next = y_next.squeeze(1)

        forecast_loss = torch.sqrt(self.forecast_criterion(y_next, preds))
        recon_loss    = torch.sqrt(self.recon_criterion(xw, recons))

        # Inter-brain contrast loss across all directed pair blocks.
        inter, intra = self._inter_intra_means(A_feat)
        # support multi-label y_coop [B,3]: any pair cooperating → mask [B]
        if y_coop.dim() == 2:
            coop_mask = (y_coop > 0.5).any(dim=1).float()                        # [B]
        else:
            coop_mask = (y_coop > 0.5).float()                                   # [B]
        inter_loss = (coop_mask * F.relu(intra - inter)).mean()

        # ② Cooperation classification loss (BCE with pos_weight)
        coop_loss = self.coop_criterion(coop_logits, y_coop)
        align_loss = torch.zeros((), device=self.device)
        delta_loss = torch.zeros((), device=self.device)

        total = (
            forecast_loss
            + recon_loss
            + self.lambda_inter * inter_loss
            + self.cur_lambda_coop * coop_loss
            + self.cur_lambda_align * align_loss
            + self.cur_lambda_delta * delta_loss
        )
        return total, forecast_loss, recon_loss, inter_loss, coop_loss, align_loss, delta_loss

    # ------------------------------------------------------------------
    def fit(self, train_loader, val_loader=None):
        amp_tag = " AMP" if self.use_amp else ""
        es_tag  = f" patience={self.patience}" if self.patience > 0 else ""
        print(f"[TrainerCoop] Training for {self.n_epochs} epochs  "
              f"lambda_inter={self.lambda_inter}  lambda_coop={self.lambda_coop}  "
              f"lambda_align={self.lambda_align}  lambda_delta={self.lambda_delta}"
              f"{amp_tag}{es_tag}")
        if self.lambda_coop_schedule or self.lambda_align_schedule or self.lambda_delta_schedule:
            print(
                "[LambdaSchedule] "
                f"coop={self.lambda_coop_schedule} "
                f"align={self.lambda_align_schedule} "
                f"delta={self.lambda_delta_schedule}"
            )
        t0 = time.time()

        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        for epoch in range(self.n_epochs):
            ep_t = time.time()
            self.model.train()
            self._update_epoch_lambdas(epoch)

            ep = {"total": [], "fc": [], "rc": [], "il": [], "cl": [], "al": [], "dl": []}
            for xw, y_next, y_coop in train_loader:
                xw     = xw.to(self.device)
                y_next = y_next.to(self.device)
                y_coop = y_coop.float().to(self.device)

                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss, fl, rl, il, cl, al, dl = self._unpack_loss_tuple(
                        self._compute_losses(xw, y_next, y_coop)
                    )

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                ep["total"].append(loss.item())
                ep["fc"].append(fl.item())
                ep["rc"].append(rl.item())
                ep["il"].append(il.item())
                ep["cl"].append(cl.item())
                ep["al"].append(al.item())
                ep["dl"].append(dl.item())

            t_total = float(np.mean(ep["total"]))
            t_fc    = float(np.mean(ep["fc"]))
            t_rc    = float(np.mean(ep["rc"]))
            t_il    = float(np.mean(ep["il"]))
            t_cl    = float(np.mean(ep["cl"]))
            t_al    = float(np.mean(ep["al"]))
            t_dl    = float(np.mean(ep["dl"]))

            self.losses["train_total"].append(t_total)
            self.losses["train_forecast"].append(t_fc)
            self.losses["train_recon"].append(t_rc)
            self.losses["train_inter"].append(t_il)
            self.losses["train_coop"].append(t_cl)
            self.losses["train_align"].append(t_al)
            self.losses["train_delta"].append(t_dl)

            v_total = "NA"
            if val_loader is not None:
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    v_total, v_fc, v_rc, v_il, v_cl, v_al, v_dl = self.evaluate(val_loader)
                self.losses["val_total"].append(v_total)
                self.losses["val_forecast"].append(v_fc)
                self.losses["val_recon"].append(v_rc)
                self.losses["val_inter"].append(v_il)
                self.losses["val_coop"].append(v_cl)
                self.losses["val_align"].append(v_al)
                self.losses["val_delta"].append(v_dl)

                # best model + early stopping
                if v_total < self._best_val - self.min_delta:
                    self._best_val = v_total
                    self._patience_counter = 0
                    self.save("models.pt")
                else:
                    if self.patience > 0:
                        self._patience_counter += 1
                        if self._patience_counter >= self.patience:
                            self.stopped_epoch = epoch + 1
                            print(f"[EarlyStopping] epoch {epoch+1} "
                                  f"no improvement for {self.patience} epochs "
                                  f"best_val={self._best_val:.5f}")
                            break

            if self.log_tensorboard:
                self._write_tb(epoch, t_fc, t_rc, t_il, t_cl, t_al, t_dl, t_total, v_total)

            elapsed = time.time() - ep_t
            self.epoch_times.append(elapsed)

            if epoch % self.print_every == 0:
                s = (f"[Epoch {epoch+1:3d}] "
                     f"fc={t_fc:.4f}  rc={t_rc:.4f}  "
                     f"inter={t_il:.4f}  coop={t_cl:.4f}  "
                      f"align={t_al:.4f}  delta={t_dl:.4f}  "
                     f"total={t_total:.4f}  "
                     f"lmb(c/a/d)=({self.cur_lambda_coop:.3f}/{self.cur_lambda_align:.3f}/{self.cur_lambda_delta:.3f})")
                if v_total != "NA":
                    s += f"  | val={v_total:.4f}  patience={self._patience_counter}"
                s += f"  [{elapsed:.1f}s]"
                print(s)

        if val_loader is None:
            self.save("models.pt")

        elapsed_total = int(time.time() - t0)
        stop_info = (f"  (stopped at epoch {self.stopped_epoch})"
                     if self.stopped_epoch else "")
        print(f"-- Training done in {elapsed_total}s.{stop_info}")

    # ------------------------------------------------------------------
    def evaluate(self, loader):
        self.model.eval()
        ep = {"total": [], "fc": [], "rc": [], "il": [], "cl": [], "al": [], "dl": []}
        with torch.no_grad():
            for xw, y_next, y_coop in loader:
                xw     = xw.to(self.device)
                y_next = y_next.to(self.device)
                y_coop = y_coop.float().to(self.device)
                loss, fl, rl, il, cl, al, dl = self._unpack_loss_tuple(
                    self._compute_losses(xw, y_next, y_coop)
                )
                ep["total"].append(loss.item())
                ep["fc"].append(fl.item())
                ep["rc"].append(rl.item())
                ep["il"].append(il.item())
                ep["cl"].append(cl.item())
                ep["al"].append(al.item())
                ep["dl"].append(dl.item())
        return (
            float(np.mean(ep["total"])),
            float(np.mean(ep["fc"])),
            float(np.mean(ep["rc"])),
            float(np.mean(ep["il"])),
            float(np.mean(ep["cl"])),
            float(np.mean(ep["al"])),
            float(np.mean(ep["dl"])),
        )

    # ------------------------------------------------------------------
    def save(self, fn: str):
        torch.save(self.model.state_dict(), os.path.join(self.dload, fn))

    def _write_tb(self, epoch, fc, rc, il, cl, al, dl, total, val_total):
        if not self.log_tensorboard:
            return
        self.writer.add_scalar("Loss/train_forecast", fc,    epoch)
        self.writer.add_scalar("Loss/train_recon",    rc,    epoch)
        self.writer.add_scalar("Loss/train_inter",    il,    epoch)
        self.writer.add_scalar("Loss/train_coop",     cl,    epoch)
        self.writer.add_scalar("Loss/train_align",    al,    epoch)
        self.writer.add_scalar("Loss/train_delta",    dl,    epoch)
        self.writer.add_scalar("Loss/train_total",    total, epoch)
        if val_total != "NA":
            self.writer.add_scalar("Loss/val_total", val_total, epoch)
