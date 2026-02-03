from typing import Optional, Dict
from mmengine.hooks import Hook
from mmengine.runner import Runner
from tqdm import tqdm
import sys

class PrettyPrintHook(Hook):
    """Простой tqdm-бар + сводка по эпохе."""

    def __init__(self, priority: str = 'NORMAL', update_interval: int = 10):
        super().__init__()
        self.priority = priority
        self.update_interval = update_interval
        self.pbar: Optional[tqdm] = None

    def before_train_epoch(self, runner: Runner, **kwargs):
        if self.pbar is not None:
            self.pbar.close()

        epoch = runner.epoch + 1
        max_epochs = runner.max_epochs

        # длина эпохи: честное число батчей
        try:
            total = len(runner.train_dataloader)
        except Exception:
            total = None  # fallback: покажет N/?

        desc = f"Epoch [{epoch}/{max_epochs}]"
        self.pbar = tqdm(
            total=total,
            desc=desc,
            leave=True,
            dynamic_ncols=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

    def after_train_iter(self, runner: Runner, batch_idx: int, data_batch=None, outputs=None, **kwargs):
        if self.pbar is None:
            return

        self.pbar.update(1)

        if runner.iter % self.update_interval == 0:
            loss_str = self._get_loss_str(runner)
            if loss_str:
                self.pbar.set_postfix_str(loss_str)

    def after_train_epoch(self, runner: Runner, **kwargs):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

        print()

        scalars = runner.message_hub.log_scalars
        lr = 0.0
        if 'lr' in scalars:
            lr = scalars['lr'].current()

        train_metrics = []
        for key, buf in scalars.items():
            if 'loss' in key:
                try:
                    val = buf.current()
                except Exception:
                    continue
                train_metrics.append(f"{key}: {val:.4f}")

        train_metrics.sort(key=lambda x: 0 if x.startswith('loss:') else 1)
        metrics_str = "  |  ".join(train_metrics) if train_metrics else "Нет метрик"

        print(f"{'='*80}")
        print(f"Train (Epoch {runner.epoch + 1}) | LR: {lr:.2e}")
        print(f"{'-'*80}")
        print(f"   {metrics_str}")
        print(f"{'='*80}")
        sys.stdout.flush()

    def after_val_epoch(self, runner: Runner, metrics: Optional[Dict[str, float]] = None, **kwargs):
        if not metrics:
            return
        print()
        print(f"{'='*80}")
        # Было: runner.epoch + 1
        print(f"Validation (Epoch {runner.epoch})")  # epoch уже инкрементирован, показываем как есть
        print(f"{'-'*80}")
        parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        print(f"   {'  |  '.join(parts)}")
        print(f"{'='*80}\n")
        sys.stdout.flush()

    def _get_loss_str(self, runner: Runner) -> str:
        try:
            if 'loss' in runner.message_hub.log_scalars:
                val = runner.message_hub.log_scalars['loss'].current()
                return f"loss: {val:.4f}"
        except Exception:
            pass
        return ""
