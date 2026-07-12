"""
Curriculum Fault Injection Scheduler

Schedules the fault injection rate over training:
  Warmup: 0 fault rate (clean data)
  Ramp:   linearly increase from min to max_fault_rate
  Steady: constant max_fault_rate

Curriculum training prevents instability from early heavy fault injection.
"""


class CurriculumScheduler:
    """
    Manages fault injection rate over training epochs.

    Args:
        warmup_epochs: Epochs on clean data only.
        ramp_epochs: Epochs to linearly ramp up fault rate.
        max_fault_rate: Final steady-state fault rate.
        min_fault_rate: Starting rate after warmup.
    """

    def __init__(self, warmup_epochs: int = 10, ramp_epochs: int = 30, max_fault_rate: float = 0.6, min_fault_rate: float = 0.1):
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.max_fault_rate = max_fault_rate
        self.min_fault_rate = min_fault_rate

    def get_fault_rate(self, epoch: int) -> float:
        """
        Returns fault injection rate for the given epoch.

        Args:
            epoch: 0-indexed training epoch.
        Returns:
            Fault rate in [0, max_fault_rate].
        """
        if epoch < self.warmup_epochs:
            return 0.0
        ramp_e = epoch - self.warmup_epochs
        if ramp_e >= self.ramp_epochs:
            return self.max_fault_rate
        progress = ramp_e / self.ramp_epochs
        return self.min_fault_rate + progress * (self.max_fault_rate - self.min_fault_rate)

    def __repr__(self) -> str:
        return (f"CurriculumScheduler(warmup={self.warmup_epochs}, ramp={self.ramp_epochs}, "
                f"max_rate={self.max_fault_rate})")
