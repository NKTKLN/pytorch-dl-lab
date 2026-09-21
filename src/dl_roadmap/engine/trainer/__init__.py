"""Training loop split into a trainer and the collaborators it delegates to."""

from dl_roadmap.engine.trainer.base import (
    BaseTrainer,
    Batch,
    EpochCallback,
    LossFn,
    PairBatch,
)
from dl_roadmap.engine.trainer.config import TrainingConfig
from dl_roadmap.engine.trainer.context import Phase, StepContext
from dl_roadmap.engine.trainer.early_stopping import (
    CombinedEarlyStopping,
    EarlyStopping,
    GapThresholdEarlyStopping,
    GeneralizationGapEarlyStopping,
    MetricEarlyStopping,
    ThresholdEarlyStopping,
    ValLossEarlyStopping,
)
from dl_roadmap.engine.trainer.loss_tracker import (
    LossTracker,
    MeanLossTracker,
    PerTokenLossTracker,
)
from dl_roadmap.engine.trainer.losses import LossBundle, make_token_loss
from dl_roadmap.engine.trainer.metrics_manager import MetricsManager
from dl_roadmap.engine.trainer.online_metrics import (
    GeneratedRougeScore,
    Metric,
    RougeScore,
    TokenAccuracy,
)
from dl_roadmap.engine.trainer.optimization import (
    AmpMode,
    GradNormalizer,
    NoOptimization,
    Optimization,
    OptimizationConfig,
    OptimizationEngine,
)
from dl_roadmap.engine.trainer.schedulers import (
    EpochWarmupScheduler,
    WarmupScheduler,
    step_scheduler,
)
from dl_roadmap.engine.trainer.state_store import TrainerStateStore
from dl_roadmap.engine.trainer.teacher_forcing import TeacherForcingTrainer

__all__ = [
    "AmpMode",
    "BaseTrainer",
    "Batch",
    "CombinedEarlyStopping",
    "EarlyStopping",
    "EpochCallback",
    "EpochWarmupScheduler",
    "GapThresholdEarlyStopping",
    "GeneralizationGapEarlyStopping",
    "GeneratedRougeScore",
    "GradNormalizer",
    "LossBundle",
    "LossFn",
    "LossTracker",
    "MeanLossTracker",
    "Metric",
    "MetricEarlyStopping",
    "MetricsManager",
    "NoOptimization",
    "Optimization",
    "OptimizationConfig",
    "OptimizationEngine",
    "PairBatch",
    "PerTokenLossTracker",
    "Phase",
    "RougeScore",
    "StepContext",
    "TeacherForcingTrainer",
    "ThresholdEarlyStopping",
    "TokenAccuracy",
    "TrainerStateStore",
    "TrainingConfig",
    "ValLossEarlyStopping",
    "WarmupScheduler",
    "make_token_loss",
    "step_scheduler",
]
