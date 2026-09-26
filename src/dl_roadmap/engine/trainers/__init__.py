"""Trainer variants that change how the loop reads a batch or calls the model."""

from dl_roadmap.engine.trainers.teacher_forcing import (
    TeacherForcingParts,
    TeacherForcingTrainer,
)

__all__ = ["TeacherForcingParts", "TeacherForcingTrainer"]
