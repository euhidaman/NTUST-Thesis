"""
EmberVLM Distillation Package

Teacher-student distillation utilities for training smaller models
using outputs from larger teacher models.
"""

from embervlm.distillation.teacher import TeacherWrapper
from embervlm.distillation.losses import DistillationLoss, HiddenStateDistillationLoss
from embervlm.distillation.generator import SyntheticDataGenerator

__all__ = [
    "TeacherWrapper",
    "DistillationLoss",
    "HiddenStateDistillationLoss",
    "SyntheticDataGenerator",
]

