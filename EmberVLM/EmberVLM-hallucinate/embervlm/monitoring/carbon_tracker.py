"""
Carbon Emissions Tracker for EmberVLM

Uses CodeCarbon to track CO2 emissions during training.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CarbonTracker:
    """
    Carbon emissions tracker using CodeCarbon.

    Tracks:
    - CO2 emissions (kg CO2eq)
    - Energy consumption (kWh)
    - Power draw (W)
    - Training efficiency (samples/kWh)
    """

    def __init__(
        self,
        output_dir: str = "./emissions",
        project_name: str = "EmberVLM",
        measure_power_secs: int = 30,
        save_to_file: bool = True,
        log_level: str = "WARNING",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.project_name = project_name
        self.measure_power_secs = measure_power_secs
        self.save_to_file = save_to_file

        self.tracker = None
        self.enabled = True
        self.start_time = None
        self.total_samples = 0

        # Try to initialize CodeCarbon
        try:
            from codecarbon import EmissionsTracker

            self.tracker = EmissionsTracker(
                output_dir=str(self.output_dir),
                measure_power_secs=measure_power_secs,
                project_name=project_name,
                log_level=log_level,
                save_to_file=save_to_file,
            )
            logger.info("CodeCarbon initialized successfully")

        except ImportError:
            logger.warning("codecarbon not installed. Emissions tracking disabled.")
            self.enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize CodeCarbon: {e}")
            self.enabled = False

        # Manual tracking as backup
        self.manual_metrics = {
            'start_time': None,
            'end_time': None,
            'duration_hours': 0,
            'estimated_kwh': 0,
            'estimated_co2_kg': 0,
        }

    def start(self):
        """Start tracking emissions."""
        self.start_time = datetime.now()
        self.manual_metrics['start_time'] = self.start_time.isoformat()

        if self.enabled and self.tracker:
            try:
                self.tracker.start()
                logger.info("Started carbon tracking")
            except Exception as e:
                logger.warning(f"Failed to start tracker: {e}")

    def stop(self) -> float:
        """
        Stop tracking and return total emissions.

        Returns:
            Total CO2 emissions in kg
        """
        end_time = datetime.now()
        self.manual_metrics['end_time'] = end_time.isoformat()

        if self.start_time:
            duration = (end_time - self.start_time).total_seconds() / 3600
            self.manual_metrics['duration_hours'] = duration

            # Estimate if CodeCarbon not available
            # Assuming ~300W average GPU power
            estimated_kwh = duration * 0.3
            # Average carbon intensity ~0.5 kg CO2/kWh
            estimated_co2 = estimated_kwh * 0.5

            self.manual_metrics['estimated_kwh'] = estimated_kwh
            self.manual_metrics['estimated_co2_kg'] = estimated_co2

        emissions = 0.0

        if self.enabled and self.tracker:
            try:
                emissions = self.tracker.stop()
                logger.info(f"Total emissions: {emissions:.4f} kg CO2eq")
            except Exception as e:
                logger.warning(f"Failed to stop tracker: {e}")
                emissions = self.manual_metrics['estimated_co2_kg']
        else:
            emissions = self.manual_metrics['estimated_co2_kg']

        # Save summary
        self._save_summary(emissions)

        return emissions

    def update_samples(self, num_samples: int):
        """Update total samples processed."""
        self.total_samples += num_samples

    def get_emissions(self) -> float:
        """Return the latest emissions estimate in kg CO2eq."""
        if self.enabled and self.tracker:
            # Prefer tracker internal total if available
            if hasattr(self.tracker, '_total_emissions'):
                try:
                    return float(self.tracker._total_emissions)
                except Exception:
                    pass
            # Fall back to last stop() return if tracker stored it
            if hasattr(self.tracker, 'final_emissions'):
                try:
                    return float(self.tracker.final_emissions)
                except Exception:
                    pass

        # Manual estimate fallback
        return float(self.manual_metrics.get('estimated_co2_kg', 0.0))

    def _save_summary(self, emissions: float):
        """Save emissions summary."""
        summary = {
            'project_name': self.project_name,
            'total_emissions_kg_co2': emissions,
            'total_samples': self.total_samples,
            **self.manual_metrics,
        }

        # Calculate efficiency
        if self.manual_metrics['estimated_kwh'] > 0:
            summary['samples_per_kwh'] = self.total_samples / self.manual_metrics['estimated_kwh']

        # Save to file
        summary_path = self.output_dir / 'emissions_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved emissions summary to {summary_path}")

    def get_current_emissions(self) -> Dict[str, float]:
        """Get current emissions metrics."""
        if not self.enabled or not self.tracker:
            return self.manual_metrics

        try:
            # Get current state from tracker if available
            # This depends on CodeCarbon implementation
            return {
                'emissions_kg': self.tracker._total_emissions if hasattr(self.tracker, '_total_emissions') else 0,
                'energy_kwh': self.tracker._total_energy if hasattr(self.tracker, '_total_energy') else 0,
            }
        except Exception:
            return self.manual_metrics

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class FLOPsTracker:
    """
    Track FLOPs during training.
    """

    def __init__(self):
        self.total_flops = 0
        self.steps = 0

    def add(self, flops: int, batch_size: int = 1):
        """Add FLOPs for a step."""
        self.total_flops += flops * batch_size
        self.steps += 1

    def get_total(self) -> int:
        """Get total FLOPs."""
        return self.total_flops

    def get_average_per_step(self) -> float:
        """Get average FLOPs per step."""
        if self.steps == 0:
            return 0
        return self.total_flops / self.steps

    def format_flops(self, flops: Optional[int] = None) -> str:
        """Format FLOPs in human-readable format."""
        if flops is None:
            flops = self.total_flops

        if flops >= 1e18:
            return f"{flops / 1e18:.2f} EFLOPs"
        elif flops >= 1e15:
            return f"{flops / 1e15:.2f} PFLOPs"
        elif flops >= 1e12:
            return f"{flops / 1e12:.2f} TFLOPs"
        elif flops >= 1e9:
            return f"{flops / 1e9:.2f} GFLOPs"
        elif flops >= 1e6:
            return f"{flops / 1e6:.2f} MFLOPs"
        else:
            return f"{flops} FLOPs"


def estimate_training_emissions(
    num_gpus: int,
    training_hours: float,
    gpu_tdp_watts: int = 300,
    carbon_intensity: float = 0.5,  # kg CO2/kWh
) -> Dict[str, float]:
    """
    Estimate training emissions.

    Args:
        num_gpus: Number of GPUs used
        training_hours: Total training hours
        gpu_tdp_watts: GPU TDP in watts
        carbon_intensity: Carbon intensity of electricity (kg CO2/kWh)

    Returns:
        Dictionary with emission estimates
    """
    # Calculate energy consumption
    total_power_kw = (num_gpus * gpu_tdp_watts) / 1000
    total_energy_kwh = total_power_kw * training_hours

    # Calculate emissions
    co2_kg = total_energy_kwh * carbon_intensity

    return {
        'num_gpus': num_gpus,
        'training_hours': training_hours,
        'total_power_kw': total_power_kw,
        'total_energy_kwh': total_energy_kwh,
        'co2_emissions_kg': co2_kg,
        'carbon_intensity_used': carbon_intensity,
    }

