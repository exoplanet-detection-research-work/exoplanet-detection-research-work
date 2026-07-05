"""Multi-sector stitching of TESS light curves.

TESS observes a target in one or more ~27-day sectors, each with an
arbitrary instrumental flux offset. Stitching brings all sectors onto
a common relative scale by dividing each sector by its own median
flux, then time-orders the combined series.

Two entry points are provided:
    * :meth:`SectorStitcher.stitch` combines any number of separate
      per-sector :class:`~exodet.data.base.LightCurve` objects into one.
    * :meth:`SectorStitcher.apply` (the pipeline interface) normalizes
      a single curve that already carries a per-cadence
      ``meta["sector"]`` array.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from exodet.data.base import LightCurve
from exodet.exceptions import PipelineError
from exodet.preprocessing.base import PREPROCESSORS, BasePreprocessor
from exodet.preprocessing.common import PER_CADENCE_META_KEYS

__all__ = ["SectorStitcher"]

logger = logging.getLogger(__name__)


@PREPROCESSORS.register("sector_stitch")
class SectorStitcher(BasePreprocessor):
    """Normalizes per-sector flux offsets and time-orders the curve.

    Each sector's flux (and uncertainties) is divided by that sector's
    median flux, removing arbitrary inter-sector offsets while
    preserving relative transit depths. The applied medians are stored
    in ``meta["sector_medians"]`` for full reversibility.

    Curves without a per-cadence ``meta["sector"]`` array are treated
    as single-sector observations.
    """

    def apply(self, light_curve: LightCurve) -> LightCurve:
        """Stitches the sectors of a combined multi-sector curve.

        Args:
            light_curve: Curve whose ``meta["sector"]`` (if present)
                assigns a sector number to every cadence.

        Returns:
            A time-sorted curve with per-sector median normalization
            applied and sector medians recorded in metadata.

        Raises:
            PipelineError: If a sector's median flux is zero or not
                finite, making normalization impossible.
        """
        n = len(light_curve)
        sector_meta = light_curve.meta.get("sector")
        if sector_meta is None:
            sectors = np.zeros(n, dtype=np.int64)
        else:
            sectors = np.asarray(sector_meta, dtype=np.int64)
            if sectors.shape != (n,):
                raise PipelineError(
                    f"meta['sector'] shape {sectors.shape} does not match "
                    f"curve length {n}."
                )

        order = np.argsort(light_curve.time, kind="stable")
        time = light_curve.time[order]
        flux = light_curve.flux[order].copy()
        flux_err = (
            None if light_curve.flux_err is None else light_curve.flux_err[order].copy()
        )
        sectors = sectors[order]

        unique_sectors = np.unique(sectors)
        medians: dict[int, float] = {}
        # A handful of sectors at most, so a per-sector loop costs nothing;
        # the inner operations are all vectorized.
        for sector in unique_sectors:
            in_sector = sectors == sector
            median = float(np.nanmedian(flux[in_sector]))
            if not np.isfinite(median) or median == 0.0:
                raise PipelineError(
                    f"Target {light_curve.target_id}: sector {int(sector)} has "
                    f"non-finite or zero median flux ({median}); cannot stitch."
                )
            flux[in_sector] /= median
            if flux_err is not None:
                flux_err[in_sector] /= abs(median)
            medians[int(sector)] = median

        step = f"{self.name}(n_sectors={unique_sectors.size})"
        result = light_curve.replace_flux(
            flux, step_name=step, time=time, flux_err=flux_err
        )
        for key in PER_CADENCE_META_KEYS:
            value = result.meta.get(key)
            if isinstance(value, np.ndarray) and value.shape == (n,):
                result.meta[key] = value[order]
        result.meta["sector"] = sectors
        result.meta["sector_medians"] = medians
        logger.info(
            "Target %s: stitched %d sector(s) (medians: %s).",
            light_curve.target_id,
            unique_sectors.size,
            medians,
        )
        return result

    @classmethod
    def stitch(cls, curves: Sequence[LightCurve]) -> LightCurve:
        """Combines separate per-sector curves into one stitched curve.

        Each input curve should identify its sector via a scalar
        ``meta["sector"]`` entry; missing entries are numbered by
        position. Provenance of every input (target, sector, size, and
        prior history) is preserved in ``meta["stitched_from"]``.

        Args:
            curves: One or more single-sector light curves of the same
                target.

        Returns:
            A single stitched, time-sorted, sector-normalized curve.

        Raises:
            PipelineError: If no curves are given or targets differ.
        """
        if not curves:
            raise PipelineError("Cannot stitch an empty sequence of curves.")
        target_ids = {curve.target_id for curve in curves}
        if len(target_ids) > 1:
            raise PipelineError(
                f"Refusing to stitch different targets: {sorted(target_ids)}."
            )

        has_err = all(curve.flux_err is not None for curve in curves)
        sector_ids = [
            int(curve.meta.get("sector", index)) for index, curve in enumerate(curves)
        ]
        combined = LightCurve(
            target_id=curves[0].target_id,
            time=np.concatenate([curve.time for curve in curves]),
            flux=np.concatenate([curve.flux for curve in curves]),
            flux_err=(
                np.concatenate([curve.flux_err for curve in curves])  # type: ignore[misc]
                if has_err
                else None
            ),
            label=curves[0].label,
            mission=curves[0].mission,
            meta={
                "sector": np.repeat(sector_ids, [len(curve) for curve in curves]),
                "stitched_from": [
                    {
                        "target_id": curve.target_id,
                        "sector": sector,
                        "n_points": len(curve),
                        "history": list(curve.history),
                    }
                    for curve, sector in zip(curves, sector_ids)
                ],
            },
        )
        if all(
            isinstance(curve.meta.get("quality"), np.ndarray) for curve in curves
        ):
            combined.meta["quality"] = np.concatenate(
                [curve.meta["quality"] for curve in curves]
            )
        return cls().apply(combined)
