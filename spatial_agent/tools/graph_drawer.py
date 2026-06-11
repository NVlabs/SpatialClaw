"""Graph drawer tool for time-series visualization.

Returns VisualFeedback (use ``show()`` to view it inline).
"""

from typing import Optional

import numpy as np
from PIL import Image

from spatial_agent.kernel_types.visual_feedback import VisualFeedback
from spatial_agent.tools.base import CPUTool


class GraphDrawer(CPUTool):
    """Matplotlib-based plotting tool.

    Usage::

        chart = tools.Graph.plot(distances, x_label="Frame", y_label="Distance (m)")
        show(chart)  # inspect the plot inline
    """

    TOOL_PROMPT_DESCRIPTION = """
### tools.Graph — Time-Series Plotting (CPU)

`tools.Graph.plot(values, validity=None, x_label="Frame", y_label="Value", title=None)` → `VisualFeedback`

Plots a **1D numpy array** as a line chart. Returns a `VisualFeedback` (use `show()` to view it).

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `values` | 1D array | required | The data to plot |
| `validity` | 1D bool array | `None` | Optional mask; `False` entries become gaps |
| `x_label` | str | `"Frame"` | X-axis label |
| `y_label` | str | `"Value"` | Y-axis label |
| `title` | str | `None` | Optional title |

The returned `VisualFeedback.description` includes min/max/mean/trend summary as text.

```python
distances = np.array([1.2, 1.5, 1.8, 2.1])
chart = tools.Graph.plot(distances, x_label="Frame", y_label="Distance (m)")
show(chart)  # inspect the plot yourself
print(chart.description)  # read the text summary
```
"""

    @staticmethod
    def plot(
        values: np.ndarray,
        validity: Optional[np.ndarray] = None,
        x_label: str = "Frame",
        y_label: str = "Value",
        title: Optional[str] = None,
    ) -> VisualFeedback:
        """Plot a 1D time series and return as VisualFeedback.

        Args:
            values: ``(T,)`` array of values.
            validity: Optional ``(T,)`` boolean mask (gaps shown as breaks).
            x_label: X-axis label.
            y_label: Y-axis label.
            title: Optional title.

        Returns:
            ``VisualFeedback`` with the plot image and a text description.
        """
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 0:
            raise ValueError(
                "`values` must be a 1D array, got scalar. "
                "Wrap in a list: np.array([value])"
            )
        if values.ndim != 1:
            raise ValueError(
                f"`values` must be 1D (shape (T,)), got shape {values.shape}. "
                f"If 2D, select one column: values[:, col_idx]"
            )
        if len(values) == 0:
            raise ValueError("`values` is empty — nothing to plot.")

        if validity is not None:
            validity = np.asarray(validity, dtype=bool)
            if validity.shape != values.shape:
                raise ValueError(
                    f"`validity` shape {validity.shape} must match `values` shape {values.shape}."
                )

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        T = len(values)
        x = np.arange(T)

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        if validity is not None:
            masked_values = np.where(validity, values, np.nan)
            ax.plot(x, masked_values, "-o", markersize=2, linewidth=1.5)
        else:
            ax.plot(x, values, "-o", markersize=2, linewidth=1.5)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # Convert to PIL
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = Image.frombuffer(
            "RGBA",
            fig.canvas.get_width_height(),
            buf,
        ).convert("RGB")
        plt.close(fig)

        # Build text description
        valid_vals = values if validity is None else values[validity]
        if len(valid_vals) > 0:
            desc = (
                f"Plot of {y_label} vs {x_label}: "
                f"min={valid_vals.min():.4f}, max={valid_vals.max():.4f}, "
                f"mean={valid_vals.mean():.4f}, "
                f"first={valid_vals[0]:.4f}, last={valid_vals[-1]:.4f}"
            )
            # Trend
            if len(valid_vals) > 1:
                diff = valid_vals[-1] - valid_vals[0]
                if abs(diff) < 0.01 * (abs(valid_vals.max() - valid_vals.min()) + 1e-8):
                    desc += ", trend=stable"
                elif diff > 0:
                    desc += ", trend=increasing"
                else:
                    desc += ", trend=decreasing"
        else:
            desc = f"Plot of {y_label} vs {x_label}: no valid data"

        return VisualFeedback(
            image=img,
            source="GraphDrawer.plot",
            description=desc,
        )

    def __repr__(self) -> str:
        return "GraphDrawer(method: plot)"
