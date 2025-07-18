import base64
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _extract_numeric_array(container):
    """Return numpy array from scattergl packed array or list."""
    if isinstance(container, (list, tuple, np.ndarray)):
        return np.asarray(container)
    # Packed binary dict expected: keys bdata, dtype
    if isinstance(container, dict) and "bdata" in container:
        raw = base64.b64decode(container["bdata"])
        dtype = np.dtype(container.get("dtype", "f8"))
        shape = container.get("shape")
        arr = np.frombuffer(raw, dtype=dtype)
        if shape:
            arr = arr.reshape(shape)
        return arr
    raise TypeError("Unsupported container format for y data")


def test_scattergl_series_should_not_include_infinity():
    df = pd.DataFrame({
        "x": [0, 1, 2, 3, 4],
        "y": [0.0, 1.0, 4.0, 9.0, np.inf],
    })

    fig = go.Figure(go.Scattergl(x=df["x"], y=df["y"], mode="lines"))
    packed_y = fig.to_plotly_json()["data"][0]["y"]
    y_arr = _extract_numeric_array(packed_y)

    assert not any(math.isinf(v) for v in y_arr), (
        "Infinity value leaked into Scattergl y data when using pandas Series."
    )
