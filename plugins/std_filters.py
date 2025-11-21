from plugin import GeneratorModule
import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter

class LevelMapper(GeneratorModule):
    def init(self):
        self.set_type("level_mapper", "filter")
        self.create_setting("in", "", "Image input name", "input")
        self.create_setting("out", "", "Image output name", "output")
        self.create_setting("input_low", 0, "Input brightness that will be mapped to output_low.")
        self.create_setting("input_high", 255, "Input brightness that will be mapped to output_high.")
        self.create_setting("output_low", 0, "Output brightness that input_low will become.")
        self.create_setting("output_high", 255, "Output brightness that input_high will become.")
        self.create_setting("clip", True, "Whether to clip the brightness outside of input range.")
        
        return "Maps brightness range specified for input to brightness range specified for output."
    
    def apply(self, map_width, map_height, settings, inputs, rng):
        # Handle Parameters
        input_low = settings["input_low"]
        input_high = settings["input_high"]
        output_low = settings["output_low"]
        output_high = settings["output_high"]
        clip = settings["clip"]
        
        input_range = input_high - input_low
        output_range = output_high - output_low

        # Process
        tmp = ((inputs["in"] - input_low) / input_range)
        if clip:
            np.clip(tmp, 0, 1, out=tmp)
        return {"out": tmp * output_range + output_low}
    
class GradientMapper(GeneratorModule):
    def init(self):
        self.set_type("gradient_mapper", "filter")
        self.create_setting("in", "", "Image input name", "input")
        self.create_setting("out", "", "Image output name", "output")
        self.create_setting("gradient", [0, 1, 2, 3], "JSON array of custom size, defining what output brightness the input brightness will be mapped to, from darkest to brightest input.")
        return "Maps brightness of input to the specified gradient. The darkest input value will become the first gradient value and the brightest will become the last. Everything in between will be mapped accordingly."

    def apply(self, map_width, map_height, settings, inputs, rng):
        # Handle Parameters
        gradient = np.asarray(settings["gradient"])
        
        h_max = inputs["in"].max()
        h_min = inputs["in"].min()
        if h_max == h_min:
            h_max += 1
            h_min -= 1

        # Process
        indices = ((inputs["in"] - h_min) / (h_max - h_min)) * len(gradient)
        indices = np.clip(indices.astype(int), 0, len(gradient) - 1)

        return {"out": gradient[indices]}

class CurveMapper(GeneratorModule):
    def init(self):
        self.set_type("curve_mapper", "filter")
        self.create_setting("in", "", "Image input name", "input")
        self.create_setting("out", "", "Image output name", "output")

        # Input/output range like LevelMapper
        self.create_setting("input_low", 0, "Input brightness that maps to 0 in the curve.")
        self.create_setting("input_high", 255, "Input brightness that maps to 255 in the curve.")
        self.create_setting(
            "auto_input_range",
            False,
            "If true, ignore input_low/high and use min/max of the input image."
        )
        self.create_setting("clip", True, "Whether to clip brightness outside of input range.")

        self.create_setting("output_low", 0, "Output brightness for curve value 0.")
        self.create_setting("output_high", 255, "Output brightness for curve value 255.")

        # Curve definition: list of [source, target] points in 0–255
        self.create_setting(
            "curve_points",
            [[0, 0], [127, 127], [255, 255]],
            "Curve defined as list of [source, target] points in 0–255."
        )

        # Smoothness: 0 = piecewise-linear, 1 = cubic Hermite (Bezier-like) through all points
        self.create_setting(
            "smoothness",
            0.0,
            "Curve smoothing amount (0 = linear between points, 1 = fully smoothed)."
        )

        # If true, output an image of the curve instead of transforming the input
        self.create_setting(
            "output_curve",
            False,
            "If true, output an image visualizing the curve instead of the transformed input."
        )

        return "Maps brightness via a user-defined curve with optional smoothing and auto input range."

    # --- Curve construction: piecewise linear + cubic Hermite blend ---

    def _build_lut(self, curve_points, smoothness):
        # Fallback if something weird comes in
        if not curve_points:
            curve_points = [[0, 0], [255, 255]]

        pts = np.asarray(curve_points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            pts = np.array([[0, 0], [255, 255]], dtype=np.float32)

        xs = pts[:, 0]
        ys = pts[:, 1]

        # Clamp to [0, 255]
        xs = np.clip(xs, 0.0, 255.0)
        ys = np.clip(ys, 0.0, 255.0)

        # Ensure endpoints 0 and 255 exist, mapping to identity by default
        if not np.any(xs == 0.0):
            xs = np.concatenate(([0.0], xs))
            ys = np.concatenate(([0.0], ys))
        if not np.any(xs == 255.0):
            xs = np.concatenate((xs, [255.0]))
            ys = np.concatenate((ys, [255.0]))

        # Sort by source value
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]

        # Deduplicate xs (keep last value for each x)
        uniq_xs = [xs[0]]
        uniq_ys = [ys[0]]
        for x, y in zip(xs[1:], ys[1:]):
            if x == uniq_xs[-1]:
                uniq_ys[-1] = y
            else:
                uniq_xs.append(x)
                uniq_ys.append(y)

        xs = np.array(uniq_xs, dtype=np.float32)
        ys = np.array(uniq_ys, dtype=np.float32)

        # Base piecewise-linear LUT evaluated at integer 0..255 (this already passes through all points)
        x_int = np.arange(256, dtype=np.float32)
        lut_linear = np.interp(x_int, xs, ys).astype(np.float32)

        smoothness = float(smoothness)
        if smoothness <= 0.0 or len(xs) < 3:
            return lut_linear

        # Cubic Hermite spline that passes smoothly through all points
        n = len(xs)
        # If only 2 points, cubic degenerates to linear
        if n == 2:
            return lut_linear

        # Tangents at each knot (standard C1 Hermite / "auto Bezier" style)
        m = np.zeros(n, dtype=np.float32)

        # Endpoint tangents: one-sided differences
        dx0 = xs[1] - xs[0]
        dxn = xs[-1] - xs[-2]
        m[0] = (ys[1] - ys[0]) / dx0 if dx0 != 0 else 0.0
        m[-1] = (ys[-1] - ys[-2]) / dxn if dxn != 0 else 0.0

        # Interior tangents: centered differences
        for i in range(1, n - 1):
            dx = xs[i + 1] - xs[i - 1]
            if dx == 0.0:
                m[i] = 0.0
            else:
                m[i] = (ys[i + 1] - ys[i - 1]) / dx

        # Evaluate cubic Hermite at x = 0..255
        X = x_int

        # Segment index for each X: xs[i] <= X < xs[i+1], last point included in last segment
        seg = np.searchsorted(xs, X, side="right") - 1
        seg = np.clip(seg, 0, n - 2)

        x0 = xs[seg]
        x1 = xs[seg + 1]
        h = x1 - x0

        # Normalized parameter within segment
        t = (X - x0) / h
        t = np.clip(t, 0.0, 1.0)

        t2 = t * t
        t3 = t2 * t

        # Hermite basis
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2

        y0 = ys[seg]
        y1 = ys[seg + 1]
        m0 = m[seg]
        m1 = m[seg + 1]
        Delta = h

        lut_cubic = (
            h00 * y0 +
            h10 * Delta * m0 +
            h01 * y1 +
            h11 * Delta * m1
        )

        lut_cubic = np.clip(lut_cubic, 0.0, 255.0).astype(np.float32)

        # Blend linear vs cubic: 0 = pure linear, 1 = pure smooth cubic
        alpha = max(0.0, min(1.0, smoothness))
        lut = (1.0 - alpha) * lut_linear + alpha * lut_cubic
        lut = np.clip(lut, 0.0, 255.0).astype(np.float32)

        return lut

    def _eval_lut_norm(self, norm_vals, lut):
        """
        Evaluate the LUT for normalized inputs in [0,1], using the same logic
        as for the actual image transform.
        """
        idx = np.clip(norm_vals, 0.0, 1.0) * 255.0
        idx_floor = np.floor(idx).astype(np.int32)
        idx_ceil = np.clip(idx_floor + 1, 0, 255)
        frac = idx - idx_floor

        return lut[idx_floor] * (1.0 - frac) + lut[idx_ceil] * frac

    def _apply_curve_lut(self, norm_img, lut):
        """
        norm_img: input normalized to [0, 1]
        lut: mapping for indices 0..255 (values in 0..255)
        returns: mapped values still in [0, 255] (float)
        """
        return self._eval_lut_norm(norm_img, lut)

    def _thicken_line(self, img, radius):
        """
        Isotropic thickening of a 1px line using iterative 3x3 max filter.
        radius: roughly how many pixels to expand outward.
        """
        if radius <= 0:
            return img

        img = img.copy()
        for _ in range(radius):
            base = img.copy()
            # center already in base
            # up
            img[:-1, :] = np.maximum(img[:-1, :], base[1:, :])
            # down
            img[1:, :] = np.maximum(img[1:, :], base[:-1, :])
            # left
            img[:, :-1] = np.maximum(img[:, :-1], base[:, 1:])
            # right
            img[:, 1:] = np.maximum(img[:, 1:], base[:, :-1])
            # diagonals
            img[:-1, :-1] = np.maximum(img[:-1, :-1], base[1:, 1:])
            img[1:, :-1] = np.maximum(img[1:, :-1], base[:-1, 1:])
            img[:-1, 1:] = np.maximum(img[:-1, 1:], base[1:, :-1])
            img[1:, 1:] = np.maximum(img[1:, 1:], base[:-1, :-1])
        return img

    def _render_curve_image(self, map_width, map_height, lut):
        """
        Render the curve into an image of size (map_height, map_width).
        X-axis: input brightness 0..255
        Y-axis: output brightness 0..255
        White, relatively thick line on black background.
        Uses exactly the same LUT evaluation as the real transform.
        """
        width = map_width
        height = map_height

        # Normalized input across the width
        if width > 1:
            x_norm = np.linspace(0.0, 1.0, width, dtype=np.float32)
        else:
            x_norm = np.array([0.0], dtype=np.float32)

        ys = self._eval_lut_norm(x_norm, lut)  # 0..255

        # Convert ys to row indices (0 at top, 255 at bottom)
        rows = (height - 1 - (ys / 255.0) * (height - 1)).astype(np.int32)
        rows = np.clip(rows, 0, height - 1)

        curve_img = np.zeros((height, width), dtype=np.float32)
        cols = np.arange(width, dtype=np.int32)

        # Draw base 1px line
        curve_img[rows, cols] = 255.0

        # Choose thickness based on image size so it remains visible when downscaled
        base = min(width, height)
        thickness = max(1, int(round(base / 64.0)))  # e.g. 256 -> ~4px, 512 -> ~8px
        thickness = min(thickness, 10)               # clamp so kernel stays reasonable
        radius = max(0, thickness // 2)

        if radius > 0:
            curve_img = self._thicken_line(curve_img, radius)

        return curve_img

    # --- Main entry point ---

    def apply(self, map_width, map_height, settings, inputs, rng):
        img = inputs["in"].astype(np.float32)

        input_low = float(settings["input_low"])
        input_high = float(settings["input_high"])
        auto_input_range = bool(settings["auto_input_range"])
        clip = bool(settings["clip"])
        output_low = float(settings["output_low"])
        output_high = float(settings["output_high"])
        curve_points = settings["curve_points"]
        smoothness = settings["smoothness"]
        output_curve = bool(settings["output_curve"])

        # Auto input range based on actual image min/max
        if auto_input_range:
            img_min = float(np.min(img))
            img_max = float(np.max(img))
            if img_max > img_min:
                input_low = img_min
                input_high = img_max

        input_range = input_high - input_low
        if input_range == 0.0:
            norm = np.zeros_like(img, dtype=np.float32)
        else:
            norm = (img - input_low) / input_range

        if clip:
            np.clip(norm, 0.0, 1.0, out=norm)

        # Build LUT for 0..255 -> 0..255
        lut = self._build_lut(curve_points, smoothness)

        if output_curve:
            out_img = self._render_curve_image(map_width, map_height, lut)
            return {"out": out_img}

        # Apply curve in 0..255 domain (still float)
        mapped_0_255 = self._apply_curve_lut(norm, lut)

        # Back to [0, 1]
        mapped_norm = mapped_0_255 / 255.0

        output_range = output_high - output_low
        out = mapped_norm * output_range + output_low

        return {"out": out}

class Blur(GeneratorModule):
    def init(self):
        self.set_type("blur", "filter")
        self.create_setting("in", "", "Image input name", "input")
        self.create_setting("out", "", "Image output name", "output")
        self.create_setting("passes", 1, "Number of blur iterations")
        self.create_setting("strength", 1, "Blur intensity (for gaussian and median modes)")
        self.create_setting("mode", "gaussian", "Blur mode: convolve, gaussian, median")
        return "Applies a blur effect to the input image using different blur techniques."

    def apply(self, map_width, map_height, settings, inputs, rng):
        # Handle parameters
        passes = settings["passes"]
        strength = settings["strength"]
        mode = settings["mode"]

        # Retrieve input
        map_src = inputs.get("in", np.full((map_height, map_width), 0.0, dtype=np.float32))
        map_out = np.zeros_like(map_src)

        # 3x3 Averaging Kernel
        kernel = np.ones((3, 3)) / 9.0

        # Process
        for i in range(passes):
            if i == 0:
                np.copyto(map_out, map_src)

            if mode == "convolve":
                map_out = convolve(map_out, kernel, mode="nearest")
            elif mode == "gaussian":
                map_out = gaussian_filter(map_out, sigma=strength)
            elif mode == "median":
                map_out = median_filter(map_out, size=int(strength))
            else:
                raise ValueError(f"Unknown mode: {mode}")

        return {"out": map_out}

class ModuleDespiker(GeneratorModule):
    def init(self):
        self.set_type("despiker", "filter")
        self.create_setting("in", "", "Image input name", "input")
        self.create_setting("out", "", "Image output name", "output")
        self.create_setting("passes", 1, "Number of despike iterations")
        self.create_setting("majority", 5, "Minimum count of identical neighbours required to replace a pixel with their value")
        return "Removes isolated pixel spikes by replacing them with the most common surrounding value."

    def apply(self, map_width, map_height, settings, inputs, rng):
        # Handle parameters
        passes = settings["passes"]
        majority = settings["majority"]

        # Retrieve input
        map_src = inputs.get("in", np.full((map_height, map_width), 0.0, dtype=np.float32))
        map_out = np.copy(map_src)

        for i in range(passes):
            # Swap buffers if needed
            if i > 0:
                map_src, map_out = map_out, map_src

            # Extract 8-neighborhoods efficiently using np.roll()
            neighbors = np.stack([
                np.roll(np.roll(map_src, dr, axis=0), dc, axis=1)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                               (-1, -1), (-1, 1), (1, -1), (1, 1)]
            ], axis=0)  # Shape: (8, height, width)

            # Count occurrences of each value at every pixel position
            unique_vals = np.unique(neighbors)  # Get unique possible values
            counts = np.zeros((len(unique_vals), map_height, map_width), dtype=np.int32)  # Store counts

            for idx, val in enumerate(unique_vals):
                counts[idx] = np.sum(neighbors == val, axis=0)  # Count occurrences per pixel

            # Find the most common value for each pixel
            max_idx = np.argmax(counts, axis=0)  # Index of most common value per pixel
            most_common_value = unique_vals[max_idx]  # Extract actual values
            most_common_count = np.max(counts, axis=0)  # Get max count per pixel

            # Apply majority rule condition (only for inner region)
            mask = (most_common_count >= majority) & (map_src != most_common_value)

            # Apply updates **only to the inner part**
            map_out[1:-1, 1:-1] = np.where(mask[1:-1, 1:-1], most_common_value[1:-1, 1:-1], map_src[1:-1, 1:-1])

            # Count number of changes
            changes = np.count_nonzero(mask[1:-1, 1:-1])

            if changes == 0:
                break  # Stop early if stable

        return {"out": map_out}
