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
