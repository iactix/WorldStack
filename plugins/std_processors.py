from plugin import GeneratorModule
import numpy as np

class Combiner(GeneratorModule):
    def init(self):
        self.set_type("combiner", "processor")
        self.create_setting("in1", "", "First input image", "input")
        self.create_setting("in2", "", "Second input image", "input")
        self.create_setting("out", "", "Image output name", "output")
        self.create_setting("mode", "add", "Combination mode: add, subtract, multiply, divide, higher, lower")
        self.create_setting("limit_low", 0, "Minimum brightness limit")
        self.create_setting("limit_high", 255, "Maximum brightness limit")
        self.create_setting("clip", True, "Whether to clip the output within limits")
        return "Combines two input images using basic mathematical operations. Example: out = in1 - in2"

    def apply(self, map_width, map_height, settings, inputs, rng):
        # Handle parameters
        mode = settings["mode"]
        limit_low = settings["limit_low"]
        limit_high = settings["limit_high"]
        clip = settings["clip"]

        # Retrieve inputs
        in1 = inputs.get("in1", np.zeros((map_height, map_width), dtype=np.float32))
        in2 = inputs.get("in2", np.zeros((map_height, map_width), dtype=np.float32))

        # Process
        if mode == "add":
            map_out = in1 + in2
        elif mode == "subtract":
            map_out = in1 - in2
        elif mode == "multiply":
            map_out = in1 * in2
        elif mode == "divide":
            with np.errstate(divide="ignore", invalid="ignore"):  # Handle division by zero safely
                map_out = np.where(in2 != 0, in1 / in2, 0)
        elif mode == "higher":
            map_out = np.maximum(in1, in2)
        elif mode == "lower":
            map_out = np.minimum(in1, in2)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Apply clipping if enabled
        if clip:
            map_out = np.clip(map_out, limit_low, limit_high)

        return {"out": map_out}

