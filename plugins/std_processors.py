from plugin import GeneratorModule
import numpy as np
from scipy.ndimage import affine_transform

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

class Transform(GeneratorModule):
    def init(self):
        self.set_type("transform", "processor")
        self.create_setting("in", "", "Image input name", "input")
        self.create_setting("out", "", "Image output name", "output")

        self.create_setting("move_x", 0.0, "Move image right/left (pixels, right = positive)")
        self.create_setting("move_y", 0.0, "Move image down/up (pixels, down = positive)")
        self.create_setting("scale_x", 1.0, "Scale image horizontally (1 = unchanged)")
        self.create_setting("scale_y", 1.0, "Scale image vertically (1 = unchanged)")
        self.create_setting("rotate", 0.0, "Rotate image around center (degrees counter-clockwise)")
        self.create_setting("background", 0.0, "Background brightness for areas outside the image")

        return "Scales, rotates, and moves the image. Order of operations: scale → rotate → move."

    def apply(self, map_width, map_height, settings, inputs, rng):
        map_src = inputs.get("in", np.full((map_height, map_width), 0.0, dtype=np.float32))

        move_x = settings["move_x"]
        move_y = settings["move_y"]
        scale_x = settings["scale_x"]
        scale_y = settings["scale_y"]
        angle_deg = settings["rotate"]
        background = settings["background"]

        if (
            move_x == 0.0 and move_y == 0.0 and
            scale_x == 1.0 and scale_y == 1.0 and
            angle_deg == 0.0
        ):
            return {"out": map_src.copy()}

        angle_rad = -np.deg2rad(angle_deg)  # negative for counter-clockwise rotation
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        matrix = np.array([
            [cos_a / scale_y, -sin_a / scale_x],
            [sin_a / scale_y,  cos_a / scale_x],
        ])

        center_y = map_height / 2.0
        center_x = map_width / 2.0

        offset = np.array([
            center_y - (matrix[0, 0] * center_y + matrix[0, 1] * center_x) - move_y,
            center_x - (matrix[1, 0] * center_y + matrix[1, 1] * center_x) - move_x,
        ])

        map_out = affine_transform(
            map_src,
            matrix=matrix,
            offset=offset,
            output_shape=(map_height, map_width),
            order=1,
            mode='constant',
            cval=background
        )

        return {"out": map_out}
