from plugin import GeneratorModule
import numpy as np
from scipy.interpolate import CubicSpline
from skimage.draw import polygon, disk
from PIL import Image
import os
from plugin import GeneratorModule
import numpy as np
from scipy.interpolate import CubicSpline
from skimage.draw import polygon, disk
from PIL import Image
import os
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite, NoiseType

class PerlinNoiseGenerator(GeneratorModule):
    def init(self):
        self.set_type("perlin_noise", "generator")
        self.create_setting("out", "", "Image output name", "output")
        self.create_setting("scale", 30, "Scales the noise pattern; higher values result in wider waves.")
        self.create_setting("max", 255, "Maximum brightness")
        return "Generates a random Perlin noise pattern."

    def apply(self, map_width, map_height, settings, inputs, rng):
        scale = 1.0 / settings["scale"]
        half_noise_height = settings["max"] / 2
        map_out = np.zeros((map_height, map_width), dtype=np.float32)

        noise = FastNoiseLite(seed=rng.randint(0, 1000000))
        noise.noise_type = NoiseType.NoiseType_Perlin
        noise.frequency = scale  # Set frequency directly

        rndx = rng.randint(0, 1000000) / 100.0
        rndy = rng.randint(0, 1000000) / 100.0

        for i in range(map_height):
            for j in range(map_width):
                val = noise.get_noise(i + rndx, j + rndy)
                map_out[i][j] = (val + 1) * half_noise_height

        return {"out": map_out}

class SimplexNoiseGenerator(GeneratorModule):
    def init(self):
        self.set_type("simplex_noise", "generator")
        self.create_setting("out", "", "Image output name", "output")
        self.create_setting("scale", 30, "Scales the noise pattern; higher values result in wider waves.")
        self.create_setting("max", 255, "Maximum brightness")
        return "Generates a random Simplex noise pattern."

    def apply(self, map_width, map_height, settings, inputs, rng):
        scale = 1.0 / (settings["scale"] * 2)
        half_noise_height = settings["max"] / 2
        map_out = np.zeros((map_height, map_width), dtype=np.float32)

        noise = FastNoiseLite(seed=rng.randint(0, 1000000))
        noise.noise_type = NoiseType.NoiseType_OpenSimplex2
        noise.frequency = scale  # Set frequency directly

        rndx = rng.randint(0, 1000000) / 100.0
        rndy = rng.randint(0, 1000000) / 100.0

        for i in range(map_height):
            for j in range(map_width):
                val = noise.get_noise(i + rndx, j + rndy)
                map_out[i][j] = (val + 1) * half_noise_height

        return {"out": map_out}
    
class Circle(GeneratorModule):
    def init(self):
        self.set_type("circle", "generator")
        self.create_setting("out", "", "Image output name", "output")
        self.create_setting("outer_radius", 16, "At what radius the circle reaches outer_brightness")
        self.create_setting("inner_radius", 8, "At what radius the circle reaches inner_brightness")
        self.create_setting("outer_brightness", 0, "Brightness at outer_radius")
        self.create_setting("inner_brightness", 255, "Brightness at inner_radius")
        self.create_setting("pos_x", 32, "X position of the circle center")
        self.create_setting("pos_y", 32, "Y position of the circle center")
        return "Generates a circular gradient with configurable position and size."

    def apply(self, map_width, map_height, settings, inputs, rng):
        # Handle parameters
        r_out = settings["outer_radius"]
        r_in = settings["inner_radius"]
        h_out = settings["outer_brightness"]
        h_in = settings["inner_brightness"]
        px = settings["pos_x"]
        py = settings["pos_y"]

        # Generate coordinate grids
        y, x = np.ogrid[:map_height, :map_width]

        # Compute radial distance from (px, py)
        r = np.sqrt((y - py) ** 2 + (x - px) ** 2)

        # Clip and normalize
        r = np.clip(r, r_in, r_out)
        r = (r - r_in) / (r_out - r_in)

        # Compute brightness values
        map_out = r * (h_out - h_in) + h_in

        return {"out": map_out.astype(np.float32)}
    
class ImageLoader(GeneratorModule):
    def init(self):
        self.set_type("image_loader", "generator")
        self.create_setting("out", "", "Image output name", "output")
        self.create_setting("file", "undefined.png", "Filename of the image to load.")
        self.create_setting("mode", "fit", "How the image should be fitted into the map dimensions: fit, stretch, crop")
        self.create_setting("background", 0, "Background brightness for regions possibly not covered by the image file")
        return "Loads a png image from the 'images' directory."

    def apply(self, map_width, map_height, settings, inputs, rng):
        # Read settings
        filename = settings.get("file", "")
        mode = settings.get("mode", "fit").lower()
        bg = settings.get("background", 0)

        # Attempt to open image, fallback to background canvas
        try:
            img_path = os.path.join('images', filename)
            img = Image.open(img_path).convert('L')  # grayscale
            src_w, src_h = img.size
        except Exception:
            # Return solid background if load fails
            canvas = Image.new('L', (map_width, map_height), color=bg)
            map_out = np.array(canvas, dtype=np.float32)
            return {"out": map_out}

        # Process by mode
        if mode == 'stretch':
            # Resize to exactly fill the map
            out_img = img.resize((map_width, map_height), Image.BILINEAR)

        elif mode == 'fit':
            # Scale to fit inside, preserving aspect ratio
            scale = min(map_width / src_w, map_height / src_h)
            new_w, new_h = int(src_w * scale), int(src_h * scale)
            resized = img.resize((new_w, new_h), Image.BILINEAR)
            canvas = Image.new('L', (map_width, map_height), color=bg)
            offset_x = (map_width - new_w) // 2
            offset_y = (map_height - new_h) // 2
            canvas.paste(resized, (offset_x, offset_y))
            out_img = canvas

        elif mode == 'crop':
            # No scaling: small images are centered; large images are center-cropped
            if src_w > map_width or src_h > map_height:
                # Center-crop oversized image
                left = (src_w - map_width) // 2
                top = (src_h - map_height) // 2
                out_img = img.crop((left, top, left + map_width, top + map_height))
            else:
                # Place smaller image centered on background
                canvas = Image.new('L', (map_width, map_height), color=bg)
                offset_x = (map_width - src_w) // 2
                offset_y = (map_height - src_h) // 2
                canvas.paste(img, (offset_x, offset_y))
                out_img = canvas

        else:
            # Unknown mode: return background
            out_img = Image.new('L', (map_width, map_height), color=bg)

        map_out = np.array(out_img, dtype=np.float32)
        return {"out": map_out}

class Spline(GeneratorModule):
    def init(self):
        self.set_type("spline", "generator")
        self.create_setting("out", "", "Image output name", "output")
        self.create_setting("from_x", 0, "Starting X coordinate of the spline")
        self.create_setting("from_y", 0, "Starting Y coordinate of the spline")
        self.create_setting("to_x", 64, "Ending X coordinate of the spline")
        self.create_setting("to_y", 64, "Ending Y coordinate of the spline")
        self.create_setting("wiggle", 10, "Maximum deviation from the straight path")
        self.create_setting("segment_length", 30, "Length of each spline segment")
        self.create_setting("thickness", 3, "Thickness of the spline path")
        self.create_setting("brightness", 255, "Brightness value of the spline path")
        return "Generates a wiggly spline path that smoothly connects two points."

    def apply(self, map_width, map_height, settings, inputs, rng):
        """Applies a wiggly spline path to a 2D map, wiggling in both X and Y directions."""

        # Handle parameters
        p1x = settings["from_x"]
        p1y = settings["from_y"]
        p2x = settings["to_x"]
        p2y = settings["to_y"]
        wiggle = settings["wiggle"]
        seglen = settings["segment_length"]
        thickness = settings["thickness"]
        color = settings["brightness"]

        # Prepare output map
        map_out = np.full((map_height, map_width), 0.0, dtype=np.float32)

        # Configure wiggly path parameters
        num_points = max(abs(p2x - p1x), abs(p2y - p1y)) // seglen
        num_points = max(num_points, 3)  # Ensure minimum resolution

        # Generate intermediate points
        x_values = np.linspace(p1x, p2x, num_points)
        y_values = np.linspace(p1y, p2y, num_points)

        # Add random wiggles to both X and Y
        if num_points > 2:
            x_wiggle = [rng.randint(-wiggle, wiggle) for _ in range(num_points - 2)]
            y_wiggle = [rng.randint(-wiggle, wiggle) for _ in range(num_points - 2)]
            x_values[1:-1] += x_wiggle
            y_values[1:-1] += y_wiggle

        # Create smooth cubic splines
        t_values = np.linspace(0, 1, num_points)
        spline_x = CubicSpline(t_values, x_values, bc_type="natural")
        spline_y = CubicSpline(t_values, y_values, bc_type="natural")

        # Generate interpolated points
        t_interp = np.linspace(0, 1, num_points * 5)
        x_interp = spline_x(t_interp)
        y_interp = spline_y(t_interp)

        # Compute normal vectors for thickness
        dx = np.gradient(x_interp)
        dy = np.gradient(y_interp)
        length = np.hypot(dx, dy) + 1e-6  # Avoid division by zero
        nx, ny = -dy / length, dx / length  # Perpendicular unit vectors

        # Manually correct the first and last normal vectors
        nx[0], ny[0] = -dy[1] / np.hypot(dx[1], dy[1]), dx[1] / np.hypot(dx[1], dy[1])
        nx[-1], ny[-1] = -dy[-2] / np.hypot(dx[-2], dy[-2]), dx[-2] / np.hypot(dx[-2], dy[-2])

        # Compute adjusted thickness based on curvature
        thickness_factors = np.ones_like(t_interp)
        for i in range(1, len(t_interp) - 1):
            angle_change = np.arctan2(dy[i], dx[i]) - np.arctan2(dy[i - 1], dx[i - 1])
            if np.abs(angle_change) > np.pi / 4:  # Detect sharp turns
                thickness_factors[i] = 0.5  # Reduce thickness in tight curves

        # Compute offset points for thickness with curvature adjustments
        left_x = x_interp + nx * thickness * thickness_factors
        left_y = y_interp + ny * thickness * thickness_factors
        right_x = x_interp - nx * thickness * thickness_factors
        right_y = y_interp - ny * thickness * thickness_factors

        # Construct the final polygon outline with smooth transitions
        poly_x = np.concatenate((left_x, right_x[::-1]))
        poly_y = np.concatenate((left_y, right_y[::-1]))

        # Rasterize the polygon using skimage
        rr, cc = polygon(poly_y, poly_x, shape=map_out.shape)
        map_out[rr, cc] = color  # Fill the polygon

        # — Add circular end‑caps —
        radius = int(np.ceil(thickness))
        for (x_c, y_c) in [(x_interp[0], y_interp[0]), (x_interp[-1], y_interp[-1])]:
            rr_c, cc_c = disk((y_c, x_c), radius, shape=map_out.shape)
            map_out[rr_c, cc_c] = color

        return {"out": map_out}

class Circle(GeneratorModule):
    def init(self):
        self.set_type("circle", "generator")
        self.create_setting("out", "", "Image output name", "output")
        self.create_setting("outer_radius", 16, "At what radius the circle reaches outer_brightness")
        self.create_setting("inner_radius", 8, "At what radius the circle reaches inner_brightness")
        self.create_setting("outer_brightness", 0, "Brightness at outer_radius")
        self.create_setting("inner_brightness", 255, "Brightness at inner_radius")
        self.create_setting("pos_x", 32, "X position of the circle center")
        self.create_setting("pos_y", 32, "Y position of the circle center")
        return "Generates a circular gradient with configurable position and size."

    def apply(self, map_width, map_height, settings, inputs, rng):
        # Handle parameters
        r_out = settings["outer_radius"]
        r_in = settings["inner_radius"]
        h_out = settings["outer_brightness"]
        h_in = settings["inner_brightness"]
        px = settings["pos_x"]
        py = settings["pos_y"]

        # Generate coordinate grids
        y, x = np.ogrid[:map_height, :map_width]

        # Compute radial distance from (px, py)
        r = np.sqrt((y - py) ** 2 + (x - px) ** 2)

        # Clip and normalize
        r = np.clip(r, r_in, r_out)
        r = (r - r_in) / (r_out - r_in)

        # Compute brightness values
        map_out = r * (h_out - h_in) + h_in

        return {"out": map_out.astype(np.float32)}
    
class ImageLoader(GeneratorModule):
    def init(self):
        self.set_type("image_loader", "generator")
        self.create_setting("out", "", "Image output name", "output")
        self.create_setting("file", "undefined.png", "Filename of the image to load.")
        self.create_setting("mode", "fit", "How the image should be fitted into the map dimensions: fit, stretch, crop")
        self.create_setting("background", 0, "Background brightness for regions possibly not covered by the image file")
        return "Loads a png image from the 'images' directory."

    def apply(self, map_width, map_height, settings, inputs, rng):
        # Read settings
        filename = settings.get("file", "")
        mode = settings.get("mode", "fit").lower()
        bg = settings.get("background", 0)

        # Attempt to open image, fallback to background canvas
        try:
            img_path = os.path.join('images', filename)
            img = Image.open(img_path).convert('L')  # grayscale
            src_w, src_h = img.size
        except Exception:
            # Return solid background if load fails
            canvas = Image.new('L', (map_width, map_height), color=bg)
            map_out = np.array(canvas, dtype=np.float32)
            return {"out": map_out}

        # Process by mode
        if mode == 'stretch':
            # Resize to exactly fill the map
            out_img = img.resize((map_width, map_height), Image.BILINEAR)

        elif mode == 'fit':
            # Scale to fit inside, preserving aspect ratio
            scale = min(map_width / src_w, map_height / src_h)
            new_w, new_h = int(src_w * scale), int(src_h * scale)
            resized = img.resize((new_w, new_h), Image.BILINEAR)
            canvas = Image.new('L', (map_width, map_height), color=bg)
            offset_x = (map_width - new_w) // 2
            offset_y = (map_height - new_h) // 2
            canvas.paste(resized, (offset_x, offset_y))
            out_img = canvas

        elif mode == 'crop':
            # No scaling: small images are centered; large images are center-cropped
            if src_w > map_width or src_h > map_height:
                # Center-crop oversized image
                left = (src_w - map_width) // 2
                top = (src_h - map_height) // 2
                out_img = img.crop((left, top, left + map_width, top + map_height))
            else:
                # Place smaller image centered on background
                canvas = Image.new('L', (map_width, map_height), color=bg)
                offset_x = (map_width - src_w) // 2
                offset_y = (map_height - src_h) // 2
                canvas.paste(img, (offset_x, offset_y))
                out_img = canvas

        else:
            # Unknown mode: return background
            out_img = Image.new('L', (map_width, map_height), color=bg)

        map_out = np.array(out_img, dtype=np.float32)
        return {"out": map_out}

class Spline(GeneratorModule):
    def init(self):
        self.set_type("spline", "generator")
        self.create_setting("out", "", "Image output name", "output")
        self.create_setting("from_x", 0, "Starting X coordinate of the spline")
        self.create_setting("from_y", 0, "Starting Y coordinate of the spline")
        self.create_setting("to_x", 64, "Ending X coordinate of the spline")
        self.create_setting("to_y", 64, "Ending Y coordinate of the spline")
        self.create_setting("wiggle", 10, "Maximum deviation from the straight path")
        self.create_setting("segment_length", 30, "Length of each spline segment")
        self.create_setting("thickness", 3, "Thickness of the spline path")
        self.create_setting("brightness", 255, "Brightness value of the spline path")
        return "Generates a wiggly spline path that smoothly connects two points."

    def apply(self, map_width, map_height, settings, inputs, rng):
        """Applies a wiggly spline path to a 2D map, wiggling in both X and Y directions."""

        # Handle parameters
        p1x = settings["from_x"]
        p1y = settings["from_y"]
        p2x = settings["to_x"]
        p2y = settings["to_y"]
        wiggle = settings["wiggle"]
        seglen = settings["segment_length"]
        thickness = settings["thickness"]
        color = settings["brightness"]

        # Prepare output map
        map_out = np.full((map_height, map_width), 0.0, dtype=np.float32)

        # Configure wiggly path parameters
        num_points = max(abs(p2x - p1x), abs(p2y - p1y)) // seglen
        num_points = max(num_points, 3)  # Ensure minimum resolution

        # Generate intermediate points
        x_values = np.linspace(p1x, p2x, num_points)
        y_values = np.linspace(p1y, p2y, num_points)

        # Add random wiggles to both X and Y
        if num_points > 2:
            x_wiggle = [rng.randint(-wiggle, wiggle) for _ in range(num_points - 2)]
            y_wiggle = [rng.randint(-wiggle, wiggle) for _ in range(num_points - 2)]
            x_values[1:-1] += x_wiggle
            y_values[1:-1] += y_wiggle

        # Create smooth cubic splines
        t_values = np.linspace(0, 1, num_points)
        spline_x = CubicSpline(t_values, x_values, bc_type="natural")
        spline_y = CubicSpline(t_values, y_values, bc_type="natural")

        # Generate interpolated points
        t_interp = np.linspace(0, 1, num_points * 5)
        x_interp = spline_x(t_interp)
        y_interp = spline_y(t_interp)

        # Compute normal vectors for thickness
        dx = np.gradient(x_interp)
        dy = np.gradient(y_interp)
        length = np.hypot(dx, dy) + 1e-6  # Avoid division by zero
        nx, ny = -dy / length, dx / length  # Perpendicular unit vectors

        # Manually correct the first and last normal vectors
        nx[0], ny[0] = -dy[1] / np.hypot(dx[1], dy[1]), dx[1] / np.hypot(dx[1], dy[1])
        nx[-1], ny[-1] = -dy[-2] / np.hypot(dx[-2], dy[-2]), dx[-2] / np.hypot(dx[-2], dy[-2])

        # Compute adjusted thickness based on curvature
        thickness_factors = np.ones_like(t_interp)
        for i in range(1, len(t_interp) - 1):
            angle_change = np.arctan2(dy[i], dx[i]) - np.arctan2(dy[i - 1], dx[i - 1])
            if np.abs(angle_change) > np.pi / 4:  # Detect sharp turns
                thickness_factors[i] = 0.5  # Reduce thickness in tight curves

        # Compute offset points for thickness with curvature adjustments
        left_x = x_interp + nx * thickness * thickness_factors
        left_y = y_interp + ny * thickness * thickness_factors
        right_x = x_interp - nx * thickness * thickness_factors
        right_y = y_interp - ny * thickness * thickness_factors

        # Construct the final polygon outline with smooth transitions
        poly_x = np.concatenate((left_x, right_x[::-1]))
        poly_y = np.concatenate((left_y, right_y[::-1]))

        # Rasterize the polygon using skimage
        rr, cc = polygon(poly_y, poly_x, shape=map_out.shape)
        map_out[rr, cc] = color  # Fill the polygon

        # — Add circular end‑caps —
        radius = int(np.ceil(thickness))
        for (x_c, y_c) in [(x_interp[0], y_interp[0]), (x_interp[-1], y_interp[-1])]:
            rr_c, cc_c = disk((y_c, x_c), radius, shape=map_out.shape)
            map_out[rr_c, cc_c] = color

        return {"out": map_out}
