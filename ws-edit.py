#!/usr/bin/env python3
# This is terrible, sloppy code that was cobbled together to get things done.
# Lots of workarounds to make this work with tkinter, which is inappropriate for rendering the main view

import os, json, subprocess, tkinter as tk, math
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import sys
import threading
import textwrap

# Setup
generator_file = "ws-gen.py"
def module_output_path(preset_name):
    return os.path.join("output", preset_name, "module_output")

# --- Resampling filter ---
#resample_filter = Image.Resampling.LANCZOS
#resample_filter = Image.Resampling.BILINEAR
resample_filter = Image.Resampling.BICUBIC

# ---------------------------
# Layout Constants for Nodes
# ---------------------------
NODE_MARGIN = 5
NODE_WIDTH = 200
NODE_HEADER_HEIGHT = 30
NODE_PROPERTY_HEIGHT = 15
THUMBNAIL_SIZE = (200, 200)

# Valid map sizes.
VALID_MAP_SIZES = [64, 128, 256, 512, 1024, 2048, 4096]

# FONTS
try:
    font_regular_10 = ImageFont.truetype("arial.ttf", 10)
    font_regular_12 = ImageFont.truetype("arial.ttf", 12)
    font_regular_16 = ImageFont.truetype("arial.ttf", 16)
    font_bold_14 = ImageFont.truetype("arialbd.ttf", 16)
            
except:
    font_regular_10 = ImageFont.load_default()
    font_regular_12 = ImageFont.load_default()
    font_regular_16 = ImageFont.load_default()
    font_bold_14 = ImageFont.load_default()

g_editor = None
NODE_DEFINITIONS = {}

# ---------------------------
# Helper Functions
# ---------------------------
def get_node_rect(node):
    w, h = node.get_intrinsic_size()
    return (node.x, node.y, node.x + w, node.y + h)

def get_center(rect):
    x1, y1, x2, y2 = rect
    return ((x1+x2)/2, (y1+y2)/2)

def calc_border_point(rect, target):
    x1, y1, x2, y2 = rect
    cx, cy = (x1+x2)/2, (y1+y2)/2
    hw = (x2 - x1)/2
    hh = (y2 - y1)/2
    tx, ty = target
    dx = tx - cx
    dy = ty - cy
    if dx == 0 and dy == 0:
        return (cx, cy)
    scale = min(hw/abs(dx) if dx != 0 else float('inf'),
                hh/abs(dy) if dy != 0 else float('inf'))
    return (cx + dx * scale, cy + dy * scale)

def format_for_display(value):
    """Convert JSON values to user-friendly string representation."""
    if isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, list):
        return json.dumps(value)  # Display lists as valid JSON (with brackets)
    return str(value)  # Default: show numbers and strings as-is

def parse_from_display(text, expected_type):
    """Convert user-edited text back into its expected JSON type."""
    text = text.strip()

    if expected_type is bool:
        if text.lower() == "true":
            return True
        elif text.lower() == "false":
            return False
        raise ValueError("Boolean values must be 'true' or 'false'.")

    elif expected_type is int:
        try:
            return int(text)
        except ValueError:
            raise ValueError(f"Expected an integer, got '{text}'.")

    elif expected_type is float:
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"Expected a float, got '{text}'.")

    elif expected_type is list:
        try:
            return json.loads(text)  # Ensure lists are stored as valid JSON arrays
        except json.JSONDecodeError:
            raise ValueError(f"Expected a valid JSON array, got '{text}'.")

    return text  # Default to string

# ==========================
# Connection Class
# ==========================
class Connection:
    def __init__(self, source_node, source_port, target_node, target_port):
        self.source_node = source_node
        self.source_port = source_port
        self.target_node = target_node
        self.target_port = target_port
        self.canvas_line = None
        self.canvas_text_image = None

        # Pre-render the text
        font = font_regular_12
        txt = str(self.source_node.properties.get(self.source_port, ""))
        try:
            bbox = font.getbbox(txt)
            self.text_width = bbox[2] - bbox[0]
            self.text_height = bbox[3] - bbox[1]
        except Exception:
            self.text_width, self.text_height = font.getsize(txt)
        self.text_height = int(self.text_height * 1.5 + 1)
        self.text_img = Image.new("RGBA", (self.text_width, self.text_height), (255,255,255,0))
        self.text_img_photo = None
        d = ImageDraw.Draw(self.text_img)
        d.text((0,0), txt, font=font, fill="black")

    def update_line(self, canvas):
        # Compute world endpoints.
        rect_source = get_node_rect(self.source_node)
        if self.target_node is not None:
            rect_target = get_node_rect(self.target_node)
        else:
            mouse_size = 10
            rect_target = (
                g_editor.view_cursor_x - mouse_size,
                g_editor.view_cursor_y - mouse_size,
                g_editor.view_cursor_x + mouse_size,
                g_editor.view_cursor_y + mouse_size
            )
        center_target = get_center(rect_target)
        source_point = calc_border_point(rect_source, center_target)
        center_source = get_center(rect_source)
        target_point = calc_border_point(rect_target, center_source)
        # Transform to screen coordinates.
        zoom = self.source_node.editor.current_zoom
        cam_x = self.source_node.editor.camera_offset_x
        cam_y = self.source_node.editor.camera_offset_y
        sx, sy = source_point
        tx, ty = target_point
        sx_screen = (sx - cam_x) * zoom
        sy_screen = (sy - cam_y) * zoom
        tx_screen = (tx - cam_x) * zoom
        ty_screen = (ty - cam_y) * zoom

        if not self.canvas_line:
            #self.canvas_line = canvas.create_line(0, 0, 0, 0, arrow=tk.LAST, fill="#555555", width=2 * self.current_zoom)
            self.canvas_line = canvas.create_line(0, 0, 0, 0, arrow=tk.LAST, state="disabled")
        # Update arrow line.
        canvas.itemconfig(self.canvas_line, fill="#555555", width=2*zoom)
        canvas.coords(self.canvas_line, sx_screen, sy_screen, tx_screen, ty_screen)

        # Compute arrow angle.
        dx = tx_screen - sx_screen
        dy = ty_screen - sy_screen
        angle = math.degrees(math.atan2(dy, dx))
        if angle > 90 or angle < -90:
            angle += 180

        # Midpoint.
        mid_x = (sx_screen + tx_screen) / 2
        mid_y = (sy_screen + ty_screen) / 2

        # Compute a normal vector.
        angle_rad = math.radians(angle)
        normal1 = (-math.sin(angle_rad), math.cos(angle_rad))
        normal2 = (math.sin(angle_rad), -math.cos(angle_rad))

        # Choose the normal with a negative y (upward).
        normal = normal1 if normal1[1] < 0 else normal2

        offset = 10 * zoom  # scale offset with zoom.
        text_x = mid_x + normal[0] * offset
        text_y = mid_y + normal[1] * offset

        new_size = (int(self.text_width * zoom), int(self.text_height * zoom))
        scaled = self.text_img.resize(new_size, resample_filter)
        rotated = scaled.rotate(-angle, resample = resample_filter, expand=1)
        self.text_img_photo = ImageTk.PhotoImage(rotated)
        
        if self.canvas_text_image:
            canvas.itemconfig(self.canvas_text_image, image=self.text_img_photo)
            canvas.coords(self.canvas_text_image, text_x, text_y)
        else:
            self.canvas_text_image = canvas.create_image(text_x, text_y, image=self.text_img_photo, anchor="center", state="disabled")

# ==========================
# Node Class (with baked texture and live editor)
# ==========================
class Node:
    def __init__(self, editor, node_type, properties=None, x=0, y=0):
        self.editor = editor
        self.canvas = editor.canvas
        self.node_type = node_type
        self.definition = NODE_DEFINITIONS[node_type]
        self.properties = {}
        self.id = editor.get_next_node_id()
        # Initialize inputs
        for inp in self.definition.get("inputs", []):
            self.properties[inp["name"]] = inp.get("default", None)
        # Initialize outputs
        for outp in self.definition.get("outputs", []):
            self.properties[outp["name"]] = f"{node_type}_{self.id}"
        # Initialize settings
        for setting in self.definition.get("settings", []):
            self.properties[setting["name"]] = setting.get("default")
        # Initialize properties to outside specs
        if properties:
            for key, value in properties.items():
                self.properties[key] = value

        # Use saved position if available.
        if properties and "x" in properties and "y" in properties:
            self.x = properties["x"]
            self.y = properties["y"]
        else:
            self.x = x
            self.y = y

        self.baked_texture = None
        self.baked_dimensions = (0, 0)
        self.scaled_photo_image = None
        self.last_zoom = 0
        self.canvas_image_item = None

        self.selected = False
        self._drag_start = None
        self.editor.dragging = False

        self.editor_entries = {}
        self.editor_graphics_canvas = None
        self.canvas_border_item = None

        self.draw()

    def get_filtered_properties(self):
        result = {}
        for key, value in self.properties.items():
            if key != "type" and not key.startswith("_"):
                result[key] = value
        return result

    def draw(self):
        self.bake_texture()
        self.update_display()

    def update(self):
        self.update_display()

    def on_click_node(self, event):
        self.dragging = False
        # 1) Convert to world/logical coords
        zoom = self.editor.current_zoom
        world_x = self.canvas.canvasx(event.x) / zoom + self.editor.camera_offset_x
        world_y = self.canvas.canvasy(event.y) / zoom + self.editor.camera_offset_y

        # 2) Convert to node-local coords
        local_x = world_x - self.x
        local_y = world_y - self.y

        # 3) Hit-test inputs
        for name, (x0, y0, x1, y1) in self.input_regions.items():
            if x0 <= local_x <= x1 and y0 <= local_y <= y1:
                self.editor.set_selected_node(self)
                return self.on_input_click(name)

        # 4) Hit-test outputs
        for name, (x0, y0, x1, y1) in self.output_regions.items():
            if x0 <= local_x <= x1 and y0 <= local_y <= y1:
                self.editor.set_selected_node(self)
                return self.on_output_click(name)

        # 5) Fallback to dragging:
        self.editor.cancel_pending_connection()
        self.editor.set_selected_node(self)
        self.dragging = True
        self.on_start_drag(event)


    def on_start_drag(self, event):
        if self.editor.is_panning or self.dragging == False:
            return
        zoom = self.editor.current_zoom
        world_x = self.canvas.canvasx(event.x) / zoom + self.editor.camera_offset_x
        world_y = self.canvas.canvasy(event.y) / zoom + self.editor.camera_offset_y
        self._drag_start = (world_x, world_y)
        self.editor.drag_data = {"node": self, "x": world_x, "y": world_y}
        self.update_display()

    def on_output_click(self, output_name):
        """
        User clicked this node’s output: record it as the pending source.
        """
        self.editor.cancel_pending_connection()
        self.editor.pending_connection = {
            "from_node": self,
            "from_output": output_name
        }
        #print(f"Connection source set: {self}.{output_name}")
        conn = Connection(self, output_name, None, None)
        self.editor.connections.append(conn)


    def on_input_click(self, input_name):
        """
        User clicked an input: if a source is pending, complete the link
        by copying the source’s output identifier into this input property.
        """
        ec = getattr(self.editor, "pending_connection", None)
        if not ec or "from_node" not in ec:
            #print("No pending connection source; please click an output first.")
            return

        src_node   = ec["from_node"]
        src_field  = ec["from_output"]
        src_value  = src_node.properties[src_field]

        # Assign the connection by setting this input property
        self.properties[input_name] = src_value
        #print(f"Connected {src_node}.{src_field} → {self}.{input_name}")

        # Clear pending state
        self.editor.cancel_pending_connection()

        # Refresh all nodes and their connections
        for node in self.editor.nodes:
            node.draw()
        #self.editor.update_all_connections()
        self.editor.rebuild_connections()
        self.editor.set_unsaved()

    def on_drag(self, event):
        if self.editor.is_panning or self.dragging == False:
            return
        zoom = self.editor.current_zoom
        new_logical_x = self.canvas.canvasx(event.x) / zoom + self.editor.camera_offset_x
        new_logical_y = self.canvas.canvasy(event.y) / zoom + self.editor.camera_offset_y
        dx = new_logical_x - self.editor.drag_data["x"]
        dy = new_logical_y - self.editor.drag_data["y"]
        self.x += dx
        self.y += dy
        self.editor.drag_data["x"] = new_logical_x
        self.editor.drag_data["y"] = new_logical_y
        self.update_display()
        self.editor.update_all_connections()

    def on_release_drag(self, event):
        if self.editor.is_panning or self.dragging == False:
            return
        self.editor.drag_data = {}
        self._drag_start = None
        self.update_display()

    def select(self):
        self.selected = True
        self.update_display()

    def deselect(self):
        self.selected = False
        self.update_display()

    def to_json(self):
        data = {"type": self.node_type}
        data.update(self.properties)
        data["_editor_px"] = self.x
        data["_editor_py"] = self.y
        return data

    def get_intrinsic_size(self):
        return self.baked_dimensions

    def update_display(self):
        zoom = self.editor.current_zoom
        w, h = self.get_intrinsic_size()
        new_size = (int(w * zoom), int(h * zoom))
        if self.last_zoom != zoom or self.scaled_photo_image is None:
            scaled_image = self.baked_texture.resize(new_size, resample_filter)
            self.scaled_photo_image = ImageTk.PhotoImage(scaled_image)
        cam_x = self.editor.camera_offset_x
        cam_y = self.editor.camera_offset_y
        scaled_x = (self.x - cam_x) * zoom
        scaled_y = (self.y - cam_y) * zoom

        if self.canvas_image_item is None:
            # Add a tag to the node image.
            self.canvas_image_item = self.canvas.create_image(scaled_x, scaled_y, anchor="nw", image=self.scaled_photo_image, tags=(f"node_{self.id}",))
            self.canvas.tag_bind(self.canvas_image_item, "<ButtonPress-1>", self.on_click_node)
            self.canvas.tag_bind(self.canvas_image_item, "<B1-Motion>", self.on_drag)
            self.canvas.tag_bind(self.canvas_image_item, "<ButtonRelease-1>", self.on_release_drag)
        else:
            if self.last_zoom != zoom:
                self.canvas.itemconfig(self.canvas_image_item, image=self.scaled_photo_image)
            self.canvas.coords(self.canvas_image_item, scaled_x, scaled_y)
        self.last_zoom = zoom

        # Border rectangle for selection
        x1, y1 = scaled_x, scaled_y
        x2 = x1 + new_size[0]
        y2 = y1 + new_size[1]
        border_color = "blue" if self.selected else ""

        if self.canvas_border_item is None:
            self.canvas_border_item = self.canvas.create_rectangle(
                x1, y1, x2, y2, outline=border_color, width=2
            )
        else:
            self.canvas.coords(self.canvas_border_item, x1, y1, x2, y2)
            self.canvas.itemconfig(self.canvas_border_item, outline=border_color)

        # Ensure border is behind the image
        if self.canvas_image_item is not None:
            self.canvas.tag_lower(self.canvas_border_item, self.canvas_image_item)

    def create_thumbnail_image(self, file_path, ph_font, width = THUMBNAIL_SIZE[0], height = THUMBNAIL_SIZE[1]):
        if file_path and os.path.exists(file_path):
            try:
                img = Image.open(file_path).convert("RGBA")
                orig_w, orig_h = img.size
                max_w, max_h = (width, height)
                if orig_w >= orig_h:
                    new_w = max_w
                    new_h = int(round(orig_h * (max_w / orig_w)))
                else:
                    new_h = max_h
                    new_w = int(round(orig_w * (max_h / orig_h)))
                return img.resize((new_w, new_h), resample_filter)
            except Exception as e:
                canvas = Image.new("RGBA", (width, height), "white")
                draw = ImageDraw.Draw(canvas)
                draw.rectangle([0, 0, width - 1, height - 1], outline="red")
                draw.text((5, 5), "Error", fill="red", font=ph_font)
                return canvas
        else:
            canvas = Image.new("RGBA", (width, height), "white")
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([0, 0, width - 1, height - 1], outline="gray")
            draw.text((5, 5), "No image yet\nRun generation to update", fill="gray", font=ph_font)
            return canvas
        
    def bake_texture(self):
        # Create the thumbnail first so we know its dimensions.
        preset_name = self.editor.current_preset_name()
        outname = self.properties.get("out", None)
        norm_path = None
        if outname and preset_name:
            norm_path = os.path.join(module_output_path(preset_name),
                                    f"norm_{outname}.png")
        ph_font = font_regular_10
        norm_thumb = self.create_thumbnail_image(
            norm_path, ph_font,
            THUMBNAIL_SIZE[0], THUMBNAIL_SIZE[1]
        )
        thumb_w, thumb_h = norm_thumb.size

        # Gather filtered properties (skip internal "_*" and type)
        filtered_props = self.get_filtered_properties()

        # Identify inputs and outputs by definition
        input_names  = [inp["name"] for inp in self.definition.get("inputs", [])]
        output_names = [outp["name"] for outp in self.definition.get("outputs", [])]

        # Separate settings/other props
        settings_and_others = [
            k for k in filtered_props
            if k not in input_names + output_names and not k.startswith("_")
        ]

        # Build ordered rows: inputs, outputs, then settings/others
        all_rows = input_names + output_names + settings_and_others

        # Fonts
        header_font = font_bold_14
        prop_font   = font_regular_12
        io_font     = font_regular_16

        # Compute header text size
        hbbox = header_font.getbbox(self.node_type)
        header_text_w = hbbox[2] - hbbox[0]
        header_text_h = hbbox[3] - hbbox[1]

        # Measure text heights for padding
        io_tb = io_font.getbbox("Hg")
        io_text_h = io_tb[3] - io_tb[1]
        prop_tb = prop_font.getbbox("Hg")
        prop_text_h = prop_tb[3] - prop_tb[1]

        # Padding and extra spacing
        IO_PADDING = 8      # total vertical padding for I/O rows
        EXTRA_BOTTOM = 3    # extra space below text (increased for internal lower padding)
        SPACING = 2         # vertical gap between boxes, accounts for border

        # Compute per-row heights
        IO_ROW_HEIGHT    = max(NODE_PROPERTY_HEIGHT, io_text_h + IO_PADDING) + EXTRA_BOTTOM
        OTHER_ROW_HEIGHT = NODE_PROPERTY_HEIGHT + EXTRA_BOTTOM
        row_heights = [
            IO_ROW_HEIGHT if key in input_names + output_names else OTHER_ROW_HEIGHT
            for key in all_rows
        ]

        # Total height for all property boxes + gaps
        total_props_h = sum(row_heights) + SPACING * (len(row_heights) - 1)

        # Determine overall dimensions
        final_width = max(thumb_w, header_text_w, NODE_WIDTH) + 2 * NODE_MARGIN
        final_height = (
            NODE_MARGIN +
            NODE_HEADER_HEIGHT +
            NODE_MARGIN +
            total_props_h +
            NODE_MARGIN +
            thumb_h +
            NODE_MARGIN
        )

        # Create canvas
        image = Image.new("RGBA", (final_width, final_height), "white")
        draw = ImageDraw.Draw(image)

        # Draw header
        header_top    = NODE_MARGIN
        header_bottom = header_top + NODE_HEADER_HEIGHT
        draw.rectangle(
            [NODE_MARGIN, header_top, final_width - NODE_MARGIN, header_bottom],
            fill="lightblue", outline="black"
        )
        header_x = NODE_MARGIN + ((final_width - 2 * NODE_MARGIN) - header_text_w) // 2
        header_y = header_top + (NODE_HEADER_HEIGHT - header_text_h) // 2
        draw.text((header_x, header_y), self.node_type,
                fill="black", font=header_font)

        # Prepare click regions
        self.input_regions  = {}
        self.output_regions = {}

        # Draw each property/I/O box
        y = header_bottom + NODE_MARGIN
        content_left  = NODE_MARGIN
        content_right = final_width - NODE_MARGIN

        for key, row_h in zip(all_rows, row_heights):
            y0 = y
            y1 = y0 + row_h

            # Determine background and region (colors flipped)
            if key in input_names:
                bg_color = "#fce8b2"  # pale orange for inputs
                self.input_regions[key] = (content_left, y0, content_right, y1)
                font = io_font
            elif key in output_names:
                bg_color = "#c8f7c5"  # pale green for outputs
                self.output_regions[key] = (content_left, y0, content_right, y1)
                font = io_font
            else:
                bg_color = "#f0f0f0"  # very light gray for other properties
                font = prop_font

            # Draw background box
            draw.rectangle([content_left, y0, content_right, y1],
                        fill=bg_color, outline="black")

            # Draw text, vertically centered with extra bottom padding
            text = f"{key}: {filtered_props.get(key)}"
            tbbox = font.getbbox(text)
            text_h = tbbox[3] - tbbox[1]
            text_y = y0 + (row_h - text_h) // 2
            draw.text((content_left + 4, text_y), text,
                    fill="black", font=font)

            # Advance y by box height + spacing
            y = y1 + SPACING

        # Draw thumbnail
        graphics_top = y + NODE_MARGIN
        slot_x = NODE_MARGIN + ((final_width - 2 * NODE_MARGIN) - thumb_w) // 2
        image.paste(norm_thumb, (slot_x, graphics_top))

        # Outer border
        draw.rectangle([0, 0, final_width - 1, final_height - 1],
                    outline="black")

        # Store result
        self.baked_texture    = image
        self.baked_dimensions = (final_width, final_height)
        self.last_zoom        = -100

    def create_live_editor(self, parent):
        w, h = self.get_intrinsic_size()
        frame = tk.Frame(parent, width=w, height=h, relief="raised", borderwidth=2)
        frame.pack_propagate(False)

        # --- Header ---
        header = tk.Label(frame, text=self.node_type, font=("Arial", 14, "bold"), bg="lightblue")
        header.pack(fill="x", padx=NODE_MARGIN, pady=(NODE_MARGIN, 0))
        
        # --- Module Documentation Widget using full width ---
        module_doc = self.definition.get(
            "doc",
            "Module doc not found"
        )
        # Use a tk.Message instead of tk.Label to get a neat, full-width, auto-wrapping text display.
        doc_message = tk.Message(frame,
                                text=module_doc,
                                width=int(THUMBNAIL_SIZE[0] * 1.5),#w - 2 * NODE_MARGIN,
                                bg="lightyellow",
                                font=("Arial", 10),
                                justify="center")
        doc_message.pack(fill="x", padx=NODE_MARGIN, pady=(NODE_MARGIN, 0))

        # --- Properties Section ---
        prop_frame = tk.Frame(frame)
        prop_frame.pack(fill="x", padx=NODE_MARGIN, pady=NODE_MARGIN)

        self.editor_entries = {}
        # Mouse-over tooltip now uses the same tone of yellow ("lightyellow")
        self.tooltip = tk.Label(frame, text="", bg="lightyellow", relief="solid", borderwidth=1, wraplength=200)
        self.tooltip.place_forget()  # Hide initially

        filtered_props = self.get_filtered_properties()
        for key, value in filtered_props.items():
            row = tk.Frame(prop_frame)
            row.pack(fill="x", pady=2)

            lbl = tk.Label(row, text=f"{key}:", width=15, anchor="w")
            lbl.pack(side="left")

            ent = tk.Entry(row)
            help_text = "No documentation available"
            for setting in self.definition.get("settings", []):
                if setting["name"] == key:
                    help_text = f'{setting["doc"]}\nDefaults to \'{setting["default"]}\''
                    break

            for setting in self.definition.get("inputs", []):
                if setting["name"] == key:
                    help_text = f'{setting["doc"]}\nThe name of some other module\'s output.'
                    break

            for setting in self.definition.get("outputs", []):
                if setting["name"] == key:
                    help_text = f'{setting["doc"]}\nTo be used in some other module as input.\nThe name \'final\' makes it the final output for this template.'
                    break

            original_type = type(value)
            self.editor_entries[key] = (ent, original_type)

            ent.insert(0, format_for_display(value))
            ent.pack(side="left", fill="x", expand=True)

            ent.bind("<Enter>", lambda event, text=help_text: self.show_tooltip(event, text))
            ent.bind("<Leave>", lambda event: self.hide_tooltip())

        # --- Graphics Section ---
        graphics_frame = tk.Frame(frame)
        graphics_frame.pack(padx=NODE_MARGIN, pady=NODE_MARGIN)

        # Commit/Delete buttons placed above the thumbnail
        btn_frame = tk.Frame(graphics_frame)
        btn_frame.pack(pady=NODE_MARGIN)

        commit_btn = tk.Button(btn_frame, text="Apply Changes", command=self.commit_from_editor)
        commit_btn.pack(side="left", padx=5)

        #delete_btn = tk.Button(btn_frame, text="Delete", command=lambda: self.editor.delete_selected_node())
        #delete_btn.pack(side="left", padx=5)

        preset_name = self.editor.current_preset_name()
        outname = self.properties.get("out", None)
        norm_path = None
        if outname and preset_name:
            norm_path = os.path.join(module_output_path(preset_name), f"norm_{outname}.png")
        ph_font = font_regular_10

        # Create the norm thumbnail which preserves aspect ratio.
        norm_thumb = self.create_thumbnail_image(norm_path, ph_font,
                                                int(THUMBNAIL_SIZE[0] * 1.5),
                                                int(THUMBNAIL_SIZE[1] * 1.5))
        thumb_w, thumb_h = norm_thumb.size

        # Set the canvas to exactly match the thumbnail's dimensions.
        canvas = tk.Canvas(graphics_frame, width=thumb_w, height=thumb_h, bg="white")
        canvas.pack()

        # Convert the PIL thumbnail into a Tkinter PhotoImage.
        norm_photo = ImageTk.PhotoImage(norm_thumb)

        # Place the image onto the canvas, centered.
        canvas.create_image(thumb_w // 2, thumb_h // 2, image=norm_photo)

        # Keep a reference to avoid garbage collection.
        self._norm_photo = norm_photo

        self.editor_graphics_canvas = canvas
        canvas.bind("<Button-1>", self.on_graphics_click)

        # Compute brightness info from the raw image (instead of the norm variant)
        raw_path = None
        if outname and preset_name:
            raw_path = os.path.join(module_output_path(preset_name), f"raw_{outname}.png")
        try:
            raw_image = Image.open(raw_path)
            raw_gray = raw_image.convert("L")
            min_brightness, max_brightness = raw_gray.getextrema()
            brightness_text = f"Brightness {min_brightness} to {max_brightness}"
        except Exception as e:
            brightness_text = "Unable to determine brightness info"

        brightness_label = tk.Label(graphics_frame, text=brightness_text)
        brightness_label.pack(pady=NODE_MARGIN)

        update_label = tk.Label(graphics_frame,
                                text="[Click image to open]\n[Click on node to update image to latest generation]",
                                font=("Arial", 10))
        update_label.pack(pady=(0, NODE_MARGIN))

        return frame

    def show_tooltip(self, event, text):
        """Show tooltip near the entry widget."""
        self.tooltip.config(text=text)
        self.tooltip.place(x=event.widget.winfo_rootx() - self.tooltip.master.winfo_rootx(),
                           y=event.widget.winfo_rooty() - self.tooltip.master.winfo_rooty() + 25)
        self.tooltip.lift()  # Ensure tooltip appears on top

    def hide_tooltip(self, event=None):
        """Hide tooltip."""
        self.tooltip.place_forget()
        
    def commit_from_editor(self):
        """Parse values back into their correct types and store them."""
        for key, (entry, original_type) in self.editor_entries.items():
            new_val = entry.get().strip()

            try:
                self.properties[key] = parse_from_display(new_val, original_type)
            except ValueError as e:
                messagebox.showerror("Error", str(e))
                return

        self.draw()
        # this probably should include a slight update of node editor graphics but nevermind for now
        self.editor.cancel_pending_connection()
        self.editor.rebuild_connections()
        self.editor.set_unsaved(True)

    def on_graphics_click(self, event):
        self.open_full_image_in_editor("norm")

    def open_full_image_in_editor(self, slot):
        preset_name = self.editor.current_preset_name()
        outname = self.properties.get("out", None)
        if not (outname and preset_name):
            return
        path = os.path.join(module_output_path(preset_name), f"{slot}_{outname}.png")
        if not os.path.exists(path):
            return
        top = tk.Toplevel(self.editor)
        top.title(os.path.basename(path))
        canvas = tk.Canvas(top, bg="black")
        canvas.pack(fill="both", expand=True)
        try:
            original = Image.open(path).convert("RGBA")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")
            return
        def resize_image(event):
            cw, ch = event.width, event.height
            ow, oh = original.size
            scale = min(cw/ow, ch/oh)
            new_size = (int(ow*scale), int(oh*scale))
            resized = original.resize(new_size, resample_filter)
            photo = ImageTk.PhotoImage(resized)
            canvas.photo = photo
            canvas.delete("all")
            canvas.create_image(cw/2, ch/2, image=photo, anchor="center")
        canvas.bind("<Configure>", resize_image)


# ==========================
# Main Editor Application
# ==========================

class NodeEditorApp(tk.Tk):
    def __init__(self):
        global g_editor
        super().__init__()
        g_editor = self
        self.current_preset = "Untitled"
        self.unsaved = False
        self.map_width = 256
        self.map_height = 256
        self.update_title()
        self.geometry("1200x800")
        self.nodes = []
        self.connections = []
        self.drag_data = {}
        self._node_id_counter = 1
        self.is_panning = False
        self.current_zoom = 1.0
        self.camera_offset_x = 0.0
        self.camera_offset_y = 0.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.pan_start_camera_x = 0
        self.pan_start_camera_y = 0
        self.shift_pressed = False
        self.mouse_x = 0
        self.mouse_y = 0
        self.pending_connection = None

        self.create_menu()
        self.create_toolbar()
        self.create_ui()
        
        self.bind("<Control-s>", lambda event: self.save_template())
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.canvas.bind("<ButtonPress-3>", self.on_right_button_press)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_button_release)
        self.canvas.bind("<B3-Motion>", self.on_right_button_drag)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<ButtonPress-1>", self.on_left_mouse_button)
        #self.bind("<KeyPress-space>", self.on_space_press)
        #self.bind("<KeyRelease-space>", self.on_space_release)
        self.bind("<KeyPress-Shift_L>", self.on_shift_press)
        self.bind("<KeyRelease-Shift_L>", self.on_shift_release)
        self.bind("<KeyPress-Shift_R>", self.on_shift_press)
        self.bind("<KeyRelease-Shift_R>", self.on_shift_release)
        self.bind("<KeyRelease-Escape>", self.on_escape)

    def update_title(self):
        if self.current_preset:
            title = f"WorldStack Editor - {self.current_preset}"
        else:
            title = "WorldStack Editor"
        if self.unsaved:
            title += " *"
        self.title(title)

    def set_unsaved(self, flag=True):
        self.unsaved = flag
        self.update_title()

    # --- Menus ---
    def create_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="New", command=self.new_template)
        filemenu.add_command(label="Open", command=self.open_template)
        filemenu.add_command(label="Save", command=self.save_template)
        filemenu.add_command(label="Save As", command=self.save_template_as)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.on_close)
        menubar.add_cascade(label="File", menu=filemenu)
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Map Size", command=self.settings_map_size)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        self.config(menu=menubar)

    def settings_map_size(self):
        top = tk.Toplevel(self)
        top.title("Map Size")

        tk.Label(top, text="Select Map Width:").grid(row=0, column=0, padx=5, pady=5)
        width_var = tk.IntVar(value=self.map_width)
        width_combo = ttk.Combobox(top, textvariable=width_var, values=VALID_MAP_SIZES, state="readonly")
        width_combo.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(top, text="Select Map Height:").grid(row=1, column=0, padx=5, pady=5)
        height_var = tk.IntVar(value=self.map_height)
        height_combo = ttk.Combobox(top, textvariable=height_var, values=VALID_MAP_SIZES, state="readonly")
        height_combo.grid(row=1, column=1, padx=5, pady=5)

        def apply():
            self.map_width = width_var.get()
            self.map_height = height_var.get()
            self.set_unsaved(True)
            top.destroy()

        tk.Button(top, text="OK", command=apply).grid(row=2, column=0, columnspan=2, pady=10)

    def create_toolbar(self):
        toolbar = tk.Frame(self, bd=1, relief=tk.RAISED)
        go_button = tk.Button(toolbar, text="Regenerate", command=self.run_generation)
        go_button.pack(side=tk.LEFT, padx=2, pady=2)
        add_node_button = tk.Button(toolbar, text="New Node", command=self.add_node_dialog)
        add_node_button.pack(side=tk.LEFT, padx=2, pady=2)
        add_node_button = tk.Button(toolbar, text="Delete Node(s)", command=self.delete_selected_node)
        add_node_button.pack(side=tk.LEFT, padx=2, pady=2)
        add_node_button = tk.Button(toolbar, text="Help", command=self.show_help)
        add_node_button.pack(side=tk.LEFT, padx=2, pady=2)

        self.status_label = tk.Label(toolbar, text="", fg="red")
        self.status_label.pack(side=tk.RIGHT, padx=10)
        toolbar.pack(side=tk.TOP, fill=tk.X)

    def create_ui(self):
        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(main_pane, bg="white", width=800, height=800)
        self.canvas.config(scrollregion=(-5000, -5000, 5000, 5000))
        self.canvas_bg = self.canvas.create_rectangle(-5000, -5000, 5000, 5000, fill="white", outline="", state="disabled")
        main_pane.add(self.canvas, stretch="always")
        self.prop_frame = tk.LabelFrame(main_pane, text="Node Editor", width=300)
        main_pane.add(self.prop_frame)
        self.prop_widgets = {}
        self.update_title()

    # --- Camera Methods ---
    def on_left_mouse_button(self, event):
        items = self.canvas.find_overlapping(event.x, event.y, event.x, event.y)
        interactive_items = [i for i in items if self.canvas.itemcget(i, "state") != "disabled"]
        if not interactive_items:
            #print("Clicked on blank canvas (or disabled items only)")
            self.cancel_pending_connection()
            self.deselect_all()
        else:
            #print("Clicked on something??")
            pass

    def on_pan_start(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.pan_start_camera_x = self.camera_offset_x
        self.pan_start_camera_y = self.camera_offset_y

    def on_pan_drag(self, event):
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.camera_offset_x = self.pan_start_camera_x - dx / self.current_zoom
        self.camera_offset_y = self.pan_start_camera_y - dy / self.current_zoom
        for node in self.nodes:
            node.update()
        self.update_all_connections()

    def on_right_button_press(self, event):
        self.config(cursor="fleur")
        self.on_pan_start(event)
    
    def on_right_button_release(self, event):
        self.config(cursor="")

    def on_right_button_drag(self, event):
        self.on_pan_drag(event)

    def on_mouse_move(self, event):
        self.view_cursor_x = event.x / self.current_zoom + self.camera_offset_x
        self.view_cursor_y = event.y / self.current_zoom + self.camera_offset_y
        for c in self.connections:
            if c.target_node == None:
                c.update_line(self.canvas)

    def on_shift_press(self, event):
        self.shift_pressed = True

    def on_shift_release(self, event):
        self.shift_pressed = False

    def on_mousewheel(self, event):
        if event.num == 4 or event.delta > 0:
            factor = 1.1
        elif event.num == 5 or event.delta < 0:
            factor = 0.9
        else:
            factor = 1.0
        old_zoom = self.current_zoom
        raw_zoom = old_zoom * factor
        step = 0.01  # adjustable zoom step
        new_zoom = round(raw_zoom / step) * step
        if new_zoom == old_zoom:
            if factor > 1:
                new_zoom = old_zoom + step
            elif factor < 1:
                new_zoom = old_zoom - step
        if new_zoom < 0.2 or new_zoom > 2.0:
            return
        mx, my = event.x, event.y
        logical_x = mx / old_zoom + self.camera_offset_x
        logical_y = my / old_zoom + self.camera_offset_y
        self.current_zoom = new_zoom
        self.camera_offset_x = logical_x - mx / new_zoom
        self.camera_offset_y = logical_y - my / new_zoom

        # Instead of immediately updating every node and connection, schedule the heavy update.
        if hasattr(self, '_zoom_update_job') and self._zoom_update_job is not None:
                self.after_cancel(self._zoom_update_job)
        if self.current_zoom > 1.0:
            self._zoom_update_job = self.after(25, self.perform_zoom_update)
        else:
            self.perform_zoom_update()

    def perform_zoom_update(self):
        for node in self.nodes:
            node.update()
        self.update_all_connections()
        self._zoom_update_job = None

    def on_escape(self, event):
        self.cancel_pending_connection()
        self.deselect_all()

    # --- Node Management ---
    def get_next_node_id(self):
        nid = self._node_id_counter
        self._node_id_counter += 1
        return nid

    def deselect_all(self):
        for node in self.nodes:
                if node.selected:
                    node.deselect()

    def set_selected_node(self, node):
        if self.shift_pressed:
            if node.selected:
                node.deselect()
            else:
                node.select()
        else:
            for other in self.nodes:
                if other is not node and other.selected:
                    other.deselect()
            if node.selected:
                #node.deselect()
                pass
            else:
                node.select()

        if node.selected:
            self.show_node_editor(node)
        else:
            self.hide_node_editor()

    def hide_node_editor(self):
        for widget in self.prop_frame.winfo_children():
            widget.destroy()

    def show_node_editor(self, node):
        for widget in self.prop_frame.winfo_children():
            widget.destroy()
        editor_widget = node.create_live_editor(self.prop_frame)
        editor_widget.pack(fill="both", expand=True)

    def ask_node_type(self):
        top = tk.Toplevel(self)
        top.title("New Node")
        #top.geometry("400x250")
        window_width, window_height = 400, 250
        top.resizable(False, False)
        
        # Ensure the parent's geometry info is updated.
        self.update_idletasks()
        parent_x = self.winfo_rootx()
        parent_y = self.winfo_rooty()
        parent_width = self.winfo_width()
        parent_height = self.winfo_height()
        
        # Calculate position for centering the dialog relative to the parent.
        x = parent_x + (parent_width - window_width) // 2
        y = parent_y + (parent_height - window_height) // 2
        top.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Prepare sorted module list and mapping.
        modules_sorted = sorted(NODE_DEFINITIONS.values(), key=lambda m: (m.get("category", ""), m.get("type", "")))
        display_items = []
        mapping = {}
        for module in modules_sorted:
            display = f"{module.get('category', '')}.{module.get('type', '')}"
            display_items.append(display)
            mapping[display] = module.get("type", "")
        
        # Add default option at the top.
        display_items = ["Select node type"] + display_items
        var = tk.StringVar(value="Select node type")
        combo = ttk.Combobox(top, textvariable=var, values=display_items, width=60, state="readonly")
        combo.pack(padx=10, pady=5)
        
        # Scrollable documentation area (default text is empty).
        doc_frame = tk.Frame(top)
        doc_frame.pack(padx=10, pady=5, fill="both", expand=False)
        text_doc = tk.Text(doc_frame, height=10, width=60, wrap="word")
        text_doc.insert("1.0", "Select a node type to get a description.\n\n- Generators create an image, you probably want to start here.\n\n- Filters usually tweak a single image.\n\n- Processors do everytyhing else, like combining two images.")  # default documentation text
        text_doc.config(state="disabled")
        scrollbar = tk.Scrollbar(doc_frame, command=text_doc.yview)
        text_doc.config(yscrollcommand=scrollbar.set)
        text_doc.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        def update_doc(event=None):
            selected_display = var.get()
            if selected_display in mapping:
                module_type = mapping[selected_display]
                doc_text = NODE_DEFINITIONS[module_type].get("doc", "No documentation available.")
            else:
                doc_text = ""
            text_doc.config(state="normal")
            text_doc.delete("1.0", tk.END)
            text_doc.insert("1.0", doc_text)
            text_doc.config(state="disabled")
        
        combo.bind("<<ComboboxSelected>>", update_doc)
        # Initially, no valid selection so documentation remains as default ("")
        
        result = {}
        
        def ok():
            if var.get() not in mapping:
                messagebox.showerror("Error", "Please select a valid node type.")
                return
            result["type"] = mapping[var.get()]
            top.destroy()
        
        def cancel():
            top.destroy()
        
        btn_frame = tk.Frame(top)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="OK", command=ok).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=cancel).pack(side="left", padx=5)
        
        top.wait_window()
        return result.get("type")
        
    def add_node_dialog(self):
        node_type = self.ask_node_type()
        if not node_type:
            return
        cx = self.canvas.winfo_width() / 2 / self.current_zoom + self.camera_offset_x
        cy = self.canvas.winfo_height() / 2 / self.current_zoom + self.camera_offset_y
        node = Node(self, node_type, x=cx, y=cy)
        self.nodes.append(node)
        self.update_all_connections()
        self.set_unsaved(True)

    def delete_selected_node(self):
        # 1) Find all selected nodes
        selected_nodes = [n for n in self.nodes if n.selected]

        if not selected_nodes:
            return  # nothing to do

        self.cancel_pending_connection()

        # 2) Remove any connections involving those nodes
        remaining_conns = []
        for conn in self.connections:
            if conn.source_node in selected_nodes or conn.target_node in selected_nodes:
                # delete its canvas elements
                self.canvas.delete(conn.canvas_line)
                if conn.canvas_text_image:
                    self.canvas.delete(conn.canvas_text_image)
            else:
                remaining_conns.append(conn)
        self.connections = remaining_conns

        # 3) Remove the nodes
        for node in selected_nodes:
            # delete their canvas items
            if node.canvas_image_item:
                self.canvas.delete(node.canvas_image_item)
            if node.canvas_border_item:
                self.canvas.delete(node.canvas_border_item)
            # remove from the list
            self.nodes.remove(node)

        # 4) Clean up the editor UI & mark dirty
        self.hide_node_editor()
        self.set_unsaved(True)

    def get_node_by_id(self, nid):
        for node in self.nodes:
            if node.id == nid:
                return node
        return None

    def cancel_pending_connection(self):
        """
        Abort any half‐started connection (e.g. on Escape key).
        """
        
        for c in self.connections[:]:  # iterate over a copy to allow safe removal
            if c.target_node is None:
                # delete its canvas elements
                self.canvas.delete(c.canvas_line)
                if c.canvas_text_image:
                    self.canvas.delete(c.canvas_text_image)
                self.connections.remove(c)
                    
        if self.pending_connection != None:
            self.pending_connection = None
            self.update_all_connections()

    def update_all_connections(self):
        for conn in self.connections:
            conn.update_line(self.canvas)

    def rebuild_connections(self):
        if self.connections:
            for conn in self.connections:
                if conn.canvas_line:
                    self.canvas.delete(conn.canvas_line)
                if conn.canvas_text_image:
                    self.canvas.delete(conn.canvas_text_image)
        self.connections = []
        for node in self.nodes:
            for inp in node.definition.get("inputs", []):
                inp_name = inp["name"]
                val = node.properties.get(inp_name)
                if not val:
                    continue
                for other in self.nodes:
                    for outp in other.definition.get("outputs", []):
                        out_name = outp["name"]
                        if other.properties.get(out_name) == val:
                            conn = Connection(other, out_name, node, inp_name)
                            self.connections.append(conn)
                            break
        self.update_all_connections()

    def current_preset_name(self):
        return self.current_preset

    def open_full_image(self, image_path):
        top = tk.Toplevel(self)
        top.title(os.path.basename(image_path))
        try:
            im = Image.open(image_path)
            photo = ImageTk.PhotoImage(im)
            lbl = tk.Label(top, image=photo)
            lbl.image = photo
            lbl.pack(fill="both", expand=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")

    def new_template(self):
        if self.unsaved:
            if not messagebox.askyesno("New Template", "Discard unsaved changes?"):
                return
        self.clear_canvas()
        self.nodes = []
        self.connections = []
        self.current_preset = "Untitled"
        #self.save_template()
        self.set_unsaved(False)

    def open_template(self):
        if self.unsaved:
            if not messagebox.askyesno("Open Template", "Discard unsaved changes?"):
                return
        tmpl_dir = os.path.join(os.getcwd(), "templates")
        if not os.path.exists(tmpl_dir):
            os.makedirs(tmpl_dir)
        filepath = filedialog.askopenfilename(initialdir=tmpl_dir, filetypes=[("JSON files", "*.json")], title="Open Template")
        if not filepath:
            return
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file: {e}")
            return
        self.clear_canvas()
        self.nodes = []
        self.connections = []
        self.current_preset = os.path.splitext(os.path.basename(filepath))[0]
        self.map_width = data.get("width", 256)
        self.map_height = data.get("height", 256)
        modules = data.get("modules", [])
        positions_present = all("_editor_px" in mod and "_editor_py" in mod for mod in modules)
        for mod in modules:
            node_type = mod.get("type")
            if node_type not in NODE_DEFINITIONS:
                continue
            if positions_present:
                node = Node(self, node_type, properties=mod, x=mod["_editor_px"], y=mod["_editor_py"])
            else:
                node = Node(self, node_type, properties=mod)
            self.nodes.append(node)
        
        self.rebuild_connections()
        
        if not positions_present:
            self.auto_layout()
        self.canvas.configure(scrollregion=self.canvas.bbox("all") or (-5000, -5000, 5000, 5000))
        self.canvas.update_idletasks()
        for node in self.nodes:
            node.draw()
        self.set_unsaved(False)

    def save_template(self):
        self.save_template_as(no_as = True)

    def save_template_as(self, no_as = False):
        tmpl_dir = os.path.join(os.getcwd(), "templates")
        if not os.path.exists(tmpl_dir):
            os.makedirs(tmpl_dir)
        if no_as == False or not self.current_preset:
            file_path = filedialog.asksaveasfilename(initialdir=tmpl_dir, defaultextension=".json",
                                                    filetypes=[("JSON files", "*.json")],
                                                    title="Save Template As")
        else:
            file_path = os.path.join(tmpl_dir, f"{self.current_preset}.json")

        if not file_path:
            return
        self.current_preset = os.path.splitext(os.path.basename(file_path))[0]
        data = {"width": self.map_width, "height": self.map_height, "modules": [node.to_json() for node in self.nodes]}
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")
            return
        self.set_unsaved(False)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.canvas_bg = self.canvas.create_rectangle(-5000, -5000, 5000, 5000, fill="white", outline="", state="disabled")

    def auto_layout(self):
        in_degree = {node.id: 0 for node in self.nodes}
        parent_map = {node.id: [] for node in self.nodes}
        for conn in self.connections:
            tgt = conn.target_node.id
            src = conn.source_node.id
            in_degree[tgt] += 1
            parent_map[tgt].append(src)
        levels = {}
        for node in self.nodes:
            if in_degree[node.id] == 0:
                levels[node.id] = 0
            else:
                levels[node.id] = None
        changed = True
        while changed:
            changed = False
            for node in self.nodes:
                if levels[node.id] is None and parent_map[node.id]:
                    p_levels = [levels[pid] for pid in parent_map[node.id] if levels[pid] is not None]
                    if p_levels:
                        new_level = max(p_levels) + 1
                        if levels[node.id] != new_level:
                            levels[node.id] = new_level
                            changed = True
        level_groups = {}
        for node in self.nodes:
            lvl = levels.get(node.id, 0)
            level_groups.setdefault(lvl, []).append(node)
        margin_x = 50
        current_x = margin_x
        level_x = {}
        for lvl in sorted(level_groups.keys()):
            level_x[lvl] = current_x
            current_x += NODE_WIDTH + margin_x
        margin_y = 50
        for lvl, nodes in level_groups.items():
            spacing_y = margin_y + max(self.get_node_height(n) for n in nodes) + 50
            y0 = margin_y
            for i, node in enumerate(nodes):
                node.x = level_x[lvl]
                node.y = y0 + i * spacing_y
                node.update()
        self.update_all_connections()

    def get_node_height(self, node):
        _, h = node.get_intrinsic_size()
        return h
    
    def show_help(self, width=950, height=700):
        raw_help_text = """
    Welcome to WorldStack Editor!

    This tool lets you design your own templates for use in WorldStack Generator - the actual core of WorldStack. Here you make a generator template, the generator (a commandline tool) then creates heightmap variations from it, and these can be imported in games like OpenTTD as a basis for actual map generation. The console output points out at what maximum height the map should be imported if the game asks.

    The core concept of WorldStack is to combine the functionality of various nodes to generate specific kinds of maps, instead of just the same old, non-specific maps one may be used to. I tried my best to make this as accessible as possible, but please be aware that creating good generator templates is just inherently difficult. The intention is that regular users would use templates created by advanced users.

    Controls:

    - Drag the main view by holding the right mouse button
    - Zoom with the mouse wheel
    - Hold shift to add a node to your selection (currently only used for node deletion)
    - Click an output and then an input to quickly create a new connection
    - Press Escape to cancel selections/connecting
    - Press Ctrl-S to save
    - There is no undo functionality

    Start by creating a single node of the type "simplex_noise" and naming the output "final". That's already a minimal, complete WorldStack Generator template. You can select the map dimensions in the menu. For non-square maps, please use landscape format as that integrates much better into the node display.

    Press "Regenerate" to update image previews after making changes. The generator will run in the background and then update your previews. This can be quite slow on big maps. You can watch the console window to see what it's doing.

    Additional help is provided on demand throughout the editor.

    This is a work in progress, please understand that things are not final and that you may run into bugs, missing features and general oddities.

    Nodes are based on plugins. Look at plugin.py and the provided plugins for more documentation on how you can make your own node type. However this is not a stable interface yet and you should expect breaking changes in future WorldStack releases.

    Created by Simon Galle.
    Check out my game Swarm Universe on Steam if you feel like supporting me.
    """
        help_text = textwrap.dedent(raw_help_text).strip()

        # Create the help window
        help_window = tk.Toplevel(self)
        help_window.title("Help")
        help_window.geometry(f"{width}x{height}")
        help_window.resizable(False, False)

        # Center relative to parent
        self.update_idletasks()
        parent_x = self.winfo_rootx()
        parent_y = self.winfo_rooty()
        parent_w = self.winfo_width()
        parent_h = self.winfo_height()
        x = parent_x + (parent_w // 2) - (width // 2)
        y = parent_y + (parent_h // 2) - (height // 2)
        help_window.geometry(f"+{x}+{y}")

        # Layout container
        container = ttk.Frame(help_window, padding=10)
        container.pack(fill="both", expand=True)

        # Text widget with scrollbar
        text_widget = tk.Text(container, wrap="word")
        text_widget.insert("1.0", help_text)
        text_widget.configure(state="disabled")
        text_widget.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(container, orient="vertical", command=text_widget.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text_widget.configure(yscrollcommand=scrollbar.set)

        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

    def run_generation(self):
        if self.unsaved:
            if messagebox.askyesno("Generation", "Save and continue?"):
                self.save_template()
            else:
                return
        self.status_label.config(text="Processing")
        self.update_idletasks()
        preset_name = self.current_preset
        cmd = [sys.executable, generator_file, f'-i={preset_name}', "--moduleoutput", f'-o={preset_name}']
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Generation failed", f"Please check the console output for hints what went wrong.")
            self.status_label.config(text="")
            return
        self.status_label.config(text="")
        for node in self.nodes:
            node.draw()
        self.update_all_connections()

    def on_close(self):
        if self.unsaved:
            if not messagebox.askyesno("Exit", "Discard unsaved changes?"):
                return
        self.destroy()

    def run(self):
        self.mainloop()

def run_generator_doc(okay_to_fail = True):
    cmd = [sys.executable, generator_file, "--doc"]
    print("Generating module documentation...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout.strip())
    except subprocess.CalledProcessError as e: # this exception is basically supposed to mean the generator returned 1
        print(f"Error running generator:\n{e.stderr.strip()}")
        if okay_to_fail:
            sys.exit(1)
        else:
            return e.stderr.strip()
    except Exception as e: # this exception is whatever
        print(f"Error running generator:\n{e}")
        if okay_to_fail:
            sys.exit(1)
        else:
            return f'Unknown Error: {e}'
    return ""

# ==========================
# Main Entry Point
# ==========================
if __name__ == "__main__":
    import time

    # Load splash image
    img = Image.open("images/worldstack.png")
    splash = tk.Tk()
    splash.overrideredirect(True)

    photo = ImageTk.PhotoImage(img)

    img_width, img_height = img.size
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    x = (screen_width - img_width) // 2
    y = (screen_height - img_height) // 2
    splash.geometry(f"{img_width}x{img_height}+{x}+{y}")

    label = tk.Label(splash, image=photo)
    label.pack()
    label.image = photo  # keep a reference

    start_time = time.perf_counter()

    def background_setup():
        print("WorldStack Editor v0.1.0")
        run_generator_doc(False)

        try:
            with open("doc/modules.json", "r") as f:
                modules_list = json.load(f)
            globals()["NODE_DEFINITIONS"] = {
                module["type"]: module for module in modules_list
            }
        except Exception as e:
            print("Error loading doc/modules.json:", e)
            globals()["NODE_DEFINITIONS"] = {}

        elapsed = time.perf_counter() - start_time
        delay = max(0, 1.0 - elapsed)  # Minimum 1 second
        splash.after(int(delay * 1000), launch_app)

    def launch_app():
        splash.destroy()
        print("Starting Editor")
        app = NodeEditorApp()
        app.run()

    threading.Thread(target=background_setup, daemon=True).start()
    splash.mainloop()
