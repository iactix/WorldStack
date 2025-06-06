#!/usr/bin/env python3

import sys
import importlib.util
import os
import json
from datetime import datetime
import time
import random
import inspect
from pathlib import Path
import numpy as np
from PIL import Image
from plugin import GeneratorModule

def load_plugins():
    instances = []
    plugin_dir = Path(__file__).parent / "plugins"

    for plugin_path in plugin_dir.glob("*.py"):
        if plugin_path.name == "plugin.py":  # Skip the base class file
            continue

        module_name = plugin_path.stem  # Get filename without .py
        module_spec = importlib.util.spec_from_file_location(module_name, plugin_path)
        module = importlib.util.module_from_spec(module_spec)

        try:
            module_spec.loader.exec_module(module)  # Load module dynamically
        except Exception as e:
            print(f"Error loading plugin {module_name}: {e}", file=sys.stderr)
            sys.exit(1)

        # Find subclasses of GeneratorModule
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, GeneratorModule) and obj is not GeneratorModule:
                try:
                    instances.append(obj())  # Instantiate each subclass
                    print(f'Loaded module \'{name}\' from plugin \'{module_name}\'')
                    #print("Module documentation:")
                    #print(json.dumps(instances[-1].get_documentation(), indent=2))
                except Exception as e:
                    print(f'Failed to initialize module \'{name}\' from plugin \'{module_name}\': {e}')

    return instances

def add_job(target, jobs, modules, configs):
    for c in configs:
        if c in jobs:
            continue
        if c["type"] not in modules:
            raise Exception(f'Module \'{c["type"]}\' not found')
        module = modules[c["type"]]
        for o in module.outputs:
            if c[o] == target:
                for i in module.inputs:
                    if i not in c:
                        raise Exception(f'Module \'{c["type"]}\' has no input \'{i}\'')
                    add_job(c[i], jobs, modules, configs)
                jobs.append(c)
                return

def create_execution_order(modules, configs):
    jobs = []
    
    # First pass: Resolve dependencies leading to "final"
    add_job("final", jobs, modules, configs)

    # Second pass: Process everything not yet covered, but only if its output is unused
    processed = {id(c) for c in jobs}  # Track processed modules
    for c in configs:
        if id(c) in processed:
            continue  # Skip already processed modules

        module_type = c["type"]
        if module_type not in modules:
            raise Exception(f"Unknown module type: {module_type}")
        
        # Pick a valid output for this module
        module = modules[module_type]
        valid_outputs = [o for o in module.outputs if o in c]
        if not valid_outputs:
            raise Exception(f"Module '{module_type}' has no valid outputs in config: {c}")

        # Use one of the valid outputs
        add_job(c[valid_outputs[0]], jobs, modules, configs)

    # Check if all modules were processed
    if len(jobs) != len(configs):
        raise Exception(f"Job count mismatch: expected {len(configs)}, but got {len(jobs)}")

    return jobs


def save_image(destination, data, normalize, norm_min = True):
    if normalize:
        if norm_min:
            min = data.min()
        else:
            min = 0
        max = data.max()
        if max > 255: max = 255
        if min < 0: min = 0
        if min == max:
            min -= 1
            min += 1
        img = Image.fromarray(np.clip((data - min) * (255.0 / (max - min)), 0, 255).astype(np.uint8), mode="L")
    else:
        img = Image.fromarray(np.clip(data, 0, 255).astype(np.uint8), mode="L")
    
    #processed = np.stack([red, green, blue], axis=-1)  # RGB format
    #img = Image.fromarray(processed, mode="RGB")

    img.save(destination)


############################################################################################################
### Main Program
############################################################################################################

def main():
    template_name = ""
    output_file = ""
    temp_path = ""
    usage_msg = "Usage: python ws-gen.py <Options>\n"
    usage_msg += "Options:\n"
    usage_msg += "<Template Name>     |  Required, specify name of template in '/templates/' without path or .json. Also works in the form -i=\n"
    usage_msg += "-o=<Output Name>    |  Optional, specify name of target image in '/output/<Template Name>/' without path or .png. Defaults to timestamped name\n"
    usage_msg += "--moduleoutput      |  Optional, triggers creation of individual module output images in '/output/<Template Name>/module_output'\n"
    usage_msg += "--doc               |  Writes available module documentation to '/doc/' as json (overrides other behavior)\n"
    usage_msg += "-h or --help        |  Print this message (overrides other behavior)\n"
    usage_msg += "Example: python ws-gen.py mytemplate --moduleoutput"

    arg_out = ""
    arg_moduleout = False
    
    os.makedirs("templates", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    input_specs = 0
    for arg in sys.argv[1:]:
        if arg.startswith("-i="):
            template_name = arg[3:]
            input_specs += 1
        elif arg.startswith("-o="):
            arg_out = arg[3:]    
        elif arg == "--moduleoutput":
            arg_moduleout = True
        elif arg == "--doc":
            instances = load_plugins()
            doc = [i.get_documentation() for i in instances]

            os.makedirs("doc", exist_ok=True)
            with open("doc/modules.json", "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=4)  # Write the documentation as formatted JSON

            print("Documentation written to 'doc/modules.json'")
            sys.exit(0)
        elif arg == "-h" or arg == "--help":
            print(usage_msg)
            sys.exit(0)
        else:
            template_name = arg
            input_specs += 1

        if input_specs > 1:
            print(f"Error: Ambiguous input specification / Unexpected argument '{arg}'", file=sys.stderr)
            print(usage_msg)
            sys.exit(1)

    if not template_name:
        print("Error: Missing required argument -i=<TEMPLATE_NAME>", file=sys.stderr)
        print(usage_msg)
        sys.exit(1)

    os.makedirs(os.path.join("output", template_name), exist_ok=True)
    
    if arg_out != "":
        output_file = os.path.join("output", template_name, f"{arg_out}.png")
    else:
        timestamp = datetime.now().strftime("%y%m%d%H%M%S")
        output_file = os.path.join("output", template_name, f"{template_name}_{timestamp}.png")

    if arg_moduleout == True:
        temp_path = os.path.join("output", template_name, "module_output")
        os.makedirs(temp_path, exist_ok=True)

    if not temp_path and not output_file:
        print("Error: Makes no sense to run without either output image or module output", file=sys.stderr)
        print(usage_msg)
        sys.exit(1)

    template_file = os.path.join("templates", f"{template_name}.json")

    # Load the preset JSON
    if not os.path.isfile(template_file):
        print(f"Error: Template file not found: {template_file}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(template_file, "r") as f:
            preset = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{template_file}' was not found. Please check the file path.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON in '{template_file}'.", file=sys.stderr)
        print(f"Details: {e.msg} at line {e.lineno}, column {e.colno}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error while reading '{template_file}': {e}", file=sys.stderr)
        sys.exit(1)

    print("------------------------------------------------------------------o Initialization")
    # Load available modules from plugins
    instances = load_plugins()
    modules = {}
    for inst in instances:
        modules[inst.type] = inst

    # Extract overall map properties
    map_width = preset.get("width", 256)
    map_height = preset.get("height", 256)

    # Create a random number generator
    rng = random.Random()
    rng.seed(time.time_ns())

    # Apply processing chain
    print("------------------------------------------------------------------o Generation")
    try:
        jobs = create_execution_order(modules, preset.get("modules", []))
    except Exception as e:
        print(f'Error creating execution order: {e}', file=sys.stderr)
        sys.exit(1)
    outputs = {}
    for cfg in jobs:
        module = None
        try:
            print(f'--- Running module \'{cfg["type"]}\'...')
            module = modules[cfg["type"]]
        
            inputs = {}    
            for i in module.inputs:
                if cfg[i] not in outputs:
                    raise Exception(f'Missing input \'{cfg[i]}\'')
                inputs[i] = outputs[cfg[i]]
            
            settings = {}
            for s in module.settings:
                if s not in cfg:
                    print(f'  (i) Using default value for {s} = {module.defaults[s]}')
                    settings[s] = module.defaults[s]
                else:
                    settings[s] = cfg[s]

            start_time = time.time()
            
            results = module.apply(map_width, map_height, settings, inputs, rng)
            for key in results.keys():
                print(f'  -> {cfg[key]}')
                outputs[cfg[key]] = results[key]
            
            end_time = time.time()
            print(f"  Executed in ({end_time - start_time:.4f} sec.)")

            if temp_path != "":
                for key in results.keys():
                    print("  Saving module output...")
                    save_image(os.path.join(temp_path, f'raw_{cfg[key]}.png'), outputs[cfg[key]], False)
                    save_image(os.path.join(temp_path, f'norm_{cfg[key]}.png'), outputs[cfg[key]], True)

        except Exception as e:
            print(f'  Error executing module: {e}', file=sys.stderr)
            sys.exit(1)

    print("------------------------------------------------------------------o Finalization")
    if output_file != "":
        # Normalize and export as 8-bit grayscale
        if "final" not in outputs:
            print("No module was named 'final' -> No result produced")
            return
        
        highest_peak = outputs["final"].max()
        save_image(output_file, outputs["final"], True, False)
        print(f"Saved heightmap to {output_file}")
        print(f'Ideally imported with "highest peak" set to {int(highest_peak)} at {map_width}x{map_height}')

if __name__ == "__main__":
    print("WorldStack Generator v0.1.0")
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total generation time: {end_time - start_time:.6f} seconds")