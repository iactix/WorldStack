import builtins

class GeneratorModule:
    # ______________
    # Plugin Toolset

    # To be used in init(), this module will be executed if some model configuration's type setting matches this. Must set a category.
    def set_type(self, type, category):
        self.type = type
        self.category = category

    # To be used in init(), makes a setting from the module configuration available. Enforces documentation for that setting.
    # Setting types "input" and "output" make them behave as such. Outputs make image data available under the specified name and inputs receive data with that name
    # The setting's actual data type is derived frm the default value, supporting strings and numbers and json.
    def create_setting(self, setting_name, default_value, documentation, type = ""):
        if not documentation:
            raise ValueError(f'Failed to create module setting \'{setting_name}\' without proper documentation')
        
        doc = {}
        doc["name"] = setting_name
        doc["default"] = default_value
        doc["type"] = builtins.type(default_value).__name__
        doc["doc"] = documentation

        if type == "input":
            self.inputs.append(setting_name)
            self.doc["inputs"].append(doc)
        elif type == "output":
            self.outputs.append(setting_name)
            self.doc["outputs"].append(doc)
        else:
            self.settings.append(setting_name)
            self.doc["settings"].append(doc)
        
        self.defaults[setting_name] = default_value
        self.types[setting_name] = doc["type"]

    # _____________________
    # Plugin Implementation

    # Called on construction, the only correct place to call set_type and create_setting
    # Must return helptext for the module as a whole
    def init(self):
        raise NotImplementedError("Module must implement init().")    
    
    # Main module implementation, receives map dimensions, specified settings and inputs in dictionaries, must return specified outputs in dictionary
    def apply(self, map_width, map_height, settings, inputs, rng):
        raise NotImplementedError("Module must implement apply().")
    
    # ______________________
    # Internal functionality

    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.settings = []
        self.defaults = {}
        self.types = {}
        self.doc = {}
        self.doc["inputs"] = []
        self.doc["outputs"] = []
        self.doc["settings"] = []
        self.doc["doc"] = self.init()
        
        if not self.doc["doc"]:
            raise ValueError("Module init() is required to return general module documentation")

    def get_setting(self, setting_name):
        return self.config.get(setting_name, None)
    
    def get_documentation(self):
        self.doc["type"] = self.type
        self.doc["category"] = self.category
        return self.doc