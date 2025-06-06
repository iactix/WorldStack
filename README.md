# WorldStack
**A node-based heightmap generator**

WorldStack is a two-part system:

- **WorldStack Editor** – lets you design generator templates using a visual node interface
- **WorldStack Generator** – uses those templates to create actual heightmaps via command line

Templates describe how to generate terrain — combining noise, shapes, images, and filters to define a way of generating specific worlds. Games like OpenTTD understand the resulting 'heightmaps' and can generate actual game maps based on them.

## Installation

1. **Install Python**  
   Please do a websearch how to do this, it is just a matter of downloading and installing. Make sure to check **"Add Python to PATH"** during installation.

2. **Download or clone WorldStack**  

	I trust you find out how to do that. Just somehow get the content of this repository on your PC and we'll call that your installation folder.
	
3. **Install required packages**

   In your installation folder, run 'install.cmd' (it runs 'pip install -r requirements.txt') to install the python libraries required by WorldStack.
   
   These libraries (see the content of requirements.txt) are safe to the best of my knowledge, but I can not guarantee this and you're installing and using them at your own risk.

4. **Run the editor**

	Run editor.cmd, this just executes 'python ws-edit.py'. Once the editor opens just press "Help" and you will be guided from there.
	
5. **Run the generator**
	
	Open a console in your installation folder and run 'python ws-generator.py' for usage instructions

## Feedback

Bugs, ideas, and suggestions are welcome — Please open an issue on GitHub.
