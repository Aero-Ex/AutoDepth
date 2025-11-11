import importlib
import warnings
import bpy
import subprocess
import sys
import os
from pathlib import Path
import shlex
from .ascii_strings import depth_anything, installation_fin
import time

def draw(layout,depthgenius):
    col = layout.row()
    col.active = not depthgenius.installation_in_progress
    col.enabled = not depthgenius.installation_in_progress
    col.prop(depthgenius, "device", expand=True)
    status = register(depthgenius.device)
    if status == 'SUCCESS':
        row = layout.row()
        row.label(text="Dependencies are installed, nothing to do here!")
        return True
    elif status == 'ERR:CUDA':
        target = get_install_folder(f"venv_depthanything_gpu")
        if depthgenius.device == 'gpu' and target.exists() and target.is_dir():
            box = layout.box()
            box.label(text="Blender restart required for changes to take effect.")
            box.label(text="Currently using cpu")
            return False
        else:
            box = layout.box()
            labels = box.column()
            labels.label(text="Dependencies need to be installed,")
            labels.label(text="please press the button:")
            
            operators = box.column()
            operators.scale_y = 2.0
            operators.operator("depthgenius.install_dependencies",)
            return False
    else:
        box = layout.box()
        labels = box.column()
        labels.label(text="Dependencies need to be installed,")
        labels.label(text="please press the button:")
        
        operators = box.column()
        operators.scale_y = 2.0
        operators.operator("depthgenius.install_dependencies",)
        return False


def create_ascii_art(art_string):
    platform = sys.platform.lower()
    
    if platform.startswith(('linux', 'darwin', 'freebsd', 'openbsd', 'netbsd')):  # Unix-like systems
        start_delimiter = "cat << EOF"
        end_delimiter = "EOF"
        process_line = lambda l: l
    elif platform.startswith('win'):  # Windows
        start_delimiter = "@echo off"
        end_delimiter = ""
        def process_line(l):
            l = l.replace("^", "^^").replace("<", "^<").replace(">", "^>").replace("&", "^&").replace("|", "^|")
            return f"echo {l}" if l.strip() else "echo."
    else:
        raise OSError(f"Unsupported operating system: {platform}")

    lines = [start_delimiter]
    lines.extend(process_line(line) for line in art_string.split('\n'))
    if end_delimiter:
        lines.append(end_delimiter)
    
    return "\n".join(lines)

class DEPTHGENIUS_OT_InstallDependencies(bpy.types.Operator):
    bl_idname = "depthgenius.install_dependencies"
    bl_label = "Install Dependencies"
    bl_description = "Install required dependencies for DepthGenius"

    _timer = None
    _process = None
    _start_time = None

    @classmethod
    def poll(cls, context):
        return not context.scene.depthgenius.installation_in_progress
    
    def execute(self, context):
        try:
            device = context.scene.depthgenius.device
            dependencies_dir = Path(bpy.context.preferences.addons[__package__].preferences.dependencies_path)
            # Set up the virtual environment path
            venv_name = f"venv_depthanything_{device}"
            venv_path = dependencies_dir / venv_name

            # Determine the correct Python executable and activation script
            if sys.platform == "win32":
                python_exe = str(venv_path / "Scripts" / "python.exe")
                activate_cmd = f'call "{venv_path / "Scripts" / "activate.bat"}"'
            else:  # macOS and Linux
                python_exe = str(venv_path / "bin" / "python")
                activate_cmd = f'source "{venv_path / "bin" / "activate"}"'

            marker_file_path = dependencies_dir / f"installation_complete_{device}.txt"
            marker_creation_command = f'"{python_exe}" -c "import pathlib; pathlib.Path(r\'{marker_file_path}\').write_text(\'Installation completed successfully\')"'

            # Prepare the commands with proper quoting
            commands = [
                f'"{sys.executable}" -m venv "{venv_path}"',
                "echo Activating virtual environment...",
                activate_cmd,
                "echo Installing LIGHT-THE-TORCH...",
                "echo      ",
                f'"{python_exe}" -m pip install light-the-torch',
                f'"{python_exe}" -m light_the_torch install {"--cpuonly " if device=="cpu" else ""}--no-cache-dir torch torchvision opencv-python',
                marker_creation_command,
                create_ascii_art(installation_fin),
            ]

            # Create a shell script to run the commands
            if sys.platform == "win32":
                script_ext = ".bat"
                script_content = "\n".join([
                    create_ascii_art(depth_anything),
                    "@echo off",
                    "echo Initializing Virtual Environment for installing dependencies...",
                    *commands,
                    "exit",
                ])
            else:  # macOS and Linux
                script_ext = ".sh"
                script_content = "\n".join([
                    "#!/bin/bash",
                    "echo Initializing Virtual Environment for installing dependencies...",
                    *commands,
                    "exit",
                ])

            script_path = dependencies_dir / f"install_packages_{device}{script_ext}"
            with open(script_path, "w") as f:
                f.write(script_content)

            # Make the script executable on macOS and Linux
            if sys.platform != "win32":
                os.chmod(script_path, 0o755)

            context.scene.depthgenius.installation_in_progress = True
            # Run the script in a new terminal window
            if sys.platform == "win32":
                self._process = subprocess.Popen(["start", "cmd", "/k", str(script_path)], shell=True)
            elif sys.platform == "darwin":  # macOS
                apple_script = f'tell application "Terminal" to do script "bash {shlex.quote(str(script_path))}"'
                self._process = subprocess.Popen(["osascript", "-e", apple_script])
            else:  # Linux
                self._process = subprocess.Popen(["x-terminal-emulator", "-e", f"bash {shlex.quote(str(script_path))}"])


            self._start_time = time.time()
            self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
            context.window_manager.modal_handler_add(self)

            return {'RUNNING_MODAL'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start installation process: {str(e)}")
            return {'CANCELLED'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            device = context.scene.depthgenius.device
            dependencies_dir = Path(bpy.context.preferences.addons[__package__].preferences.dependencies_path)
            # Check for a file that indicates successful installation
            installation_complete_marker = dependencies_dir / f"installation_complete_{device}.txt"
            if installation_complete_marker.exists():
                self.report({'INFO'}, "Installation completed successfully.")
                self.add_venv_to_path(context)
                context.window_manager.event_timer_remove(self._timer)
                context.scene.depthgenius.installation_in_progress = False
                context.area.tag_redraw()
                return {'FINISHED'}
            else:
                # Process is still running or hasn't completed successfully
                elapsed_time = time.time() - self._start_time
                self.report({'INFO'}, f"Installation in progress... (Elapsed time: {elapsed_time:.2f} seconds)")
        return {'PASS_THROUGH'}


    def add_venv_to_path(self, context):
        device = context.scene.depthgenius.device
        dependencies_dir = Path(bpy.context.preferences.addons[__package__].preferences.dependencies_path)
        venv_path = dependencies_dir / f"venv_depthanything_{device}"
        
        if sys.platform == "win32":
            site_packages = venv_path / "Lib" / "site-packages"
        else:  # macOS and Linux
            site_packages = venv_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        
        if site_packages.exists():
            if str(site_packages) not in sys.path:
                sys.path.append(str(site_packages))
                print("Added to Path")
                self.report({'INFO'}, f"Added {site_packages} to sys.path")
            else:
                print("Already Added to Path")
                self.report({'INFO'}, f"{site_packages} is already in sys.path")
        else:
            print("Could not find site-packages")
            self.report({'WARNING'}, f"Could not find {site_packages}")

    def cancel(self, context):
        context.scene.depthgenius.installation_in_progress = False
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self.report({'INFO'}, "Installation process was cancelled.")

def get_install_folder(internal_folder):
    return Path(bpy.context.preferences.addons[__package__].preferences.dependencies_path) / internal_folder

def ensure_package_path(device='cpu', target = None):
    # Add the python path to the dependencies dir if missing
    gpu_dependencies_exists = False
    target_provided = target is not None
    target = Path(str(target))
    if target_provided: #when starting blender registering phase no accesss to Context. we write in settings.json
        target = target/"venv_depthanything_gpu"
    else:
        target = get_install_folder(f"venv_depthanything_gpu")

    if target.exists() and target.is_dir():
        gpu_dependencies_exists = True
    
    if device == 'cpu' and not gpu_dependencies_exists:
        if target_provided:
            target = target/"venv_depthanything_cpu"
        else:
            target = get_install_folder(f"venv_depthanything_cpu")

    if sys.platform == "win32":
        site_packages = target / "Lib" / "site-packages"
    else:  # macOS and Linux
        site_packages = target / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    
    if site_packages.exists():
        if device == 'gpu':
            # remove venv_cpu paths if device gpu before appending venv_gpu paths
            for path in sys.path:
                if "venv_depthanything_cpu" in path:
                    sys.path.remove(path)

        if str(site_packages) not in sys.path:
            print('DepthGenius: Found missing deps path in sys.path, appending...')
            print(str(site_packages))
            sys.path.append(str(site_packages))
            print('DepthGenius: Deps path has been appended to sys.path')
        else:
            pass
            # print("DepthGenius: Already Added to Path")
    else:
        print("DepthGenius: Could not find site-packages")

def test_packages(device='cpu'):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            import torch
            import cv2
            if device == 'gpu':
                cuda = torch.cuda.is_available()
                if not cuda:
                    return False,"ERR:CUDA"
            del torch
            del cv2
    except ImportError as e:
        print('DepthGenius: An ImportError occurred when importing the dependencies')
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        return False, "ERR:IMPORT"
    except Exception as e:
        print('DepthGenius: Something went very wrong importing the dependencies, please get that checked')
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        return False, "ERR:GENERAL"
    else:
        return True, "SUCCESS"

def register(device='cpu'):
    ensure_package_path(device)
    result,status = test_packages(device)
    if result:
        return status
    else:
        print("DepthGenius: Some dependencies are not installed, please install them using the button in the Preferences.")
        return status

def unregister(device='cpu'):
    target = get_install_folder(f"venv_depthanything_{device}")
    if sys.platform == "win32":
        site_packages = target / "Lib" / "site-packages"
    else:  # macOS and Linux
        site_packages = target / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if site_packages.exists():
        if str(site_packages) in sys.path:
            print('DepthGenius: Found deps path in sys.path, removing...')
            print(str(target))
            sys.path.remove(str(site_packages))
    result,status = test_packages(device)
    if result:
        return {status}
    else:
        print("DepthGenius: Some dependencies are not installed, please install them using the button in the Preferences.")
        return {status}
    
