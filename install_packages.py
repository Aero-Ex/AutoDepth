import warnings
import bpy
import subprocess
import sys
import os
import time
from pathlib import Path
import shlex

from .ascii_strings import depth_anything, installation_fin


def draw(layout, depthgenius):
    """Draw the installation UI - checks dependencies when UI is shown"""
    col = layout.row()
    col.active = not depthgenius.installation_in_progress
    col.enabled = not depthgenius.installation_in_progress
    col.prop(depthgenius, "device", expand=True)

    # Show CPU offload option for GPU mode
    if depthgenius.device == 'gpu':
        offload_row = layout.row()
        offload_row.prop(depthgenius, "enable_cpu_offload", text="Enable CPU Offloading (prevents OOM)")
        offload_row.active = not depthgenius.installation_in_progress
        offload_row.enabled = not depthgenius.installation_in_progress

    # Check if dependencies exist (file-based check, no imports!)
    deps_installed = check_dependencies_installed(depthgenius.device)

    if deps_installed:
        row = layout.row()
        row.label(text="Dependencies are installed!", icon='CHECKMARK')

        # Test button to verify they work
        test_row = layout.row()
        test_row.operator("depthgenius.test_dependencies", text="Test Dependencies")
        return True
    else:
        box = layout.box()
        labels = box.column()
        labels.label(text="Dependencies need to be installed", icon='ERROR')
        labels.label(text="Click below to install PyTorch + OpenCV")

        if sys.platform == 'win32':
            labels.label(text="⚠ Windows: Requires Visual C++ Redistributable", icon='INFO')

        operators = box.column()
        operators.scale_y = 2.0
        operators.operator("depthgenius.install_dependencies")
        return False


def check_dependencies_installed(device='cpu'):
    """Check if dependency folder exists (no imports!)"""
    try:
        target = get_install_folder(f"venv_depthanything_{device}")

        if sys.platform == "win32":
            site_packages = target / "Lib" / "site-packages"
        else:
            site_packages = target / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

        # Check if torch directory exists
        torch_dir = site_packages / "torch"
        cv2_dir = site_packages / "cv2"

        return torch_dir.exists() and cv2_dir.exists()
    except:
        return False


def create_ascii_art(art_string):
    """Create platform-specific ASCII art command"""
    platform = sys.platform.lower()

    if platform.startswith(('linux', 'darwin', 'freebsd', 'openbsd', 'netbsd')):
        # Unix-like systems
        start_delimiter = "cat << EOF"
        end_delimiter = "EOF"
        process_line = lambda l: l
    elif platform.startswith('win'):
        # Windows
        start_delimiter = "@echo off"
        end_delimiter = ""

        def process_line(l):
            l = l.replace("^", "^^").replace("<", "^<").replace(">", "^>")
            l = l.replace("&", "^&").replace("|", "^|")
            return f"echo {l}" if l.strip() else "echo."
    else:
        raise OSError(f"Unsupported operating system: {platform}")

    lines = [start_delimiter]
    lines.extend(process_line(line) for line in art_string.split('\n'))
    if end_delimiter:
        lines.append(end_delimiter)

    return "\n".join(lines)


class DEPTHGENIUS_OT_TestDependencies(bpy.types.Operator):
    """Test if dependencies work correctly"""
    bl_idname = "depthgenius.test_dependencies"
    bl_label = "Test Dependencies"
    bl_description = "Test if PyTorch and OpenCV can be imported"

    def execute(self, context):
        device = context.scene.depthgenius.device

        # Add DLL directory for Windows
        if sys.platform == "win32":
            try:
                add_dll_directory(device)
            except Exception as e:
                self.report({'WARNING'}, f"Could not add DLL directory: {e}")

        # Try to import
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                import torch
                import cv2

                if device == 'gpu':
                    cuda_available = torch.cuda.is_available()
                    if cuda_available:
                        self.report({'INFO'}, f"✓ Dependencies OK! PyTorch {torch.__version__}, CUDA available")
                    else:
                        self.report({'WARNING'}, "PyTorch installed but CUDA not available. Using CPU instead.")
                else:
                    self.report({'INFO'}, f"✓ Dependencies OK! PyTorch {torch.__version__} (CPU), OpenCV {cv2.__version__}")

                del torch
                del cv2
                return {'FINISHED'}

        except ImportError as e:
            self.report({'ERROR'}, f"Import Error: {str(e)}")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            if "DLL" in str(e) and sys.platform == 'win32':
                self.report({'ERROR'}, "Install Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe")
            return {'CANCELLED'}


class DEPTHGENIUS_OT_InstallDependencies(bpy.types.Operator):
    """Install required dependencies for TrueDepth addon"""
    bl_idname = "depthgenius.install_dependencies"
    bl_label = "Install Dependencies"
    bl_description = "Install required dependencies for TrueDepth (torch, torchvision, opencv)"

    _timer = None
    _process = None
    _start_time = None

    @classmethod
    def poll(cls, context):
        return not context.scene.depthgenius.installation_in_progress

    def execute(self, context):
        try:
            device = context.scene.depthgenius.device
            dependencies_dir = Path(
                bpy.context.preferences.addons[__package__].preferences.dependencies_path
            )
            dependencies_dir.mkdir(parents=True, exist_ok=True)

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
            marker_creation_command = (
                f'"{python_exe}" -c "import pathlib; '
                f'pathlib.Path(r\'{marker_file_path}\').write_text(\'Installation completed successfully\')"'
            )

            # Prepare the commands with proper quoting
            # Use PyTorch 2.1.0 instead of latest - known stable version on Windows
            if device == "cpu":
                torch_install_cmd = f'"{python_exe}" -m pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu'
            else:
                torch_install_cmd = f'"{python_exe}" -m pip install --no-cache-dir torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118'

            commands = [
                f'"{sys.executable}" -m venv "{venv_path}"',
                "echo Activating virtual environment...",
                activate_cmd,
                "echo Installing PyTorch 2.1.0 (stable, tested version)...",
                "echo      ",
                torch_install_cmd,
                f'"{python_exe}" -m pip install --no-cache-dir opencv-python',
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
                    "pause",
                    "exit",
                ])
            else:  # macOS and Linux
                script_ext = ".sh"
                script_content = "\n".join([
                    "#!/bin/bash",
                    "echo Initializing Virtual Environment for installing dependencies...",
                    *commands,
                    "read -p 'Press Enter to close...'",
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
                self._process = subprocess.Popen(
                    ["start", "cmd", "/k", str(script_path)],
                    shell=True
                )
            elif sys.platform == "darwin":  # macOS
                apple_script = (
                    f'tell application "Terminal" to do script '
                    f'"bash {shlex.quote(str(script_path))}"'
                )
                self._process = subprocess.Popen(["osascript", "-e", apple_script])
            else:  # Linux
                self._process = subprocess.Popen([
                    "x-terminal-emulator", "-e",
                    f"bash {shlex.quote(str(script_path))}"
                ])

            self._start_time = time.time()
            self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
            context.window_manager.modal_handler_add(self)

            self.report({'INFO'}, "Installation process started in new terminal window...")
            return {'RUNNING_MODAL'}

        except Exception as e:
            import traceback
            print("TrueDepth: Failed to start installation process:")
            print(traceback.format_exc())
            self.report({'ERROR'}, f"Failed to start installation process: {str(e)}")
            context.scene.depthgenius.installation_in_progress = False
            return {'CANCELLED'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            device = context.scene.depthgenius.device
            dependencies_dir = Path(
                bpy.context.preferences.addons[__package__].preferences.dependencies_path
            )

            # Check for a file that indicates successful installation
            installation_complete_marker = dependencies_dir / f"installation_complete_{device}.txt"

            if installation_complete_marker.exists():
                self.report({'INFO'}, "Installation completed successfully! Restart Blender to use the addon.")
                self.add_venv_to_path(context)
                context.window_manager.event_timer_remove(self._timer)
                context.scene.depthgenius.installation_in_progress = False
                context.area.tag_redraw()
                return {'FINISHED'}
            else:
                # Process is still running or hasn't completed successfully
                elapsed_time = time.time() - self._start_time
                if elapsed_time % 10 < 0.5:  # Print every 10 seconds
                    print(f"TrueDepth: Installation in progress... ({elapsed_time:.0f}s elapsed)")

        return {'PASS_THROUGH'}

    def add_venv_to_path(self, context):
        """Add the virtual environment to sys.path"""
        device = context.scene.depthgenius.device
        ensure_package_path(device)
        self.report({'INFO'}, "Dependencies installed! Please restart Blender.")

    def cancel(self, context):
        """Cancel the installation process"""
        context.scene.depthgenius.installation_in_progress = False
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self.report({'INFO'}, "Installation process was cancelled.")


def get_install_folder(internal_folder):
    """Get the installation folder path"""
    return Path(
        bpy.context.preferences.addons[__package__].preferences.dependencies_path
    ) / internal_folder


def add_dll_directory(device='cpu'):
    """Add DLL directories for Windows to help load torch DLLs"""
    if sys.platform != "win32":
        return

    try:
        target = get_install_folder(f"venv_depthanything_{device}")

        # Add torch lib directory
        torch_lib = target / "Lib" / "site-packages" / "torch" / "lib"
        if torch_lib.exists() and hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(str(torch_lib))
                print(f"TrueDepth: Added DLL directory: {torch_lib}")
            except Exception as e:
                print(f"TrueDepth: Could not add DLL directory: {e}")
    except Exception as e:
        print(f"TrueDepth: Error in add_dll_directory: {e}")


def ensure_package_path(device='cpu', target=None):
    """Ensure the package path is in sys.path - DOES NOT IMPORT ANYTHING"""
    gpu_dependencies_exists = False
    target_provided = target is not None
    target = Path(str(target)) if target else None

    if target_provided:
        # When starting blender registering phase no access to Context
        target = target / "venv_depthanything_gpu"
    else:
        target = get_install_folder(f"venv_depthanything_gpu")

    if target.exists() and target.is_dir():
        gpu_dependencies_exists = True

    if device == 'cpu' and not gpu_dependencies_exists:
        if target_provided:
            target = target.parent / "venv_depthanything_cpu"
        else:
            target = get_install_folder(f"venv_depthanything_cpu")

    if sys.platform == "win32":
        site_packages = target / "Lib" / "site-packages"
    else:  # macOS and Linux
        site_packages = target / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

    if site_packages.exists():
        if device == 'gpu':
            # Remove venv_cpu paths if device gpu before appending venv_gpu paths
            for path in list(sys.path):
                if "venv_depthanything_cpu" in path:
                    sys.path.remove(path)

        if str(site_packages) not in sys.path:
            print(f'TrueDepth: Adding deps path to sys.path: {site_packages}')
            sys.path.append(str(site_packages))

        # On Windows, add DLL directory
        if sys.platform == "win32":
            add_dll_directory(device)
    else:
        print(f"TrueDepth: Could not find site-packages at {site_packages}")


def register(device='cpu'):
    """Register - only adds to sys.path, does NOT test imports"""
    ensure_package_path(device)
    # Return success without testing - testing happens when user clicks test button
    return "READY"


def unregister(device='cpu'):
    """Unregister packages (remove from sys.path)"""
    target = get_install_folder(f"venv_depthanything_{device}")

    if sys.platform == "win32":
        site_packages = target / "Lib" / "site-packages"
    else:  # macOS and Linux
        site_packages = target / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

    if site_packages.exists():
        if str(site_packages) in sys.path:
            print(f'TrueDepth: Removing deps path from sys.path: {site_packages}')
            sys.path.remove(str(site_packages))
