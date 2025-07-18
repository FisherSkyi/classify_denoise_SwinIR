import subprocess
import sys
import pkg_resources
import platform
import os

def check_imagemagick():
    """Check if ImageMagick is installed on the system."""
    try:
        result = subprocess.run(['convert', '-version'], capture_output=True, text=True)
        if 'ImageMagick' in result.stdout:
            print("ImageMagick is already installed.")
            return True
        return False
    except FileNotFoundError:
        return False

def install_imagemagick():
    """Provide instructions to install ImageMagick based on the operating system."""
    system = platform.system().lower()
    print("ImageMagick is required for the Wand package but is not installed.")
    if system == "darwin":  # macOS
        print("For macOS, install ImageMagick using Homebrew:")
        print("1. Install Homebrew if not already installed: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        print("2. Run: brew install imagemagick")
    elif system == "linux":
        print("For Linux (Ubuntu/Debian), install ImageMagick using apt:")
        print("Run: sudo apt-get update && sudo apt-get install imagemagick")
    elif system == "windows":
        print("For Windows, download and install ImageMagick from:")
        print("https://imagemagick.org/script/download.php#windows")
        print("Ensure the ImageMagick 'convert' command is added to your PATH.")
    else:
        print("Unsupported OS. Please install ImageMagick manually from https://imagemagick.org.")
    print("After installing ImageMagick, rerun this script.")
    sys.exit(1)

def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("Checking and installing Python dependencies...")
    
    # Read requirements.txt
    try:
        with open('requirements.txt', 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print("Error: requirements.txt not found in the current directory.")
        sys.exit(1)

    # Check installed packages
    installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    # Install missing or outdated packages
    for req in requirements:
        try:
            pkg_name = req.split('==')[0]
            if pkg_name not in installed:
                print(f"Installing {req}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])
            else:
                print(f"{pkg_name} is already installed.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {req}: {e}")
            sys.exit(1)

def main():
    # Check for ImageMagick (required for Wand)
    if not check_imagemagick():
        install_imagemagick()

    # Install Python dependencies
    install_dependencies()
    print("All dependencies installed successfully!")

if __name__ == "__main__":
    main()
