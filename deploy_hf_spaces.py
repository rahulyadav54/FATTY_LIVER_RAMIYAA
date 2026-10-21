#!/usr/bin/env python
"""
Automated Hugging Face Spaces Deployment Helper
This script helps you deploy the Fatty Liver Classification app to HF Spaces
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    print("\n" + "="*60)
    print(f"🚀 {text}")
    print("="*60 + "\n")

def print_step(num, text):
    print(f"Step {num}: {text}")

def run_command(cmd, description=""):
    """Run a shell command"""
    if description:
        print(f"  → {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ❌ Error: {result.stderr}")
            return False
        print(f"  ✅ Success")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    print_header("Fatty Liver Classification - HF Spaces Deployment")
    
    print("""
    This script will help you deploy to Hugging Face Spaces.
    You'll need:
    ✓ Hugging Face account (free at huggingface.co)
    ✓ HF token (from settings/tokens)
    ✓ Git installed
    """)
    
    # Check prerequisites
    print_step(1, "Checking prerequisites...")
    
    # Check git
    if run_command("git --version", "Checking Git installation"):
        print("  ✅ Git is installed")
    else:
        print("  ⚠️  Please install Git from https://git-scm.com")
        return
    
    # Check files
    print_step(2, "Checking required files...")
    required_files = [
        'app_streamlit.py',
        'best_model.pth',
        'models/siamese_net.py',
        'src/data_loader.py',
        'requirements_streamlit.txt'
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ Missing: {file}")
            return
    
    print_step(3, "Getting HF credentials...")
    hf_username = input("  Enter your Hugging Face username: ").strip()
    space_name = input("  Enter Space name (default: fatty-liver-classification): ").strip() or "fatty-liver-classification"
    hf_token = input("  Enter your HF token (from huggingface.co/settings/tokens): ").strip()
    
    if not hf_username or not hf_token:
        print("  ❌ HF credentials required")
        return
    
    print_step(4, "Creating local clone of your Space...")
    space_repo = f"https://{hf_username}:{hf_token}@huggingface.co/spaces/{hf_username}/{space_name}"
    clone_dir = f"./{space_name}_space"
    
    if Path(clone_dir).exists():
        print(f"  ℹ️  Directory {clone_dir} already exists, updating...")
        os.chdir(clone_dir)
        run_command("git pull", "Pulling latest changes")
        os.chdir("..")
    else:
        if not run_command(f"git clone {space_repo} {clone_dir}", "Cloning Space repository"):
            print("  ❌ Failed to clone. Ensure:")
            print("     - Space exists on HF")
            print("     - Token has 'write' permissions")
            print("     - Username is correct")
            return
    
    print_step(5, "Copying files to Space...")
    os.chdir(clone_dir)
    
    # Configure Space metadata README for Docker SDK
    hf_space_readme = f"""---
title: Fatty Liver Detection System
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Fatty Liver Detection System - Professional Medical AI Dashboard

AI-assisted classification of liver ultrasound scans using Siamese Neural Networks with ResNet-50 encoder.
"""
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(hf_space_readme)
    print("  ✅ Configured Hugging Face Spaces Docker SDK (README.md)")

    # Copy files
    files_to_copy = {
        '../app.py': 'app.py',
        '../best_model.pth': 'best_model.pth',
        '../models': 'models',
        '../src': 'src',
        '../templates': 'templates',
        '../static': 'static',
        '../requirements.txt': 'requirements.txt',
        '../Dockerfile': 'Dockerfile',
        '../.dockerignore': '.dockerignore',
    }
    
    import shutil
    for src, dst in files_to_copy.items():
        try:
            if Path(src).is_file():
                shutil.copy(src, dst)
                print(f"  ✅ Copied {src} → {dst}")
            elif Path(src).is_dir():
                if Path(dst).exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                print(f"  ✅ Copied directory {src} → {dst}")
        except Exception as e:
            print(f"  ❌ Error copying {src}: {e}")
    
    print_step(6, "Pushing to Hugging Face...")
    
    if run_command("git add .", "Staging files"):
        pass
    else:
        print("  ⚠️  Some files couldn't be staged")
    
    if run_command('git commit -m "Deploy Fatty Liver Classification app"', "Creating commit"):
        pass
    else:
        print("  ⚠️  Commit might be empty (files already exist)")
    
    if run_command("git push", "Pushing to Hugging Face Spaces"):
        pass
    else:
        print("  ❌ Push failed")
        return
    
    os.chdir("..")
    
    # Success
    print_header("✅ Deployment Successful!")
    
    print(f"""
    Your app is now deploying to:
    
    🔗 https://huggingface.co/spaces/{hf_username}/{space_name}
    
    📋 What to do next:
    1. Visit the URL above
    2. Wait 2-3 minutes for initial build
    3. First load may take 1-2 minutes (model loading)
    4. Upload an ultrasound image to test
    
    ⚡ Tips:
    - App will auto-update when you push changes
    - Models run locally (no data privacy concerns)
    - You can make Space private/public in settings
    
    🎉 Enjoy your deployed app!
    """)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Deployment cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
